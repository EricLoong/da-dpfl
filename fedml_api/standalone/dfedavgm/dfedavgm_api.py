import copy
import logging
import pickle
import random
import pdb
import numpy as np
import networkx as nx

from fedml_api.standalone.dfedavgm.client import Client


class DFedAvgMAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )
        self.init_stat_info()

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
                self.logger,
            )
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        w_init = self.model_trainer.get_model_params()
        w_per_mdls = []
        w_per_mdl_updates = []
        # Initialize the weights for each client
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_init))
            w_per_mdl_updates.append(copy.deepcopy(self.zero_weights(w_init)))
        # device = {device} cuda:0apply mask to init weights


        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm

            comm_round_iterable = tqdm(
                range(self.args.comm_round), desc="Comm. Rounds", ncols=100
            )
        # Example: Get neighbors of client 5
        for round_idx in comm_round_iterable:
            self.logger.info(
                "################Communication round : {}".format(round_idx)
            )
            w_locals = []
            # This holds the updates from the neighbors of the client
            w_per_mdl_updates_temp = copy.deepcopy(w_per_mdl_updates)
            active_ths_rnd = np.random.choice(
                [0, 1],
                size=self.args.client_num_in_total,
                p=[1.0 - self.args.active, self.args.active],
            )
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info(
                    "@@@@@@@@@@@@@@@@ Training Client CM({}): {}".format(
                        round_idx, clnt_idx
                    )
                )
                client = self.client_list[clnt_idx]
                if active_ths_rnd[clnt_idx] == 0:
                    nei_indexs = np.array([])
                else:
                    nei_indexs = self._topology_choose(
                        current_clnt=clnt_idx,
                        client_num_in_total=self.args.client_num_in_total,
                        client_num_per_round=self.args.client_num_per_round,
                        cs=self.args.cs,
                        active_ths_rnd=active_ths_rnd,
                    )
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)

                client_indexes = np.sort(nei_indexs)

                self.logger.info("client_indexes = " + str(client_indexes))
                w_updates_for_this_client = [
                    (self.client_list[i].get_sample_number(), w_per_mdl_updates[i])
                    for i in client_indexes
                ]
                # aggregate the updates from the neighbors, which is float16 in this case. (non-Quantized is float 32)
                w_updates_agg_for_this_client = self._aggregate(
                    w_updates_for_this_client
                )
                local_mdl_before_training = copy.deepcopy(w_per_mdls[clnt_idx])
                (
                    qgrad_per_clnt_idx,
                    w_per_clnt_idx,
                    training_flops,
                    num_comm_params,
                ) = client.train(
                    copy.deepcopy(
                        self.local_agg(
                            local_mdl_before_training, w_updates_agg_for_this_client
                        )
                    ),
                    round_idx,
                )
                w_per_mdls[clnt_idx] = copy.deepcopy(w_per_clnt_idx)
                w_per_mdl_updates_temp[clnt_idx] = copy.deepcopy(qgrad_per_clnt_idx)
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            w_per_mdl_updates = copy.deepcopy(w_per_mdl_updates_temp)
            w_per_mdls_with_num = [
                (self.client_list[i].get_sample_number(), w_per_mdls[i])
                for i in range(self.args.client_num_in_total)
            ]
            w_global = self._aggregate(w_per_mdls_with_num)
            self._test_on_all_clients(w_global, w_per_mdls, round_idx)

    def _generate_expander_graph(self, N, N_selected):
        G = nx.Graph()

        # Add nodes to the graph
        nodes = list(range(N))
        G.add_nodes_from(nodes)

        # Connect each node to N_selected neighbors in a cyclic manner
        for node in nodes:
            for i in range(1, N_selected + 1):
                neighbor = (node + i) % N
                G.add_edge(node, neighbor)

        return G

    def _get_neighbors_of_client(self, G, client_idx):
        return list(G.neighbors(client_idx))

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global = {}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _topology_choose(
        self,
        current_clnt,
        client_num_in_total,
        client_num_per_round,
        cs=False,
        active_ths_rnd=None,
    ):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
            return client_indexes

        if (
            cs == "random"
        ):  ###### If the active client is still in the list of selected clients. Then select out the client.
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
            while current_clnt in client_indexes:
                client_indexes = np.random.choice(
                    range(client_num_in_total), num_clients, replace=False
                )
            return client_indexes

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (current_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (current_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])
            return client_indexes
        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd == 1)).squeeze()
            client_indexes = np.delete(
                client_indexes, int(np.where(client_indexes == current_clnt)[0])
            )
            return client_indexes
        elif cs == "cyclic":
            expander_graph = self._generate_expander_graph(
                N=int(self.args.client_num_in_total),
                N_selected=int(self.args.client_num_per_round / 2),
            )
            client_indexes = self._get_neighbors_of_client(expander_graph, current_clnt)
            return client_indexes


    def _test_on_all_clients(self, w_global, w_per_mdls, round_idx):

        self.logger.info(
            "################global_test_on_all_clients : {}".format(round_idx)
        )

        g_test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        p_test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(w_global, True)
            g_test_metrics["num_samples"].append(
                copy.deepcopy(g_test_local_metrics["test_total"])
            )
            g_test_metrics["num_correct"].append(
                copy.deepcopy(g_test_local_metrics["test_correct"])
            )
            g_test_metrics["losses"].append(
                copy.deepcopy(g_test_local_metrics["test_loss"])
            )

            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            p_test_metrics["num_samples"].append(
                copy.deepcopy(p_test_local_metrics["test_total"])
            )
            p_test_metrics["num_correct"].append(
                copy.deepcopy(p_test_local_metrics["test_correct"])
            )
            p_test_metrics["losses"].append(
                copy.deepcopy(p_test_local_metrics["test_loss"])
            )

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
        # test on test dataset
        g_test_acc = (
            sum(
                [
                    np.array(g_test_metrics["num_correct"][i])
                    / np.array(g_test_metrics["num_samples"][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )
        g_test_loss = (
            sum(
                [
                    np.array(g_test_metrics["losses"][i])
                    / np.array(g_test_metrics["num_samples"][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )

        p_test_acc = (
            sum(
                [
                    np.array(p_test_metrics["num_correct"][i])
                    / np.array(p_test_metrics["num_samples"][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )
        p_test_loss = (
            sum(
                [
                    np.array(p_test_metrics["losses"][i])
                    / np.array(p_test_metrics["num_samples"][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )

        stats = {"global_test_acc": g_test_acc, "global_test_loss": g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.logger.info(stats)

        stats = {"person_test_acc": p_test_acc, "person_test_loss": p_test_loss}
        self.stat_info["person_test_acc"].append(p_test_acc)
        self.logger.info(stats)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []

    def zero_weights(self, w):
        zeroed = {}
        for key, value in w.items():
            zeroed[key] = 0 * value
        return zeroed

    def local_agg(self, model_dict, updates):
        for key, value in updates.items():
            model_dict[key] += value
        return model_dict

    # def generate_mixing_matrix(self,G, N):
    #    mixing_matrix = np.zeros((N, N), dtype=int)

    #   for node in G.nodes():
    #        for neighbor in G.neighbors(node):
    #            mixing_matrix[node, neighbor] = 1

    #   return mixing_matrix


# Generate the mixing matrix for the graph
# mixing_matrix = generate_mixing_matrix(G, N)
# Visualize the graph
# nx.draw_circular(G, with_labels=True, node_size=500, node_color='skyblue')
# plt.show()
