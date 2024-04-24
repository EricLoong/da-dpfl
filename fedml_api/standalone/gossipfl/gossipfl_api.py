import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
from fedml_api.standalone.adpfl.sp_functions import conv_fc_condition
from fedml_api.standalone.gossipfl.client import Client


class GossipFLAPI(object):
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
        w_global = self.model_trainer.get_model_params()
        init_mask = self.generate_random_mask(
            seed=self.args.seed,
            sparsity=self.args.target_sparsity,
            model_dict=self.model_trainer.get_model_params(),
        )
        w_per_mdls = []
        w_per_compressed = []
        # w_per_mdls include all the weights of clients (dense model), which is a list of dictionaries.
        # w_per_compressed include all the weights of clients (compressed model, just for communication and will be recovered back),
        #  which is also a list of dictionaries.
        # Initialization
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global))
            w_per_compressed_temp = copy.deepcopy(w_global)
            for name in init_mask.keys():
                w_per_compressed_temp[name] = w_global[name] * init_mask[name]
            w_per_compressed.append(copy.deepcopy(w_per_compressed_temp))

        # Involve tqdm for better visualization
        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm

            comm_round_iterable = tqdm(
                range(self.args.comm_round), desc="Comm. Rounds", ncols=100
            )

        for round_idx in comm_round_iterable:
            self.logger.info(
                "################Communication round : {}".format(round_idx)
            )

            client_groups = self._client_random_match(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            # Generate random mask for each round, this should be generated in clients with the same seed,
            # as stated in GossipFL paper.
            global_mask_round_idx = self.generate_random_mask(
                seed=round_idx + self.args.seed,
                sparsity=self.args.target_sparsity,
                model_dict=self.model_trainer.get_model_params(),
            )
            global_mask_round_idx_next = self.generate_random_mask(
                seed=round_idx + 1 + self.args.seed,
                sparsity=self.args.target_sparsity,
                model_dict=self.model_trainer.get_model_params(),
            )
            # Mask generation for each client is only related to the seed, the model_dict is for the structure instruction.

            self.logger.info("client_indexes = " + str(client_groups))
            w_per_compressed_record = copy.deepcopy(w_per_compressed)
            w_per_mdls_record = copy.deepcopy(w_per_mdls)
            tst_results_ths_round = []
            for cur_clnt in range(self.args.client_num_in_total):
                self.logger.info(
                    "@@@@@@@@@@@@@@@@ Training Client CM({}): {}".format(
                        round_idx, cur_clnt
                    )
                )
                # update dataset
                client = self.client_list[cur_clnt]

                # update meta components in personal network
                neighbor_clients = self._detect_neighborhood(client_groups, cur_clnt)
                w_aggr_compress = self._aggregate_compressed_params(
                    w_per_compressed, neighbor_clients
                )

                # Each client trains its local dense model, and compresses it to a sparse model for communication.
                (
                    w_per,
                    w_per_send,
                    training_flops,
                    num_comm_params,
                    test_results,
                ) = client.train(
                    w_aggr=w_aggr_compress,
                    global_mask=global_mask_round_idx,
                    global_mask_next=global_mask_round_idx_next,
                    round=round_idx,
                    w_local_last_epoch=copy.deepcopy(w_per_mdls[cur_clnt]),
                )
                tst_results_ths_round.append(test_results)
                w_per_mdls_record[cur_clnt] = copy.deepcopy(w_per)
                w_per_compressed_record[cur_clnt] = copy.deepcopy(w_per_send)
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            # update communication parameters
            w_per_compressed = copy.deepcopy(w_per_compressed_record)
            w_per_mdls = copy.deepcopy(w_per_mdls_record)
            self._local_test_on_all_clients(tst_results_ths_round, round_idx)

    def _detect_neighborhood(self, clusters_group, client_idx):
        # return a list of neighbor clients
        neighbor_clients = []
        for cluster in clusters_group:
            if client_idx in cluster:
                neighbor_clients = cluster
                break
        return neighbor_clients

    def _client_random_match(
        self, round_idx, client_num_in_total, client_num_per_round
    ):
        # Generate a list of client indices
        clients = list(range(client_num_in_total))
        # Shuffle the client indices to ensure random distribution
        random.seed(round_idx)
        np.random.shuffle(clients)
        # Split the clients into clusters of size N_selected+1
        clusters = [
            clients[i : i + client_num_per_round + 1]
            for i in range(0, client_num_in_total, client_num_per_round + 1)
        ]
        # If the last cluster has fewer than N_selected+1 clients, distribute them to other clusters
        if len(clusters[-1]) < client_num_per_round + 1:
            extra_clients = clusters[-1]
            clusters = clusters[
                :-1
            ]  # Remove the last cluster to redistribute its clients
            for extra_client in extra_clients:
                # Randomly choose a cluster to add the extra client, avoiding repetition
                chosen_cluster = np.random.choice(len(clusters))
                clusters[chosen_cluster].append(extra_client)
        return clusters

    def generate_random_mask(self, seed, model_dict, sparsity=0.5):
        """
        Generate a random mask for each parameter in the model dictionary with a specific seed.

        Parameters:
        seed (int): The seed for random number generation to ensure reproducibility.
        model_dict (dict): A state dictionary of a DNN model.

        Returns:
        dict: A dictionary containing a random mask for each parameter in the model.
        """
        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # If using CUDA, uncomment the following line to ensure reproducibility across multiple GPU calls
        # torch.cuda.manual_seed_all(seed)

        # Create a random mask dictionary with the same structure (prunable) as the model's state_dict
        random_mask_dict = {}
        for key, tensor in model_dict.items():
            if conv_fc_condition(key):
                # Only consider convolutional and fully-connected layers
                # Generate a random mask with the same shape as the model's parameters
                # The mask will have values of 0 or 1. Adjust the probability as needed.
                random_mask = (torch.rand(tensor.size()) > sparsity).float()
                # Save the mask to the dictionary
                random_mask_dict[key] = random_mask

        return random_mask_dict

    def _aggregate_compressed_params(self, w_locals_compressed, neighbor_clients):
        """
        In GossipFL, they only adopt 2 clients as neighbors, and set Mixing matrix as 0.5 for them.
        So, we set Mixing matrix according to the random match results with mixing matrix value 1/len(neighbor_clients).
        :param w_locals_compressed: A list of all local models compressed by each client, sharing a global mask.
        :param neighbor_clients: The list of index neighbor clients
        :return: The aggregated model_compressed
        """
        # Initialize the aggregated model with zeros of the same shape as the first local model
        aggregated_model_compressed = {
            k: torch.zeros_like(v) for k, v in w_locals_compressed[0].items()
        }

        # Calculate the mixing matrix value based on the number of neighbor clients
        mixing_matrix_value = 1 / len(neighbor_clients)

        # Aggregate the compressed models from each neighbor
        for neighbor_idx in neighbor_clients:
            neighbor_model_compressed = w_locals_compressed[neighbor_idx]
            for k, v in neighbor_model_compressed.items():
                aggregated_model_compressed[k] += mixing_matrix_value * v

        return aggregated_model_compressed

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
        self.logger.info(
            "################local_test_on_all_clients after local training in communication round: {}".format(
                round_idx
            )
        )
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        for client_idx in range(self.args.client_num_in_total):
            # test data
            test_metrics["num_samples"].append(
                copy.deepcopy(tst_results_ths_round[client_idx]["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(tst_results_ths_round[client_idx]["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(tst_results_ths_round[client_idx]["test_loss"])
            )

            """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = (
            sum(
                [
                    test_metrics["num_correct"][i] / test_metrics["num_samples"][i]
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )
        test_loss = (
            sum(
                [
                    np.array(test_metrics["losses"][i])
                    / np.array(test_metrics["num_samples"][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )

        stats = {"test_acc": test_acc, "test_loss": test_loss}

        self.logger.info(stats)

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
