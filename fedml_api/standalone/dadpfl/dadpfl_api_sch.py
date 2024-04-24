import random

import torch
from fedml_api.standalone.dadpfl.client import client
import copy, math
import numpy as np
from torch.nn.functional import cosine_similarity

# Import the functions from adpfl/sp_functions.py such as SparsityIndex
from fedml_api.standalone.dadpfl.sp_functions import *
from tqdm import tqdm


class dadpflAPIsch(object):
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
            class_counts,
        ] = dataset
        # Check the property of datasets
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_counts = class_counts
        self.model_trainer = model_trainer
        # Status of basic information
        self.stat_info = {
            "label_num": self.class_counts,
            "sum_comm_params": 0,
            "sum_training_flops": 0,
            "avg_inference_flops": 0,
            "old_mask_test_acc": [],
            "new_mask_test_acc": [],
            "final_masks": [],
            "mask_dis_matrix": [],
        }
        self._setup_clients()

    def _setup_clients(self):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            user = client(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model_trainer,
                self.logger,
            )
            self.client_list.append(user)
        self.logger.info("############setup_clients (END)#############")

    def train(self):

        # We generate initial masks since each client can adjust its mask on his own. The local
        # masks set mask_local is to store the masks to compare, which are obtained after training
        # To make fair comparison, we keep the selection of clients the same as DisPFL.
        w_density = list(
            np.repeat(self.args.dense_ratio, self.args.client_num_in_total)
        )
        ###### After receiving sparsity, according to the policy to decide the initial masks
        # local_masks = []
        # for i in range(self.args.client_num_in_total):
        #    local_masks.append(self.model_trainer.init_masks_randin())
        params = self.model_trainer.get_model_params()
        if self.args.uniform:
            sparsities = self.model_trainer.calculate_sparsities(
                params, distribution="uniform", dense_ratio=self.args.dense_ratio
            )
        else:
            sparsities = self.model_trainer.calculate_sparsities(
                params, dense_ratio=self.args.dense_ratio
            )
        ###### After receiving sparsity, according to the policy to generate the initial masks

        # print('Mask initialization starts!')
        if not self.args.different_initial:
            local_masks = []
            for _ in range(self.args.client_num_in_total):
                temp = self.model_trainer.init_masks(params, sparsities)
                local_masks.append(copy.deepcopy(temp))
        else:
            local_masks = []
            mask_same_temp = self.model_trainer.init_masks(params, sparsities)
            for _ in range(self.args.client_num_in_total):
                local_masks.append(copy.deepcopy(mask_same_temp))
        # Generate list to store difference between weights updates.

        early_prune_stat_by_clients = []
        for i in range(self.args.client_num_in_total):
            early_prune_stat_by_clients.append([])
        w_init = self.model_trainer.get_model_params()
        w_per_mdls = []  ###### To store the personalized model as dictionary
        # Initialization
        for clnt_id in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_init))
            # updates_matrix.append(copy.deepcopy(w_global))
            for key in w_per_mdls[clnt_id]:
                if conv_fc_condition(key):
                    w_per_mdls[clnt_id][key] = w_init[key] * local_masks[clnt_id][key]
                # updates_matrix[clnt_id][key] = updates_matrix[clnt_id][key] - updates_matrix[clnt_id][key]
        w_init_models = copy.deepcopy(w_per_mdls)
        early_prune_flag = False
        early_prune_record = []
        # This is for update models for each iteration and finally update the w_per_mdls after selected clients finish training
        # This is for communication round updating
        for round_idx in range(self.args.comm_round):
            self.logger.info(
                "################Communication round : {}".format(round_idx)
            )
            # print('Current communication round is:', round_idx)
            # print('Keep current round personalized models')
            w_per_mdls_local = copy.deepcopy(w_per_mdls)
            reuse_index = list(range(self.args.client_num_in_total))
            random.shuffle(reuse_index)
            active_ths_rnd = np.random.choice(
                [0, 1],
                size=self.args.client_num_in_total,
                p=[1.0 - self.args.active, self.args.active],
            )
            # w_per_mdls_temp = copy.deepcopy(w_per_mdls)

            tst_results_ths_round = []
            final_tst_results_ths_round = []
            threshold_list = []
            # The following decides which clients should get trained or not.
            for clnt_idx in range(self.args.client_num_in_total):
                if active_ths_rnd[clnt_idx] == 0:
                    self.logger.info(
                        "@@@@@@@@@@@@@@@@ Client Drop this round CM({}) with density {}: {}".format(
                            round_idx, w_density[clnt_idx], clnt_idx
                        )
                    )

                self.logger.info(
                    "@@@@@@@@@@@@@@@@ Training Client CM({}) with density {}: {}".format(
                        round_idx, w_density[clnt_idx], clnt_idx
                    )
                )


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

                nei_indexs = np.sort(nei_indexs)

                # Update dist_locals

                if self.args.cs != "full":
                    self.logger.info(
                        "choose client_indexes: {}, accoring to {}".format(
                            str(nei_indexs), self.args.cs
                        )
                    )
                else:
                    self.logger.info(
                        "choose client_indexes: {}, accoring to {}".format(
                            str(nei_indexs), self.args.cs
                        )
                    )

                # Update each client's local model and the so-called consensus model
                # Allow the client average updated model.
                client = self.client_list[clnt_idx]
                if active_ths_rnd[clnt_idx] == 1:
                    if self.args.cs == "ring":
                        aggr_model = self._aggregate_func_ring(
                            c_index=clnt_idx,
                            nei_indexs=nei_indexs,
                            w_update=w_per_mdls_local,
                            w_original= w_per_mdls,reuse_index=reuse_index
                        )
                        w_local_mdl = self._apply_local_mask(
                            model_weights=aggr_model, mask=local_masks[clnt_idx]
                        )
                    else:
                        aggr_model = self._aggregate_func_bymask(
                            c_index=clnt_idx,
                            nei_indexs=nei_indexs,
                            w_update=w_per_mdls_local,w_original=w_per_mdls,round_idx=round_idx
                        )
                        # Because the topology is random selected, reuse index is also randomly selected
                        # For simplicity, we just use the client index to represent the reuse index
                        w_local_mdl = self._apply_local_mask(
                            model_weights=aggr_model, mask=local_masks[clnt_idx]
                        )


                else:
                    w_local_mdl = copy.deepcopy(w_per_mdls[clnt_idx])

                # client local training
                tst_results = client.local_test(
                    w_local=w_local_mdl, test_data_flag=True, test_before_train=True
                )
                tst_results_ths_round.append(tst_results)
                # Inside the client's training function, before starting the local training
                # sparsity =  compute_sparsity(w_local_mdl)
                # print(f"Sparsity of client {clnt_idx} model before local training starts: {sparsity}")
                mask_before_training = copy.deepcopy(local_masks[clnt_idx])
                (
                    w_local_mdl,
                    training_flops,
                    current_sparsity,
                    local_masks[clnt_idx],
                ) = client.train(
                    weight=copy.deepcopy(w_local_mdl),
                    rounds=round_idx,
                    mask=mask_before_training,
                    pq_prune=early_prune_flag,
                )

                # Record the difference between model weights to guide when to prune.
                if round_idx == 0:
                    # This will calculate the norm difference between the 1st round model and the initial model.
                    early_prune_stat_by_clients[clnt_idx].append(
                        self._weight_norm_difference(
                            w_local_mdl, w_init_models[clnt_idx]
                        )
                    )
                else:
                    delta_temp = self._weight_norm_difference(
                        w_local_mdl, w_init_models[clnt_idx]
                    )
                    early_prune_stat_by_clients[clnt_idx].append(delta_temp)
                    delta_current_all = early_prune_stat_by_clients[clnt_idx]
                    stat_threshold_temp = (
                        np.abs(delta_current_all[-1] - delta_current_all[-2])
                        / delta_current_all[0]
                    )
                    threshold_list.append(stat_threshold_temp)
                    self.logger.info(
                        "Client {}'s Model Change Ratio Delta: {}".format(
                            clnt_idx, stat_threshold_temp
                        )
                    )

                self.logger.info(
                    "Current Mask Sparsity: {}, for client {}".format(
                        str(current_sparsity), clnt_idx
                    )
                )
                # Inside the client's training function, before starting the local training

                test_local_metrics = client.local_test(
                    w_local=w_local_mdl, test_data_flag=True, test_before_train=False
                )
                final_tst_results_ths_round.append(test_local_metrics)
                # Update local model
                w_per_mdls_local[clnt_idx] = copy.deepcopy(w_local_mdl)

                self.stat_info["sum_training_flops"] += training_flops
                # self.stat_info["sum_comm_params"] += num_comm_params

            # This detect the first epoch to prune.
            # Compute the fraction of thresholds below the early prune threshold
            fraction_below_threshold = (
                len(
                    np.where(
                        np.array(threshold_list) <= self.args.early_prune_threshold
                    )[0]
                )
                / self.args.client_num_in_total
            )

            # Check if early pruning should be activated
            if fraction_below_threshold >= 0.5 and len(early_prune_record) == 0:
                self.logger.info(
                    "Early Prune Threshold Reached! Early Prune at round {}".format(
                        round_idx + 1
                    )
                )
                early_prune_record.append(round_idx)
            # Determine the pruning time based on the length of early_prune_record
            if len(early_prune_record) == 1:
                pruning_time = self.generate_pruning_freq(
                    a=early_prune_record[0], c=self.args.reconfig_reduce, step=20
                )
            else:
                pruning_time = []
            # Check if the current round index is in the pruning time
            early_prune_flag = round_idx in pruning_time
            if round_idx % 5 == 0:
                self._local_test_on_all_clients_before(tst_results_ths_round, round_idx)
                self._local_test_on_all_clients_aft(
                    final_tst_results_ths_round, round_idx
                )
            # Update the model after one communication round.
            w_per_mdls = copy.deepcopy(w_per_mdls_local)
        return

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


    def _aggregate_func_bymask(self, c_index, nei_indexs, w_update, w_original, round_idx):
        """
        Aggregate the models from neighbors
        :param c_index: current client index
        :param nei_indexs: neighbors' index
        :param w_per_mdls: model weights of each client
        :return:
        """
        max_wait = self.generate_stepwise_distributed_ns(largest_n=self.args.max_wait,smallest_n=0, total_rounds=self.args.comm_round)[round_idx]
        if max_wait >= len(nei_indexs) + 1:
            assert False, "The max_wait is larger than the number of neighbors"
        current_model = copy.deepcopy(w_original[c_index])
        # print('current client is :', c_index)
        nei_index_except_current = nei_indexs
        nei_index_except_current = nei_index_except_current[
            nei_index_except_current != c_index
        ]

        # This for selecting the top max_wait neighbors
        nei_index_except_current.sort()
        # print('neighbor index is:', nei_index_except_current)
        nei_models_update = [
            copy.deepcopy(w_update[nei_index])
            for nei_index in nei_index_except_current[:max_wait]
        ]
        nei_models_original = [
            copy.deepcopy(w_original[nei_index])
            for nei_index in nei_index_except_current[max_wait:]
        ]
        nei_models = nei_models_update + nei_models_original
        # Aggregate models
        aggregated_model = self._aggregate_models(
            c_model=current_model, nc_models=nei_models
        )

        return aggregated_model

    def _aggregate_func_ring(self, c_index, nei_indexs, w_update,w_original, reuse_index):
        """
        Aggregate the models from neighbors
        :param c_index: current client index
        :param nei_indexs: neighbors' index
        :param w_update: Track record of model weights of each client
        :param w_original: Original model weights of each client, unchanged during one communication round
        :return:
        """
        current_model = copy.deepcopy(w_original[c_index])
        # print('current client is :', c_index)
        nei_index_except_current = nei_indexs
        nei_index_except_current = nei_index_except_current[
            nei_index_except_current != c_index
        ]
        # print('neighbor index is:', nei_index_except_current)
        nei_models = []
        for nei_index in nei_index_except_current:
            cur_client_rindex = reuse_index[c_index]
            nei_client_rindex = reuse_index[nei_index]
            if cur_client_rindex>nei_client_rindex:
                nei_models.append(copy.deepcopy(w_update[nei_index]))
            else:
                nei_models.append(copy.deepcopy(w_original[nei_index]))

        # Aggregate models
        aggregated_model = self._aggregate_models(
            c_model=current_model, nc_models=nei_models
        )

        return aggregated_model

    def _aggregate_models(
        self, nc_models, c_model, omega_portion=0.5, dispfl_like=True
    ):
        """

        :param nc_models:  List of dictionaries to store personalized model weights (except current client). They are all masked by their own clients.
        :param c_models:   current client's model weights
        :param omega_portion: The portion of omega to be used in the aggregation for current model
        :return:
        """
        if not dispfl_like:
            masks = [self.model_trainer.get_mask(client) for client in nc_models]

            # Initialize aggregated mask with zeros
            aggregated_mask = {
                key: torch.zeros_like(tensor) for key, tensor in masks[0].items()
            }
            # Aggregate masks
            for mask in masks:
                for key in mask:
                    aggregated_mask[key] += mask[key]

            # Inverse aggregated mask
            inverse_aggregated_mask = {
                key: torch.zeros_like(tensor) for key, tensor in aggregated_mask.items()
            }
            for key in inverse_aggregated_mask:
                inverse_aggregated_mask[key] = torch.where(
                    aggregated_mask[key].float() != 0,
                    1 / aggregated_mask[key].float(),
                    aggregated_mask[key].float() * 0,
                )

            # aggregate the non-current client's model
            aggregated_nc_models = {
                key: torch.zeros_like(tensor) for key, tensor in nc_models[0].items()
            }
            for nc_model in nc_models:
                for key in nc_model:
                    # aggregate model with inverse aggregated mask
                    aggregated_nc_models[key] += (
                        nc_model[key] * inverse_aggregated_mask[key]
                    )

            # aggregate the current client's model
            aggregated_c_model = {
                key: torch.zeros_like(tensor)
                for key, tensor in aggregated_nc_models.items()
            }
            for key in c_model:
                aggregated_c_model[key] += (
                    omega_portion * c_model[key]
                    + (1 - omega_portion) * aggregated_nc_models[key]
                )

        else:
            all_models = [c_model] + nc_models
            masks = [self.model_trainer.get_mask(client) for client in all_models]

            # Initialize aggregated mask with zeros
            aggregated_mask = {
                key: torch.zeros_like(tensor) for key, tensor in masks[0].items()
            }
            # Aggregate masks
            for mask in masks:
                for key in mask:
                    aggregated_mask[key] += mask[key]

            # Inverse aggregated mask
            inverse_aggregated_mask = {
                key: torch.zeros_like(tensor) for key, tensor in aggregated_mask.items()
            }
            for key in inverse_aggregated_mask:
                inverse_aggregated_mask[key] = torch.where(
                    aggregated_mask[key].float() != 0,
                    1 / aggregated_mask[key].float(),
                    aggregated_mask[key].float() * 0,
                )

            # aggregate the non-current client's model
            aggregated_c_model = {
                key: torch.zeros_like(tensor) for key, tensor in c_model.items()
            }
            for model_element in all_models:
                for key in model_element.keys():
                    if key in inverse_aggregated_mask.keys():
                        # aggregate model with inverse aggregated mask
                        aggregated_c_model[key] += (
                            model_element[key] * inverse_aggregated_mask[key]
                        )
                    else:
                        aggregated_c_model[key] += model_element[key] / len(all_models)
                        # Since non-prune weights will produce full mask and the aggregation would be the same as FedAvg

        return aggregated_c_model

    def _apply_local_mask(self, model_weights, mask):
        """
        Apply mask to model weights
        :param model: model weights to be masked
        :param mask: mask to be applied
        :return:
        """
        for key in mask:
            mask[key] = mask[key].to(model_weights[key].device)
            model_weights[key] *= mask[key]
        return model_weights

    def generate_pruning_freq(self, a, b=0, c=3, step=8):
        """

        :param a: the first pruning epoch, and also stands for the gap between 0 and the first pruning epoch
        :param step: the target step to finish pruning
        :param b: Control the pruning speed to ensure a smooth pruning process, which is larger than 0
        :param c: Scale down the pruning frequency.
        :return:
        """
        pruning_freq = [a]
        for s in range(step):
            pruning_freq.append(round((pruning_freq[-1] + b) / c))

        pruning_epoch = np.cumsum(pruning_freq)

        # Plotting
        # plt.figure(figsize=(12, 6))
        # plt.scatter(range(len(pruning_epoch)), pruning_epoch, label='Rounds to Prune')
        # plt.xlabel('# Pruning')
        # plt.ylabel('Rounds')
        # plt.title('Pruning Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return pruning_epoch

    def generate_stepwise_distributed_ns(self,largest_n, smallest_n=0, total_rounds=500):
        # Calculate the number of rounds for each step
        step_count = largest_n - smallest_n + 1  # Include both endpoints
        rounds_per_step = total_rounds // step_count

        # Initialize the list to hold the values of N
        n_values = []

        # Generate the values for N
        for n in range(largest_n, smallest_n - 1, -1):  # From largest_n to smallest_n
            n_values.extend([n] * rounds_per_step)

        # Adjust the length in case the total rounds are not exactly divisible
        remainder = total_rounds % step_count
        if remainder != 0:
            # Extend with the smallest_n for the remaining rounds
            n_values.extend([smallest_n] * remainder)

        return n_values[:total_rounds]

    def _local_test_on_all_clients_aft(self, tst_results_ths_round, round_idx):
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
        self.stat_info["new_mask_test_acc"].append(test_acc)

    def _weight_norm_difference(self, model1_dict, model2_dict):
        total_diff = 0.0
        for key in model1_dict.keys():
            if key in model2_dict:  # Ensure the key exists in both models
                diff = model1_dict[key] - model2_dict[key]  # Compute the difference
                total_diff += torch.norm(
                    diff
                ).item()  # Compute the L2 norm and accumulate
        return total_diff

    def _local_test_on_all_clients_before(self, tst_results_ths_round, round_idx):
        self.logger.info(
            "################local_test_on_all_clients before local training in communication round: {}".format(
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
        self.stat_info["old_mask_test_acc"].append(test_acc)
