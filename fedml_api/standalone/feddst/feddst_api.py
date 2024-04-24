import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch

from fedml_api.standalone.adpfl.sp_functions import conv_fc_condition
from fedml_api.standalone.feddst.client import client


class FedDSTAPI(object):
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
            c = client(
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
        params = self.model_trainer.get_trainable_params()
        # w_spa = [self.args.dense_ratio for i in range(self.args.client_num_in_total)]
        if self.args.uniform:
            sparsities = self.model_trainer.calculate_sparsities(
                params, distribution="uniform", dense_ratio=self.args.dense_ratio
            )
            temp = self.model_trainer.init_masks(params, sparsities)
            mask_per_mdls = [
                copy.deepcopy(temp) for i in range(self.args.client_num_in_total)
            ]

        else:
            sparsities = self.model_trainer.calculate_sparsities(
                params, dense_ratio=self.args.dense_ratio
            )
            temp = self.model_trainer.init_masks(params, sparsities)
            mask_per_mdls = [
                copy.deepcopy(temp) for i in range(self.args.client_num_in_total)
            ]

        w_global = self.model_trainer.get_model_params()
        w_per_mdls = []

        # Initialization
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global))
        # device = {device} cuda:0apply mask to init weights
        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm

            comm_round_iterable = tqdm(
                range(self.args.comm_round), desc="Comm. Rounds", ncols=100
            )
        global_mask = mask_per_mdls[0]

        for round_idx in comm_round_iterable:
            self.logger.info(
                "################Communication round : {}".format(round_idx)
            )

            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            client_indexes = np.sort(client_indexes)

            self.logger.info("client_indexes = " + str(client_indexes))
            aggr_w_per_mdls = []
            aggr_mask_per_mdls = []
            for cur_clnt in client_indexes:
                self.logger.info(
                    "@@@@@@@@@@@@@@@@ Training Client CM({}): {}".format(
                        round_idx, cur_clnt
                    )
                )
                # update dataset
                client = self.client_list[cur_clnt]
                # update meta components in personal network
                w_per, training_flops, mask_per = client.train(
                    copy.deepcopy(w_global), round_idx, mask=copy.deepcopy(global_mask)
                )
                aggr_w_per_mdls.append(copy.deepcopy(w_per))
                aggr_mask_per_mdls.append(copy.deepcopy(mask_per))
                # self.logger.info("local weights = " + str(w))
                self.stat_info["sum_training_flops"] += training_flops

            # update global meta weights
            w_global_unpruned = self._aggregate_func(
                model_list=aggr_w_per_mdls,
                mask_list=aggr_mask_per_mdls,
                global_mask=global_mask,
            )
            w_global = self._prune_layerwise(
                weights_dict=w_global_unpruned, sp_dist=sparsities
            )
            global_mask = copy.deepcopy(self.model_trainer.get_mask(w_global))
            _, check_sparsity = self.model_trainer.output_sparsity(global_mask)
            self.logger.info("check_sparsity = " + str(check_sparsity))

            self._test_on_all_clients(w_global, round_idx)
            # self._local_test_on_all_clients(w_global, round_idx)

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

    def _aggregate_func(self, model_list, mask_list, global_mask):
        self.logger.info("Doing Global aggregation!")

        device = global_mask[next(iter(global_mask.keys()))].device
        count_mask = {
            k: torch.zeros_like(v, device=device) for k, v in global_mask.items()
        }

        # Mask Aggregation
        for k in count_mask.keys():
            for mask_per in mask_list:
                count_mask[k] += mask_per[k].to(device)

        for k in count_mask.keys():
            # Avoid division by zero by maintaining zero values as they are
            mask_non_zero = count_mask[k] != 0
            count_mask[k][mask_non_zero] = torch.reciprocal(
                count_mask[k][mask_non_zero]
            )

        total_zero_elements = 0
        total_elements = 0
        for mask in count_mask.values():
            total_elements += mask.numel()
            total_zero_elements += torch.sum(mask == 0).item()
        total_sparsity = total_zero_elements / total_elements
        self.logger.info(
            f"Total Sparsity of count_mask after reprocal: {total_sparsity:.4f}"
        )

        # Model Aggregation
        w_tmp = {
            k: torch.zeros_like(v, device=device) for k, v in model_list[0].items()
        }
        num_clients = torch.tensor(len(model_list)).to(device)

        for k in w_tmp.keys():
            for model in model_list:
                w_tmp[k] += model[k].to(device)

        total_zero_elements = 0
        total_elements = 0
        for mask in w_tmp.values():
            total_elements += mask.numel()
            total_zero_elements += torch.sum(mask == 0).item()

        total_sparsity = total_zero_elements / total_elements
        self.logger.info(
            f"Total Sparsity of global model before mask applied: {total_sparsity:.4f}"
        )

        # Model Averaging
        for k in w_tmp.keys():
            if k in count_mask.keys():
                w_tmp[k] = w_tmp[k] * count_mask[k]
            else:
                w_tmp[k] = w_tmp[k] / num_clients

        total_zero_elements = 0
        total_elements = 0
        for mask in w_tmp.values():
            total_elements += mask.numel()
            total_zero_elements += torch.sum(mask == 0).item()

        total_sparsity = total_zero_elements / total_elements
        self.logger.info(
            f"Total Sparsity of global model after mask applied: {total_sparsity:.4f}"
        )

        return w_tmp

    def _prune_layerwise(self, weights_dict, sp_dist):
        pruned_weights = {}
        for name, weight in weights_dict.items():
            if conv_fc_condition(name=name):
                desired_sparsity = sp_dist[name]
                current_sparsity = torch.mean((weight == 0).float()).item()

                if current_sparsity < desired_sparsity:
                    # Additional pruning needed
                    num_to_keep = int(weight.numel() * (1 - desired_sparsity))
                    threshold = torch.topk(
                        weight.abs().flatten(), num_to_keep
                    ).values.min()
                    mask = weight.abs() >= threshold
                else:
                    # No additional pruning needed
                    mask = weight != 0

                pruned_weights[name] = weight * mask
            else:
                pruned_weights[name] = weight
        return pruned_weights

    def _test_on_all_clients(self, w_global, round_idx):

        self.logger.info(
            "################global_test_on_all_clients : {}".format(round_idx)
        )

        g_test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

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

        stats = {"global_test_acc": g_test_acc, "global_test_loss": g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
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
