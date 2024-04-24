import copy
import logging
import math
import pickle
import random
import time

import pdb
import numpy as np
import torch

from fedml_api.standalone.beer.beer_client import Client
from fedml_api.standalone.beer.beer_comm_graph import CommunicationGraph
from fedml_api.standalone.beer.beer_utils import flatten_tensors


class beerAPI(object):
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
        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )

        # beer specific parameters
        self.compression_type = args.compression_type
        self.compression_params = args.compression_params
        self.graph_type = args.graph_type
        self.graph_params = args.graph_params
        self.gamma = args.gamma

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
        comm_graph = CommunicationGraph(
            self.args.client_num_in_total,
            graph_type=self.args.graph_type,
            graph_params=self.args.graph_params,
        )

        w_global = self.model_trainer.get_model_params()
        w_per_mdls = [
            copy.deepcopy(w_global) for _ in range(self.args.client_num_in_total)
        ]
        grads_per_mdls = list(np.zeros(self.args.client_num_in_total))

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

            tst_results_b4_round = []
            tst_results_aft_round = []
            # Actual training
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info(
                    "@@@@@@@@@@@@@@@@ Training Client CM({}): {}".format(
                        round_idx, clnt_idx
                    )
                )
                # Update each client's local model
                client = self.client_list[clnt_idx]

                (
                    w_trained,
                    training_flops,
                    num_comm_params,
                    tst_results_b4,
                    tst_results_aft,
                ) = client.train(round_idx, copy.deepcopy(w_per_mdls[clnt_idx]))

                w_per_mdls[clnt_idx] = copy.deepcopy(w_trained)
                del w_trained

                grads_tmp = client.flatten_module_grads(
                    client.model_trainer.model.module
                )
                if round_idx == 0:
                    client.V = grads_tmp.to(self.device)
                    client.grads = torch.zeros_like(
                        client.model_trainer.model.flat_parameters, device=self.device
                    )
                    # replace beer_client init() function
                    X = client.model_trainer.model.flat_parameters.to(self.device)
                    X -= self.args.lr * client.V
                    client.H += client.compression_operator(X - client.H)
                    del X
                else:
                    client.grads = grads_tmp - grads_per_mdls[clnt_idx]
                grads_per_mdls[clnt_idx] = copy.deepcopy(grads_tmp)
                del grads_tmp

                tst_results_b4_round.append(tst_results_b4)
                tst_results_aft_round.append(tst_results_aft)

                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params

            self._local_test_on_all_clients_b4(tst_results_b4_round, round_idx)
            self._local_test_on_all_clients_aft(tst_results_aft_round, round_idx)

            # init matrices for each client (now integrated with above!!!)
            # if round_idx == 0:
            #     for clnt_idx in range(self.args.client_num_in_total):
            #         client = self.client_list[clnt_idx]
            #         client.V = client.flatten_module_grads(client.model_trainer.model.module).to(self.device)
            #         client.model_trainer.model.zero_grad()
            #         client.init()

            # Beer phase 1
            for clnt_idx in range(self.args.client_num_in_total):
                client = self.client_list[clnt_idx]
                client.update_matrices_phase1()

            # Mixing of grads
            self._centralized_mix("_G", comm_graph)

            # Beer phase 2
            for clnt_idx in range(self.args.client_num_in_total):
                client = self.client_list[clnt_idx]
                client.update_matrices_phase2(copy.deepcopy(w_per_mdls[clnt_idx]))

            # Mixing of params
            self._centralized_mix("H", comm_graph)

            # Beer phase 3
            for clnt_idx in range(self.args.client_num_in_total):
                client = self.client_list[clnt_idx]
                client.update_matrices_phase3()
                # Unflatten parameters
                client.model_trainer.model.assign_unflattened_tensors_to_parameters()
                client.model_trainer.model.zero_grad()
                # finish aggregation
                w_per_mdls[clnt_idx] = client.model_trainer.get_model_params()

        return

    def _centralized_mix(self, matrix_name, communication_graph):
        if not self.client_list:
            raise ValueError("Client list is empty")

        # Check if matrix_name is valid
        valid_matrices = ["_G", "H", "V"]
        if matrix_name not in valid_matrices:
            raise ValueError(
                f"Invalid matrix name '{matrix_name}'. Valid options are {valid_matrices}"
            )

        # Mix the specified matrix for each client based on its neighbors
        for i, client in enumerate(self.client_list):
            # Initialize a buffer for aggregation
            total_matrix = torch.zeros_like(getattr(client, matrix_name))

            # Get neighbors of the client
            neighbors = communication_graph.neighbors(i)
            neighbor_count = 0

            # Aggregate matrix from the client and its neighbors
            for neighbor_idx in neighbors:
                neighbor_matrix = getattr(self.client_list[neighbor_idx], matrix_name)
                total_matrix += neighbor_matrix
                neighbor_count += 1

            # Include the client's own matrix in the average
            total_matrix += getattr(client, matrix_name)
            neighbor_count += 1

            # Calculate the average and update the client's buffer
            average_matrix = total_matrix / neighbor_count
            client.buf[:] = average_matrix

    def _local_test_on_all_clients_b4(self, tst_results_ths_round, round_idx):
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
        self.stat_info["old_mask_test_acc"].append(test_acc)

    def _local_test_on_all_clients_aft(self, tst_results_ths_round, round_idx):
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
        self.stat_info["new_mask_test_acc"].append(test_acc)

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

    def init_stat_info(
        self,
    ):
        self.stat_info = {}
        self.stat_info["label_num"] = self.class_counts
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []
