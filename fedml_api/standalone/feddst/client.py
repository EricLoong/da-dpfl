# To consistently match the evaluation of DisPFL, we inherit some client features
# Modified code from https://github.com/rong-dai/DisPFL

import copy
import logging
import math

import numpy as np
import pdb
import torch
import fedml_api.standalone.feddst.feddst_model_trainer
from fedml_api.standalone.adpfl.sp_functions import avg_importance, conv_fc_condition


class client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer: fedml_api.standalone.feddst.feddst_model_trainer,
        logger,
    ):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    # Attention! local_raw_training_data is not updated here!
    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, weight, rounds, mask):
        # com_paras_n = self.model_trainer.count_communication_params(weight)
        self.model_trainer.set_model_params(weight)
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_masks(mask)

        # Update the mask every rr_interval rounds
        if (rounds + 1) % self.args.rr_interval == 0:
            mask_dict = copy.deepcopy(mask)
            dataloader = self.local_training_data
            scores_by_layer = avg_importance(
                model=self.model_trainer.model,
                device=self.args.device,
                dataloader=dataloader,
                score_method="G2",
            )
            mask_dict, num_remove = self.update_mask_throw(
                mask_dict, self.model_trainer.get_model_params(), rounds
            )
            mask_dict = self.update_mask_regrow(mask_dict, num_remove, scores_by_layer)
            _, regrow_sp = self.model_trainer.output_sparsity(mask_dict)
            self.model_trainer.set_masks(mask_dict)
            self.logger.info("After regrowing, the sparsity is: {}".format(regrow_sp))

        # To make equivalent comparison, we also record the performance before local training.
        self.model_trainer.train(
            self.local_training_data, self.device, self.args, rounds
        )
        weights_trained = self.model_trainer.get_model_params()
        self.model_trainer.set_model_params(weights_trained)
        sparse_flops_per_data = self.model_trainer.count_training_flops_per_sample()
        full_flops = self.model_trainer.count_full_flops_per_sample()
        self.logger.info("training flops per data {}".format(sparse_flops_per_data))
        self.logger.info("full flops for search {}".format(full_flops))
        training_flops = (
            self.args.epochs * self.local_sample_number * sparse_flops_per_data
            + self.args.batch_size * full_flops
        )
        mask_dict = self.model_trainer.get_mask(self.model_trainer.model.state_dict())
        _, sp = self.model_trainer.output_sparsity(mask_dict)
        self.logger.info("No RR, the sparsity is: {}".format(sp))

        return weights_trained, training_flops, mask_dict

    def local_test(self, w_local=None, test_data_flag=True, test_before_train=True):
        if type(None) != type(w_local):
            self.model_trainer.set_model_params(w_local)
        if test_data_flag:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        prompt = "before"
        if not test_before_train:
            prompt = "after"

        tst_results = self.model_trainer.test(test_data, self.device, self.args)
        self.logger.info(
            f"test acc on this client {prompt} {tst_results['test_correct']} / {tst_results['test_total']} : {tst_results['test_acc']:.2f}"
        )
        return tst_results

    def update_mask_throw(self, masks, weights, round):
        device = self.args.device

        drop_ratio = (
            self.args.anneal_factor
            / 2
            * (1 + np.cos((round * np.pi) / self.args.comm_round))
        )
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks.keys():
            if conv_fc_condition(name):
                num_non_zeros = torch.sum(masks[name].to(device))
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
                temp_weights = torch.where(
                    masks[name].to(device) > 0,
                    torch.abs(weights[name].to(device)),
                    100000 * torch.ones_like(weights[name].to(device)),
                )
                x, idx = torch.sort(temp_weights.view(-1))
                new_masks[name].view(-1)[
                    idx[: num_remove[name]]
                ] = 0  # Prune the smallest weights

        for key in new_masks:
            new_masks[key] = new_masks[key].to(device)

        return new_masks, num_remove

    def update_mask_regrow(self, masks, num_remove, score_by_layers=None):
        new_masks = copy.deepcopy(masks)
        device = self.args.device
        for name in masks.keys():
            # print("Processing layer:", name)
            # print("Shape of score_by_layers[{}]:".format(name),
            #      score_by_layers[name].shape)
            if conv_fc_condition(name):
                # print('Prunable layer',name)
                if self.args.rigl:
                    # This is for rigl, regrow the weights with the largest gradient**2
                    # Inside the RigL block:
                    negative_tensor = -100000 * torch.ones_like(score_by_layers[name])
                    # print("Shape of masks[name]:", masks[name].shape)
                    # print("Shape of score_by_layers[count]:", score_by_layers[name].shape)
                    # print("Shape of negative_tensor:", negative_tensor.shape)

                    temp = torch.where(
                        masks[name].to(device) == 0,
                        torch.abs(score_by_layers[name].to(device)),
                        negative_tensor.to(device),
                    )
                    sort_temp, idx = torch.sort(
                        temp.view(-1).to(self.args.device), descending=True
                    )
                    new_masks[name].view(-1)[idx[: num_remove[name]]] = 1

                else:
                    temp = torch.where(
                        masks[name].to(device) == 0,
                        torch.ones_like(masks[name].to(device)),
                        torch.zeros_like(masks[name].to(device)),
                    )
                    # Randomly regrow the weights (Methods like SET)
                    idx = torch.multinomial(
                        temp.flatten(), num_remove[name], replacement=False
                    )
                    new_masks[name].view(-1)[idx] = 1

        for key in new_masks:
            new_masks[key] = new_masks[key].to(self.args.device)

        return new_masks
