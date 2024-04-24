# To consistently match the evaluation of DisPFL, we inherit some client features
# Modified code from https://github.com/rong-dai/DisPFL

import copy
import logging
import math

import numpy as np
import pdb
import torch

import fedml_api.standalone.dadpfl.dadpfl_model_trainer
from fedml_api.standalone.dadpfl.sp_functions import (
    SparsityIndex,
    avg_importance,
    conv_fc_condition,
    make_bound_si,
    compute_sparsity,
)


class client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer: fedml_api.standalone.adpfl.adpfl_model_trainer,
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

    def train(self, weight, rounds, mask, pq_prune=False):
        # com_paras_n = self.model_trainer.count_communication_params(weight)
        self.model_trainer.set_model_params(weight)
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_masks(mask)

        # To make equivalent comparison, we also record the performance before local training.
        self.model_trainer.train(
            self.local_training_data, self.device, self.args, round_idx=rounds
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

        # get mask according to rounds.
        if pq_prune:  # This is decided by when2prune statistics.
            mask_dict, _ = self.generate_mask_rigl_sparsity_awareness(
                model=self.model_trainer.model,
                target_sparsity=self.args.target_sparsity,
                scope=self.args.pq_scope,
                fixed_sparsity=self.args.fixed_sparsity,
            )

        else:
            mask_dict = self.model_trainer.get_mask(
                self.model_trainer.model.state_dict()
            )

        if (rounds + 1) % self.args.rr_interval == 0:
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
            # print('Remove and Regrow is done')
        # remove and regrow
        # if self.args.regrow:
        #    dataloader = self.local_training_data
        #    scores_by_layer = avg_importance(model=self.model_trainer.model, device=self.args.device, dataloader=dataloader, score_method='G2')
        #    mask_dict, num_remove = self.update_mask_throw(mask_dict, self.model_trainer.get_model_params(), rounds)
        #    mask_dict = self.update_mask_regrow(mask_dict, num_remove, scores_by_layer)
        #    print('Remove and Regrow is done')

        current_sparsity = compute_sparsity(self.model_trainer.model.state_dict())
        # print(f"Sparsity after throw and regrow is : {current_sparsity}")

        # uplink params
        # com_paras_n += self.model_trainer.count_communication_params(update)
        # del masks
        return weights_trained, training_flops, current_sparsity, mask_dict

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

    def generate_mask_rigl_sparsity_awareness(
        self, model, fixed_sparsity=False, scope="layer", target_sparsity=0.9
    ):
        """
        :param fixed_sparsity: if this is true, model will be pruned according to structured sparsity.
        Otherwise, the sparsity will be awared by PQ index.
        :return: Binary Mask in a dictionary form and the corresponding PQ-index
        """
        p, q = self.args.p, self.args.q
        eta_m = self.args.eta
        prune_scale = self.args.prune_scale
        device = self.args.device
        mask_kept = dict()

        _, current_sparsity = self.model_trainer.output_sparsity(
            self.model_trainer.get_mask(model=model.state_dict())
        )
        # Current setup is to prune the model if the current sparsity is less than target sparsity.
        if (not fixed_sparsity) and (current_sparsity < target_sparsity):
            self.logger.info(
                f"Current sparsity {current_sparsity}'s is lower than Target Sparsity {target_sparsity}  for client: {self.client_idx}"
            )
            # Initialize sparse_index_computer

            sparsity_index = SparsityIndex(p, q)
            mask_current = self.model_trainer.get_mask(model=model.state_dict())
            si_i = sparsity_index.make_si_(model=model, mask=mask_current, scope=scope)
            if scope == "layer":
                for name, param in model.named_parameters():
                    if conv_fc_condition(name) and param.dim() > 1:
                        mask_i = mask_current[name]
                        mask_i = mask_i.to(device)
                        d = mask_i.float().sum().to(device)
                        m = make_bound_si(si_i[name], d, p, q, eta_m)
                        param = param.to(mask_i.device)
                        pivot_param = param[mask_i.bool()].data.abs()
                        retain_ratio = m / d
                        prune_ratio = torch.clamp(
                            prune_scale * (1 - retain_ratio), 0, 0.9
                        )
                        # print('The prune ratio is', prune_ratio)
                        num_prune = torch.floor(d * prune_ratio).long()
                        pivot_value = torch.sort(pivot_param.view(-1))[0][num_prune]
                        pivot_mask = (param.data.abs() < pivot_value).to(device)
                        # The following transformation is to follow the structure of PQ index paper caculation
                        mask_kept[name] = (
                            torch.where(pivot_mask, False, mask_i.bool())
                            .float()
                            .to(device)
                        )
                        # param.data = torch.where(new_mask[name].to(param.device), param.data,
                        #                         torch.tensor(0, dtype=torch.float, device=param.device))
            else:  # scope == 'neuron'
                for name, param in model.named_parameters():
                    if conv_fc_condition(name) and param.dim() > 1:
                        mask_i = mask_current[name]
                        pivot_param = param.data.abs()
                        pivot_param[~mask_i.bool()] = float("nan")
                        d = (
                            mask_i.float()
                            .sum(dim=list(range(1, param.dim())))
                            .to(device)
                        )
                        m = make_bound_si(si_i[name], d, p, q, eta_m)
                        retain_ratio = m / d
                        prune_ratio = torch.clamp(
                            prune_scale * (1 - retain_ratio), 0, 0.9
                        )
                        num_prune = torch.floor(d * prune_ratio).long()
                        pivot_value = torch.sort(
                            pivot_param.view(pivot_param.size(0), -1), dim=1
                        )[0][torch.arange(pivot_param.size(0)), num_prune]
                        pivot_value = pivot_value.view(
                            -1, *[1 for _ in range(pivot_param.dim() - 1)]
                        )
                        pivot_mask = (param.data.abs() < pivot_value).to(device)
                        mask_kept[name] = torch.where(
                            pivot_mask, False, mask_i.bool()
                        ).float()

        else:  # Share approximately final sparsity with PQ-index method. This is just the RigL.
            self.logger.info(
                "Sparsity {}'s now is fixed for client: {}".format(
                    current_sparsity, self.client_idx
                )
            )
            mask_kept = self.model_trainer.get_mask(model=model.state_dict())
            si_i = None

        return mask_kept, si_i

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
