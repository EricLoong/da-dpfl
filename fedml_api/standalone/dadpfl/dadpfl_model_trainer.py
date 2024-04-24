# Following the source code of DisPFL, we modified and developed them to build new adpfl to compare
# Pruning code follow RigL mixed with sparsity awareness
import copy
import logging
import torch
from torch import nn
import numpy as np
from fedml_api.standalone.dadpfl.sp_functions import (
    SparsityIndex,
    avg_importance,
    conv_fc_condition,
    make_bound_si,
)
import math

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class dadpflMT(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.masks = None
        self.args = args
        self.logger = logger
        self.model = model

    def set_masks(self, masks):
        self.masks = masks

    def get_mask(self, model):
        # Model weights are not the same as model stored in client
        mask_dict = {}
        # for name, param in model.named_parameters():
        for name, param in model.items():
            if conv_fc_condition(
                name=name
            ):  # Assuming you want the mask for weights only
                mask = (param != 0).float()  # Binary mask indicating non-zero weights
                mask_dict[name] = mask
        return mask_dict

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def init_masks(self, params, sparsities):
        masks = {}
        for name in params:
            if conv_fc_condition(name=name):
                mask_name = name
                masks[mask_name] = torch.zeros_like(params[name])
                dense_numel = int(
                    (1 - sparsities[name]) * torch.numel(masks[mask_name])
                )
                if dense_numel > 0:
                    temp = masks[mask_name].view(-1)
                    perm = torch.randperm(len(temp))
                    perm = perm[:dense_numel]
                    temp[perm] = 1
        return masks

    def output_sparsity(self, mask_dict):
        sparsity_dict = {}
        zero_elements_all = 0
        total_elements_all = 0
        for layer_name, mask_tensor in mask_dict.items():
            total_elements = torch.numel(mask_tensor)
            zero_elements = torch.sum(mask_tensor == 0).item()

            sparsity = zero_elements / total_elements
            sparsity_dict[layer_name] = sparsity

            total_elements_all += total_elements
            zero_elements_all += zero_elements
        total_sparsity = zero_elements_all / total_elements_all
        # print(f"Total sparsity: {total_sparsity * 100:.2f}%")

        return sparsity_dict, total_sparsity

    def calculate_sparsities(
        self, params, tabu=[], distribution="ERK", dense_ratio=0.5
    ):
        sparsities = {}
        if distribution == "uniform":
            for name in params:
                if conv_fc_condition(name=name):
                    if name not in tabu:
                        sparsities[name] = 1 - self.args.dense_ratio
                    else:
                        sparsities[name] = 0

        elif distribution == "ERK":
            # Mute it for test
            # self.logger.info('initialize by ERK')
            total_params = 0
            for name in params:
                if conv_fc_condition(name=name):
                    total_params += params[name].numel()
            is_epsilon_valid = False
            dense_layers = set()
            density = dense_ratio
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if conv_fc_condition(name=name):
                        if name in tabu:
                            dense_layers.add(name)
                        n_param = np.prod(params[name].shape)
                        n_zeros = n_param * (1 - density)
                        n_ones = n_param * density

                        if name in dense_layers:
                            rhs -= n_zeros
                        else:
                            rhs += n_ones
                            raw_probabilities[name] = (
                                np.sum(params[name].shape) / np.prod(params[name].shape)
                            ) ** self.args.erk_power_scale
                            divisor += raw_probabilities[name] * n_param

                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if conv_fc_condition(name=name):
                    if name in dense_layers:
                        sparsities[name] = 0
                    else:
                        sparsities[name] = 1 - epsilon * raw_probabilities[name]
        return sparsities

    def train(self, train_data, device, args, round_idx):
        model = self.model
        model.to(device)
        model.train()
        mask = self.masks

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr * (args.lr_decay**round_idx),
                momentum=args.momentum,
                weight_decay=args.wd,
            )
        else:
            optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=args.lr, weight_decay=args.wd
            )

        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x)
                loss = criterion(log_probs, labels.long())
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()

                # Add the weight thresholding after optimizer step

                # Apply the mask
                for name, param in model.named_parameters():
                    if name in mask:
                        param.data *= mask[name].to(device)

                epoch_loss.append(loss.item())

            self.logger.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_acc": 0.0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
                metrics["test_acc"] = metrics["test_correct"] / metrics["test_total"]
        return metrics
