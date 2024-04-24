import copy
import logging
import time

import numpy as np
import pdb
import torch
from torch import nn

from fedml_core.trainer.model_trainer import ModelTrainer
from fedml_api.standalone.beer.beer_utils import flatten_tensors


class BeerModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger
        self.device = next(model.parameters()).device

    # from beers
    def flat_parameters(self, model):
        return (
            torch.cat(
                [t.contiguous().view(-1) for t in list(model.parameters())], dim=0
            )
            .detach()
            .to(self.device)
        )

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        # TODO: could be further optimized to save memory with increased cost of computation if needed
        self.model.flat_parameters = flatten_tensors(
            list(self.model.module.parameters())
        ).to(self.device)

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data, device, args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr * (args.lr_decay**round),
                momentum=args.momentum,
                weight_decay=args.wd,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
            )
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(x)
                if type(output) == tuple:
                    # log.info('Tuple loss!')
                    loss = criterion(output[0], labels.long())
                    loss.backward()
                    loss_ref = criterion(output[1], labels.long())
                    loss_ref.backward()
                else:
                    loss = criterion(output, labels.long())
                    loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
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

                if type(pred) == tuple:
                    pred = pred[0]

                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
                metrics["test_acc"] = metrics["test_correct"] / metrics["test_total"]
        return metrics
