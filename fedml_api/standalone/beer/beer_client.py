import copy
import logging
import math
import time
import pdb
import numpy as np
import torch
import fedml_api.standalone.beer.beer_compressors as compressor


class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer,
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

        self.gamma = args.gamma
        self.buf = torch.zeros_like(
            model_trainer.model.flat_parameters, device=self.device
        )
        self.H = torch.zeros_like(
            model_trainer.model.flat_parameters, device=self.device
        )
        self.V = torch.zeros_like(
            model_trainer.model.flat_parameters, device=self.device
        )
        self._G = torch.zeros_like(
            model_trainer.model.flat_parameters, device=self.device
        )
        self.X = None
        self.grads = None

        self.compression_operator = lambda x: getattr(
            compressor, args.compression_type
        )(x, *args.compression_params)

    def init(self):
        X = self.model_trainer.model.flat_parameters.to(self.device)
        X -= self.args.lr * self.V
        self.H += self.compression_operator(X - self.H)
        del X

    def get_sample_number(self):
        return self.local_sample_number

    def flatten_module_grads(self, module):
        """
        Flatten the gradients of all parameters of a given PyTorch module.

        Args:
        - module (torch.nn.Module): The module whose gradients will be flattened.

        Returns:
        - torch.Tensor: A single flattened tensor containing all gradients.
        """
        # Extract gradients and flatten them
        return torch.cat(
            [
                t.grad.contiguous().view(-1)
                for t in module.parameters()
                if t.grad is not None
            ],
            dim=0,
        ).detach()

    def train(self, round, w):
        # downlink params
        num_comm_params = self.model_trainer.count_communication_params(w)
        self.model_trainer.set_model_params(w)

        self.model_trainer.set_id(self.client_idx)

        tst_results_b4 = self.model_trainer.test(
            self.local_test_data, self.device, self.args
        )
        self.logger.info(
            "test acc on this client before {} / {} : {:.2f}".format(
                tst_results_b4["test_correct"],
                tst_results_b4["test_total"],
                tst_results_b4["test_acc"],
            )
        )

        self.model_trainer.train(
            self.local_training_data, self.device, self.args, round
        )
        w_trained = self.model_trainer.get_model_params()
        self.model_trainer.set_model_params(w_trained)

        tst_results_aft = self.model_trainer.test(
            self.local_test_data, self.device, self.args
        )
        self.logger.info(
            "test acc on this client after {} / {} : {:.2f}".format(
                tst_results_aft["test_correct"],
                tst_results_aft["test_total"],
                tst_results_aft["test_acc"],
            )
        )
        self.logger.info("-----------------------------------")

        # Calculation of training flops without considering sparsity
        full_flops = self.model_trainer.count_full_flops_per_sample()
        self.logger.info("full flops for training {}".format(full_flops))
        # Here we assume full density for simplicity
        training_flops = self.args.epochs * self.local_sample_number * full_flops

        return (
            w_trained,
            training_flops,
            num_comm_params,
            tst_results_b4,
            tst_results_aft,
        )

    def update_matrices_phase1(self):
        # already integrated in beer_api
        # self.grads = (self.flatten_module_grads(self.model_trainer.model.module).to(self.device) -
        #               self.flatten_module_grads(self.model_trainer.model.ref_module).to(self.device))
        # self.model_trainer.model.flat_ref_parameters[:] = self.model_trainer.model.flat_parameters[:]
        self.buf.zero_()

    def update_matrices_phase2(self, w):
        self.buf -= self._G
        self.V += self.gamma * self.buf + self.grads
        self._G += self.compression_operator(self.V - self._G)

        self.buf.zero_()

        self.model_trainer.set_model_params(w)
        del w
        self.X = self.model_trainer.model.flat_parameters.to(self.device)
        del self.grads

    def update_matrices_phase3(self):
        self.buf -= self.H
        self.X += self.gamma * self.buf - self.args.lr * self.V
        self.H += self.compression_operator(self.X - self.H)
        # Update flat_parameters
        self.model_trainer.model.flat_parameters[:] = self.X[:]
        del self.X

    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
