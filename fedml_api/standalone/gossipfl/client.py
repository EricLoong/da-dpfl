import numpy as np


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

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_aggr, global_mask, global_mask_next, round, w_local_last_epoch):
        recover_model = self._recover(global_mask, w_local_last_epoch, w_aggr)
        num_comm_params = self.model_trainer.count_communication_params(recover_model)
        self.model_trainer.set_model_params(recover_model)
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(
            self.local_training_data, self.device, self.args, round
        )
        weights = self.model_trainer.get_model_params()
        tst_results = self.model_trainer.test(
            self.local_test_data, self.device, self.args
        )
        self.logger.info(
            "training_flops{}".format(
                self.model_trainer.count_training_flops_per_sample()
            )
        )
        self.logger.info(
            "full{}".format(self.model_trainer.count_full_flops_per_sample())
        )
        training_flops = (
            self.args.epochs
            * self.local_sample_number
            * self.model_trainer.count_training_flops_per_sample()
        )
        self.logger.info("flops_one_round{}".format(training_flops))
        num_comm_params += self.model_trainer.count_communication_params(weights)
        compressed_weights = {}
        for key, param in weights.items():
            if key in global_mask_next.keys():
                compressed_weights[key] = param * global_mask_next[key]
            else:
                compressed_weights[key] = param
        return weights, compressed_weights, training_flops, num_comm_params, tst_results

    def _recover(self, mask_dict, original_model, recieve_model):
        recover_model = {}
        for key, param in original_model.items():
            if key in mask_dict.keys():
                # The mask is a torch tensor binary values (0s and 1s)
                recover_model[key] = (
                    param * (1 - mask_dict[key]) + recieve_model[key] * mask_dict[key]
                )
            else:
                recover_model[key] = param
        return recover_model

    def local_test(self, w, b_use_test_dataset=True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
