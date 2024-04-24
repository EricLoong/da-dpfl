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

    def train(self, current_w_agg, round):
        # self.logger.info(sum([torch.sum(w_per[name]) for name in w_per]))
        num_comm_params = self.model_trainer.count_communication_params(current_w_agg)
        self.model_trainer.set_model_params(current_w_agg)
        # tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        # self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(
            self.local_training_data, self.device, self.args, round
        )
        weights = self.model_trainer.get_model_params()
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
        num_comm_params += self.model_trainer.count_communication_params(weights)

        return weights, training_flops, num_comm_params

    def local_test(self, w, b_use_test_dataset=True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
