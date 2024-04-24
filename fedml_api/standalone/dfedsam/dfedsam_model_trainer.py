import copy
import torch
from torch import nn
from fedml_api.standalone.dfedsam.sam import SAM

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class DFedSAMMModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger

    def set_masks(self, masks):
        self.masks = masks
        # self.model.set_masks(masks)

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data, device, args, round):
        model = self.model
        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)

        base_optimizer = torch.optim.SGD  # Define the base optimizer
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=args.lr * (args.lr_decay**round),
            momentum=args.momentum,
            rho=args.rho,
        )

        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                # Closure function for SAM optimizer
                def closure():
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, labels)
                    loss.backward()
                    return loss

                # Perform a step with SAM optimizer
                loss = (
                    closure()
                )  # Computes the forward and backward pass with gradients
                optimizer.step(closure)
                optimizer.zero_grad()

                # Logging and accumulating the loss
                epoch_loss.append(loss.item())
                # Include any logger statements if needed

            # Logging the average loss for the epoch
            self.logger.info(
                f"Client Index = {self.id}\tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss)}"
            )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

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
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return False
