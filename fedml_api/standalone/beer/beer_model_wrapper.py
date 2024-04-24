import torch

# import torch.distributed as dist
from torch.nn.modules import Module

from copy import deepcopy
from fedml_api.standalone.beer.beer_utils import (
    flatten_tensors,
    assign_unflattened_tensors,
)

import fedml_api.utils.logger as logging_util


class ModelWrapper(Module):
    r"""The base distributed data parallel module.

    To reduce memory copy, flatten tensors into buckets, then assign unflattened
    new tensor to parameters.

    .. note::
        The actual communication happens at the beginning of each forward call.
        When training, the model should be validated before optimizer.step() to
        produce correct results.
    """

    def __init__(self, module):

        super().__init__()
        logger = logging_util.Logger()
        self.logger = logger.get_logger()

        self.logger.info("Using %s", self.__class__.__name__)

        self.param_info = [
            {"numel": param.numel(), "shape": param.shape}
            for param in module.parameters()
        ]

        self.device = next(module.parameters()).device
        self.module = module
        self.training = True

        self.flat_parameters = flatten_tensors(list(self.module.parameters())).to(
            self.device
        )
        assign_unflattened_tensors(
            self.module.parameters(), self.flat_parameters, self.param_info
        )

    @torch.no_grad()
    def eval(self):
        return self.module.eval()

    @torch.no_grad()
    def train(self):
        return self.module.train()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def zero_grad(self):
        self.module.zero_grad()

    @torch.no_grad()
    def zero_(self):
        for p in self.module.parameters():
            p.zero_()

    # New functions that do not exist in the original ModelWrapper
    @torch.no_grad()
    def assign_unflattened_tensors_to_parameters(self):
        assign_unflattened_tensors(
            self.module.parameters(), self.flat_parameters, self.param_info
        )
