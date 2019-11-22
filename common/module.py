from collections import namedtuple
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(self, config):
        super(Module, self).__init__()
        self.config = config

    def init_weight(self):
        raise NotImplementedError()

    def fix_params(self):
        raise NotImplementedError()

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.preprocess(*inputs, **kwargs)
        if self.training:
            return self.train_forward(*inputs, **kwargs)
        else:
            return self.inference_forward(*inputs, **kwargs)

    def train_forward(self, *inputs, **kwargs):
        """
        def train_forward(self, data, label, **kwargs):
            # this is a toy example for 1 output, 2 loss function

            output = None
            loss1 = torch.tensor(0.0)
            loss2 = torch.tensor(0.0)

            outputs = {'output': output,
                       'loss1': loss1,
                       'loss2': loss2}
            loss = loss1 + loss2

            return outputs, loss
        """
        raise NotImplemented

    def inference_forward(self, *inputs, **kwargs):
        """
        def inference_forward(self, data, **kwargs):
            output = None
            outputs = {'output': output}
            return outputs
        """
        raise NotImplemented

    def preprocess(self, *inputs, **kwargs):
        if self.training:
            return self.train_preprocess(*inputs, **kwargs)
        else:
            return self.inference_preprocess(*inputs, **kwargs)

    def train_preprocess(self, *inputs, **kwargs):
        return inputs, kwargs

    def inference_preprocess(self, *inputs, **kwargs):
        return inputs, kwargs
