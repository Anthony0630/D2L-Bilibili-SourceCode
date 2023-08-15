import torch
import torch.nn as nn
from torch.nn import functional as F

class MySequential(nn.Module):
    def __init__(self, *args):   # list of input arguments(use *)
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    