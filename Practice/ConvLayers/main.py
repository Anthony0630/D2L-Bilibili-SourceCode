import torch
import torch.nn as nn
from d2l import torch as d2l

# 定义二维互相关运算
def corr2d(X, Kernel):
    height, width = Kernel.shape
    Y = torch.zeros((X.shape[0] - height + 1, X.shape[1] - width + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + height, j:j + width] * Kernel).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

