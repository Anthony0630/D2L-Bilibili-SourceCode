# 使用全连接层实现1 * 1卷积
# 注意：需要对输入输出的形状进行调整
import torch
from d2l import torch as d2l

def corr2d_multi_in_out_1x1(X, K):
    channel_in, height, width = X.shape
    channel_out = K.shape[0]
    X = X.reshape((channel_in, height * width))
    K = K.reshape((channel_out, channel_in))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((channel_out, height, width))