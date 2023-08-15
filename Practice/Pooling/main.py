import torch
import torch.nn as nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode="max"):
    pool_height, pool_width = pool_size
    Y = torch.zeros((X.shape[0]-pool_height+1, X.shape[1]-pool_width+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=="max":
                Y[i, j] = X[i:i+pool_height, j:j+pool_width].max()
            elif mode=="mean":
                Y[i, j] = X[i:i+pool_height, j:j+pool_width].mean()
    return Y

X = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
print(pool2d(X,(2,2)))
print(pool2d(X,(2,2),"mean"))