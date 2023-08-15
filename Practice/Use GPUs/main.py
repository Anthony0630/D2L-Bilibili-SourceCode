import torch
import torch.nn as nn

def try_GPU(i=0):
    """如果有GPU，返回GPU(i)，否则返回CPU()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else: return torch.device('cpu')

def try_all_GPUs():
    """返回所有可用的GPU，如果没有，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

net = nn.Sequential(nn.Linear(3, 1))
net.to(device=try_GPU())
print(net[0].weight.data.device)