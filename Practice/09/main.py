import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, 10, requires_grad=True)

# 定义Softmax运算
def Softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition     # Broadcast

# 定义模型
def net(X):
    return Softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

# 分类精度
def accuracy(y_hat, y):
    """计算正确预测的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
