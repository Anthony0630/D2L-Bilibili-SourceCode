import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
def synthetic_data(W, b, num_examples):
    """生成 y = Wx + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(W)))
    y = torch.matmul(X, W) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_W = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_W, true_b, 1000)
print(f"Sample_0 ----- features:{features[0]}, label:{labels[0]}")

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取样本，没有固定顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 定义初始化模型参数
W = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, W, b):
    """线性回归模型"""
    return torch.matmul(X, W) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方误差"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def SGD(params, lr, batch_size):
    """小批量梯度下降算法"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # pytorch不会自动梯度归零

# 训练过程
lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, W, b), y)
        l.sum().backward()
        SGD([W, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, W, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

print(f"W的估计误差：{true_W - W.reshape(true_W.shape)}")
print(f"b的估计误差：{true_b - b}")
