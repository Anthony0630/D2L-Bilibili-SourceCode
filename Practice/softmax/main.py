import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# Python不会隐式调整输入的形状，因此需要通过flatten来调整网络输入的形状
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)   # 将所有全连接层的权重归一初始化

net.apply(init_weights)

# 在交叉熵损失函数中传递未归一化的预测，并同时计算softmax及其对数
loss = nn.CrossEntropyLoss(reduction='none')

# 使用学习率为0.1的小批量随机梯度下降作为优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# 调用之前的训练函数来训练模型
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
d2l.plt.show()

