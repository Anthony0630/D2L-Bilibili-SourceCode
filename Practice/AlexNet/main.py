import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

# X = torch.rand(1, 1, 224, 224)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,"output_size:\t", X.shape)

# Fashion_MNIST的图像分辨率低于ImageNet，先增加分辨率到224*224
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

learning_rate, epochs = 0.01, 100
d2l.train_ch6(net, train_iter, test_iter, epochs, learning_rate, d2l.try_gpu())