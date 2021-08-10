import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torchvision import transforms
from quantize import QuanNet, QuanConv2d, QuanLinear

device = torch.device("cpu")

# nn.Module -> QuanNet
class LeNet5(QuanNet):
    def __init__(self, n_bits: int):
        super().__init__(n_bits)

        # Conv2d -> QuanConv2d
        self.conv1 = QuanConv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = QuanConv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.conv3 = QuanConv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1))
        # Linear -> QuanLinear
        self.fc1 = QuanLinear(in_features=120, out_features=84)
        self.fc2 = QuanLinear(in_features=84, out_features=10)

    # stack layers in forward_() instead of forward()
    def forward_(self, x):
        x = self.conv1(x)
        x = self.conv1.relu(x)  # F.relu() -> self.[layer].relu()
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = self.conv2.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = self.conv3.relu(x)

        x = x.view(x.size()[0], x.size()[1])
        x = self.fc1(x)
        x = self.fc1.relu(x)

        x = self.fc2(x)

        return x

def load_data_set():
    path = r"datasets/mnist_data_set"
    # 预处理
    process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data_set = datasets.MNIST(path, train=True, download=True, transform=process)
    test_data_set = datasets.MNIST(path, train=False, download=True, transform=process)

    # 拆分训练集和测试集
    train_data, test_data = [], []
    for i in range(len(train_data_set.data)):
        train_data.append(train_data_set[i][0])
    for i in range(len(test_data_set.data)):
        test_data.append(test_data_set[i][0])
    train_data, train_label, test_data, test_label = torch.stack(train_data), train_data_set.targets, torch.stack(test_data), test_data_set.targets

    train_data, train_label, test_data, test_label = train_data.to(device), train_label.to(device), test_data.to(device), test_label.to(device)
    return train_data, train_label, test_data, test_label

def train(net, train_data, train_label, n_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=0.1)
    net = net.to(device)
    criterion = criterion.to(device)

    loss_list = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        print("epoch: {}, training_loss: {}".format(
            epoch + 1,
            loss
        ))
        loss_list.append(float(loss))

    plt.plot(np.arange(n_epochs), loss_list)
    plt.savefig("lenet5_loss.png")

def acc(net, test_data, test_label):
    correct = 0
    with torch.no_grad():
        for i in range(test_data.size()[0]):
            data, target = test_data[i], test_label[i]
            data = data.view(1, data.size()[0], data.size()[1], data.size()[2])
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / test_data.size()[0]
    error = 1 - accuracy
    print('\nTest set: Accuracy: {}/{} ({:.0f}%), Error: {}/{} ({:.0f}%)\n'.format(
        correct,
        test_data.size()[0],
        100. * accuracy,
        test_data.size()[0] - correct,
        test_data.size()[0],
        100. * error
    ))

def test_train():
    train_data, train_label, test_data, test_label = load_data_set()

    net = LeNet5(8)
    net.float_mode()
    train(net, train_data, train_label, n_epochs=1000)
    torch.save(net.state_dict(), "lenet5.pt")

    print("After train:")
    acc(net, test_data, test_label)

def test_quan():
    train_data, train_label, test_data, test_label = load_data_set()

    net = LeNet5(8)
    net.load_state_dict(torch.load("lenet5.pt"), strict=False)
    net.float_mode()
    print("Before quantization:")
    acc(net, test_data, test_label)

    net.quantize(train_data)
    net.quan_mode()
    print("After quantization:")
    acc(net, test_data, test_label)

    torch.save(net.state_dict(), "lenet5_quan_8bit.pt")

def test_load_quan():
    train_data, train_label, test_data, test_label = load_data_set()

    net = LeNet5(8)
    net.quan_mode()
    net.load_state_dict(torch.load("lenet5_quan_8bit.pt"), strict=False)
    print(net.conv1.quan_weight)

    acc(net, test_data, test_label)

if __name__ == '__main__':
    # test_train()

    test_quan()

    test_load_quan()