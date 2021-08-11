import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torchvision import datasets
from torchvision import transforms
from quantize import QuanNet, QuanConv2d, QuanLinear

class AlexNet(QuanNet):
    def __init__(self, n_bits: int, num_classes: int = 1000):
        super().__init__(n_bits)
        self.conv1 = QuanConv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.conv2 = QuanConv2d(64, 192, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = QuanConv2d(192, 384, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = QuanConv2d(384, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = QuanConv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))

        self.fc1 = QuanLinear(256 * 6 * 6, 4096)
        self.fc2 = QuanLinear(4096, 4096)
        self.fc3 = QuanLinear(4096, num_classes)

    def forward_(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv1.relu(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))

        x = self.conv2(x)
        x = self.conv2.relu(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))

        x = self.conv3(x)
        x = self.conv3.relu(x)

        x = self.conv4(x)
        x = self.conv4.relu(x)

        x = self.conv5(x)
        x = self.conv5.relu(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))

        x = F.adaptive_avg_pool2d(x, (6, 6))

        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = self.fc1.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc2.relu(x)

        x = self.fc3(x)

        return x

def load_data_set():
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
