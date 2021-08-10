import torch.nn as nn
import torch

import sys

for it in sys.path:
    print(it)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("forward")
        return x

if __name__ == '__main__':
    # net = MyNet()
    # net(torch.randn(10))
    pass