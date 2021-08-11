# CNN Pruning and Quantization
基于`Pytorch`实现对CNN的剪枝和任意bit量化
## Overview
剪枝(`Prune`)和量化(`Quantize`)可以在微小的精度损失下大幅减小CNN网络的体积，同时加快推理速度。FPGA因为其出色的并行能力，经常用于加速CNN
推理，但是由于板上资源有限，数据存储和传输经常会成为加速器的性能瓶颈。剪枝可以移除大量冗余连接，量化将浮点运算转化为定点运算，从而极大缩减了
存储需求，可以部署更大规模的网络。<br><br>
CNN压缩的流程通常为：训练模型->剪枝->微调(fine-tune)->量化->部署。`Pytorch`中提过了剪枝和量化相应的工具包，本仓库基于此设计了更通用的工具包
以简化压缩流程。
## Features
* 剪枝后微调(训练过程中只更新非零参数)<br>
* 支持任意bit量化<br>
* 非对称量化(相比对称量化提高表示精度)<br>
* 推理时只包含整数运算和移位，适合部署在边缘设备上<br>
* 导出为稀疏模型(`COO`、`CSR CSC`(未完成)等)<br>
* 简化的压缩流程<br>
## Quick Start
以`Lenet5`网络为例，参考[lenet5.py](https://github.com/XiaogegeChen/CNN_Compress/blob/master/example/lenet5.py) 。<br>
### (0) 设计模型
```python
from quantize import QuanNet, QuanConv2d, QuanLinear
import torch.nn.functional as F

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
```
按照注释处的提示，只需在原来模型定义的基础上改动少量代码即可。
### (1) 加载数据集，训练原模型
```python
def load_data_set():
    ...
    return train_data, train_label, test_data, test_label

def train(net, train_data, train_label, n_epochs=1000):
    ...
    for epoch in range(n_epochs):
        ...
        print("epoch: {}, training_loss: {}".format(epoch + 1, loss))
    ...

def acc(net, test_data, test_label):
    ...
    print('\nTest set: Accuracy: {}/{} ({:.0f}%), Error: {}/{} ({:.0f}%)\n'.format(...))


train_data, train_label, test_data, test_label = load_data_set()
# step1, train a dense model with float nn parameters
print("dense float model:")
net = LeNet5(8)
net = net.to(device)
net.float_mode()  # training should be performed in float mode 
train(net, train_data, train_label)
acc(net, test_data, test_label)
torch.save(net.state_dict(), "lenet5_dense_float.pt")
```
根据具体情况设计数据集加载和训练函数，训练出未经剪枝和量化的模型。
### (2)剪枝和微调
剪枝使用`Pytorch`提供的剪枝工具包 [prune tutorials](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) 。
工具包提供了多种剪枝策略，以全局剪枝为例：
```python
from prune import PruningTrainPlugin, sparsity_of_tensors
import torch.nn.utils.prune as pt_prune

def global_prune(net):
    parameters_to_prune = (
        (net.conv1, 'weight'),
        (net.conv2, 'weight'),
        (net.conv3, 'weight'),
        (net.fc1, 'weight'),
        (net.fc2, 'weight'),
    )

    # remove 90% weights
    pt_prune.global_unstructured(parameters_to_prune, pruning_method=pt_prune.L1Unstructured, amount=0.9)
    for p in parameters_to_prune:
        pt_prune.remove(p[0], p[1])  

    print("Sparsity in conv1.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.conv1.weight], 0)))
    print("Sparsity in conv2.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.conv2.weight], 0)))
    print("Sparsity in conv3.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.conv3.weight], 0)))
    print("Sparsity in fc1.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.fc1.weight], 0)))
    print("Sparsity in fc2.weight: {:.2f}%".format(100. * sparsity_of_tensors([net.fc2.weight], 0)))
    print("Global sparsity: {:.2f}%".format(100. * sparsity_of_tensors([net.conv1.weight, net.conv2.weight, net.conv3.weight, net.fc1.weight, net.fc2.weight], 0)))
```
这里采用全局剪枝，把所有权重中绝对值较小的90%剪去(设置为0)，只保留绝对值较大的10%的权重。通常经过剪枝后精度会迅速下降，需要冻结被剪去的权重
(更新权重时保持为0)后重新训练。因此需要重新定义一个训练函数，如下:
```python
from prune import PruningTrainPlugin, sparsity_of_tensors

def prune_train(net, train_data, train_label, n_epochs=1000):
    criterion = ...
    optimizer = ...
    criterion = criterion.to(device)

    # plugin this code snippet
    pft = PruningTrainPlugin()
    pft.set_net_named_parameters(net.named_parameters())

    loss_list = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        # plugin this code snippet
        pft.after_optimizer_step()

        print("train after prune, epoch: {}, training_loss: {}".format(...))
        loss_list.append(float(loss))

# step2, prune the dense model and retrain
print("sparse float model:")
net = LeNet5(8)
net = net.to(device)
net.load_state_dict(torch.load("lenet5_dense_float.pt"), strict=False)
net.float_mode()
global_prune(net)  # your own prune strategy
prune_train(net, train_data, train_label, n_epochs=n_epochs)  # retrain
acc(net, test_data, test_label)
torch.save(net.state_dict(), "lenet5_sparse_float.pt")
```
修改的方法非常简单，只需在原来train()函数基础上添加注释处的三行代码即可，之后开始训练剪枝后的稀疏模型。
### (3)量化
```python
# step3, quantize the model
print("sparse quantized model:")
net = LeNet5(8)
net = net.to(device)
net.load_state_dict(torch.load("lenet5_sparse_float.pt"), strict=False)
net.float_mode()  # quantize should be called in float mode
net.quantize(train_data)
net.quan_mode()  # switch to quantized mode to test the quantized model
acc(net, test_data, test_label)
torch.save(net.state_dict(), "lenet5_sparse_quantized.pt")
```
量化非常简单，只需要调用quantize()函数即可。<br>
Note:为了使推理过程完全是整数和移位运算，需要对权重和激活值都做量化。激活值的取值范围需要从一定数量的训练样本中统计出来，因此`quantize()`函数
需要传入一定数量的训练样本。
### (4) 加载模型、评估精度、导出模型
```python
# step4, used the sparse_quantized model
print("Evaluate sparse quantized model:")
net = LeNet5(8)
net = net.to(device)
net.load_state_dict(torch.load("lenet5_sparse_quantized.pt"), strict=False)
net.quan_mode()
acc(net, test_data, test_label)

# step5, save model in yaml format
storage.save_quan_model("lenet5_sparse_quantized.pt", "lenet5_sparse_quantized_float.yml", "lenet5_sparse_quantized_int.yml")
storage.save_sparse_model("lenet5_sparse_quantized.pt", "lenet5_sparse_quantized_float_coo.yml", "lenet5_sparse_quantized_int_coo.yml", form="coo")
```
加载和评估模型的方法和一般方法一样。为了可以与其它平台共享模型，这里将训练好的模型存储为yaml格式，将全部数据导出到.yml文件中，文件内容参考
[example/*.yml](https://github.com/XiaogegeChen/CNN_Compress/blob/master/example)

## Reference
* Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. [[PDF]](https://github.com/XiaogegeChen/CNN_Compress/blob/master/reference) [[Arxiv]](https://arxiv.org/abs/1712.05877v1 ) <br>
* [Prtorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) <br>