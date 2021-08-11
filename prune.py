import torch
from typing import List

class PruningTrainPlugin:
    """
    剪枝之后的训练，需要控制在训练过程中被剪枝的权重值(权重值为0)不更新。
    Example:
        def prune_train(net, n_epochs: int):
            optimizer = optim.Adadelta(net.parameters(), lr=0.1)
            criterion = nn.NLLLoss()

            # plugin this code snippet
            pft = PruningTrainPlugin()
            pft.set_net_named_parameters(net.named_parameters())

            for epoch in range(n_epochs):
                optimizer.zero_grad()
                output = net(train_data)
                loss = criterion(output, train_label)
                loss.backward()
                optimizer.step()

                # plugin this code snippet
                pft.after_optimizer_step()

                print("epoch: {}, training_loss: {}".format(
                    epoch + 1,
                    loss
                ))
    """
    def __init__(self):
        self.net_named_parameters = None
        self.net_named_masks = None

    def set_net_named_parameters(self, net_named_parameters):
        """
        把模型的可训练参数传递过来，生成对应的mask
        """
        self.net_named_parameters = list(net_named_parameters)
        self.net_named_masks = {}
        for named_param in self.net_named_parameters:
            name, param = named_param[0], named_param[1]
            mask = torch.where(param == 0, torch.zeros(param.size()).to(param.device), torch.ones(param.size()).to(param.device)).to(param.device)
            self.net_named_masks.update({name: mask})

    def after_optimizer_step(self):
        """
        在优化器更新值之后，手动把mask为0的值改成0。因为在下一次forward时会使用0值计算梯度，因此这样可以起到优化作用
        """
        if self.net_named_parameters is None:
            print("Please call set_net_named_parameters first to pass net_named_parameters")
            return
        for named_param in self.net_named_parameters:
            name, param = named_param[0], named_param[1]
            mask = self.net_named_masks[name]
            param.data = torch.mul(mask, param).data

# 计算一组张量的稀疏度，即只统计张量中无效元素的比例
def sparsity_of_tensors(ts: List[torch.Tensor], invalid_value: float = 0) -> float:
    n_elem = 0
    n_inv = 0
    for t in ts:
        n_elem += int(t.nelement())
        n_inv += int(torch.sum(t == invalid_value))
    return n_inv / n_elem


