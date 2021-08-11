import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch import Tensor

__float_mode__ = "float_mode"
__quan_mode__ = "quan_mode"

class QuanNet(nn.Module):
    """
    支持任意bit量化的神经网络
    """
    def __init__(self, n_bits: int):
        super().__init__()
        self._mode: str = __float_mode__  # 计算模式

        self.n_bits: int = n_bits

        # 输入的scale(和第一层的ins一样)、输入的zero_point(和第一层的inz一样)、输出的scale(和最后一层的outs一样)、输出的zero_point(和最后一层的outz一样)
        quan_buffers = ["_ins", "_inz", "_outs", "_outz"]
        for it in quan_buffers:
            self.register_buffer(it, torch.tensor(0.))

    def float_mode(self):
        """
        切换到float模式
        """
        self._mode = __float_mode__
        self._switch_mode(__float_mode__)

    def quan_mode(self):
        """
        切换到quan模式
        """
        self._mode = __quan_mode__
        self._switch_mode(__quan_mode__)

    def forward(self, x: Tensor) -> Tensor:
        if self._mode == __quan_mode__:
            # 需要先量化x
            x = utils.float2fixed_point_SZ_tensor(x, float(self._ins), int(self._inz))

        x = self.forward_(x)

        if self._mode == __quan_mode__:
            # 输出反量化
            x = utils.fixed_point2float_tensor(x, float(self._outs), int(self._outz))
        return x

    def forward_(self, x: Tensor) -> Tensor:
        """
        堆叠各层，定义前向传播，相当于之前的forward(x)
        """
        raise NotImplementedError("Implement forward_(x) like you implemented forward(x) before")

    def quantize(self, training_data: Tensor, out_cover_f: float = 0.75):
        """
        量化模型，首先使用training_data,training_label推理一次拿到每一层的输出，这样便可以量化每一层的输出。
        然后再量化权重和偏置。
        Note: 这个方法必须在模型训练好后调用
        :param training_data: 用于确定每一层输出范围的训练集数据
        :param out_cover_f: 覆盖多少比例(0,1]，理论上应该取所有training_data中最大值的最大值和最小值的最小值，这样会导致范围过大，使得
                            表示精度降低，因此需要舍弃一些极端的training_data
        """
        if self._mode != __float_mode__:
            raise ValueError("quantize() must be called in float mode")
        # 推理一次获得每一层的输出
        self.forward(training_data)

        # 获取层序列，因为前一层的输出和后一层的输入scale和zero_point是一样的
        module_list, module_name_list = [], []
        for named_module in self.named_modules():
            name, module = named_module[0], named_module[1]
            if name == "":
                continue
            module_list.append(module)
            module_name_list.append(name)

        # 逐层量化
        for i in range(len(module_list)):
            name, module = module_name_list[i], module_list[i]

            # 量化输入
            if i == 0:
                # 第一层的输入就是training_data
                ins, inz = self._cal_sz(training_data, out_cover_f)
                self._ins, self._inz = torch.tensor(ins), torch.tensor(inz)
            else:
                # 其它层的输入和前一层的输出scale和zero_point是一样的
                pre_module = module_list[i - 1]
                ins, inz = float(pre_module.outs), int(pre_module.outz)
            module.ins = torch.tensor(ins)
            module.inz = torch.tensor(inz)

            # 量化输出
            if not hasattr(module, "float_out"):
                raise AttributeError("Module {} does not have attr float_out, Please use modules provided by quantize package.".format(name))
            out = module.float_out
            outs, outz = self._cal_sz(out, out_cover_f)
            if i == len(module_list) - 1:
                self._outs, self._outz = torch.tensor(outs), torch.tensor(outz)
            module.outs = torch.tensor(outs)
            module.outz = torch.tensor(outz)

            # 量化权重,收缩系数和零点根据tensor的最大最小值决定
            weight = module.weight
            quan_weight, ws, wz = utils.float2fixed_point_tensor(weight, self.n_bits)
            module.quan_weight = quan_weight
            module.ws = torch.tensor(ws)
            module.wz = torch.tensor(wz)

            # 量化偏置，收缩系数为权重和特征的乘积，零点是0，这里量化为多少位无所谓，只要能保证存的下就可以
            bias = module.bias
            if bias is not None:
                bs, bz = ins * ws, 0
                quan_bias = utils.float2fixed_point_SZ_tensor(bias, bs, bz)
                module.quan_bias = quan_bias
                module.bs = torch.tensor(bs)
                module.bz = torch.tensor(bz)

            # 量化M
            M = ws * ins / outs
            mn, mqm0 = utils.quantize_M(M, self.n_bits)  # 这里把M0[0.5, 1)量化为和整体相同
            module.mn = torch.tensor(mn)
            module.mqm0 = torch.tensor(mqm0)

            module.n_bits = torch.tensor(self.n_bits)
        pass

    # 计算输出的scale和zero_point
    def _cal_sz(self, layer_out: Tensor, out_cover_f: float) -> (float, int):
        a_list, b_list = [], []
        for i in range(len(layer_out)):
            t = layer_out[i]
            a_list.append(float(t.min()))
            b_list.append(float(t.max()))
        a_list.sort()
        b_list.sort()
        selected_a = a_list[int(len(a_list) * (1 - out_cover_f))]
        selected_b = b_list[int(len(a_list) * out_cover_f)]
        s, z = utils.scale_zero_point(selected_a, selected_b, self.n_bits)
        return s, z

    # 切换计算模式
    def _switch_mode(self, new_mode: str):
        for named_module in self.named_modules():
            name, module = named_module[0], named_module[1]
            if name == "":
                continue
            module.switch_mode(new_mode)

class QuanConv2d(nn.Conv2d):
    """
    支持量化前向推理的Conv2d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1,
                 groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self._mode: str = __float_mode__  # 计算模式
        self.float_out: Tensor  # float模式下的输出，记录下来用来确定量化输出时的scale和zero_point

        # 注册输入特征图、卷积核、偏置、输出的scale和zero_point、M的量化信息
        quan_buffers = ["ins", "inz", "outs", "outz", "ws", "wz", "bs", "bz", "mn", "mqm0", "n_bits"]
        for it in quan_buffers:
            self.register_buffer(it, torch.tensor(0.))

        # 量化的weight、bias
        self.register_buffer("quan_weight", torch.randn(self.weight.size()))
        if self.bias is not None:
            self.register_buffer("quan_bias", torch.randn(self.bias.size()))

    def switch_mode(self, new_mode: str):
        """
        改变计算模式
        """
        self._mode = new_mode

    def relu(self, x: Tensor, inplace: bool = False) -> Tensor:
        """
        支持量化的relu
        """
        if self._mode == __quan_mode__:
            outz = float(self.outz)
            return torch.where(x > outz, x, torch.ones(x.size()).to(x.device) * outz)
        elif self._mode == __float_mode__:
            return F.relu(x, inplace)
        else:
            raise RuntimeError("Unknown mode {}".format(self._mode))

    def forward(self, x: Tensor) -> Tensor:
        if self._mode == __quan_mode__:
            return self._quan_forward(x)
        elif self._mode == __float_mode__:
            return self._float_forward(x)
        else:
            raise RuntimeError("Unknown mode {}".format(self._mode))

    def _float_forward(self, x: Tensor) -> Tensor:
        """
        float模式下的前向
        """
        if self._mode != __float_mode__:
            raise RuntimeError("_float_forward must be called in {} mode".format(__float_mode__))
        out = super().forward(x)
        # 记录输出值
        self.float_out = out
        return out

    def _quan_forward(self, x: Tensor) -> Tensor:
        """
        quan模式下的前向
        """
        if self._mode != __quan_mode__:
            raise RuntimeError("_quan_forward must be called in {} mode".format(__quan_mode__))

        if self.quan_weight is None or self.quan_weight.size() != self.weight.size():
            raise RuntimeError("Please call QuanNet.quantize() first")

        inz = int(self.inz)
        outz = float(self.outz)
        wz = int(self.wz)
        mn = int(self.mn)
        mqm0 = int(self.mqm0)
        n_bits = int(self.n_bits)

        # 输入特征图填充，这里的0是inz
        if self.padding:
            pd = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])  # padding (left,right,top,bottom)
            x = F.pad(x, pd, "constant", inz)

        # 整数卷积 不需要再加padding
        out = F.conv2d(x, self.quan_weight, self.quan_bias, self.stride, 0, self.dilation, self.groups)
        N = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        out = torch.add(out, N * inz * wz)

        # 参与一次卷积的所有x的和
        weight = torch.ones(self.quan_weight.size()).to(x.device)
        bias = torch.zeros(self.quan_bias.size()).to(x.device)
        xsum = F.conv2d(x, weight, bias, self.stride, 0, self.dilation, self.groups)
        xsum = torch.mul(xsum, wz)
        out = torch.sub(out, xsum)

        # 参与一次卷积的所有权重的和
        ones = torch.ones(x.size()).to(x.device)
        bias = torch.zeros(self.quan_bias.size()).to(x.device)
        wsum = F.conv2d(ones, self.quan_weight, bias, self.stride, 0, self.dilation, self.groups)
        wsum = torch.mul(wsum, inz)
        out = torch.sub(out, wsum)

        # 乘M
        out = torch.mul(out, mqm0)  # 乘qm0
        out = out >> (mn + n_bits)

        # 加零点
        out = torch.add(out, outz)

        return out

class QuanLinear(nn.Linear):
    """
    支持量化前推理的Linear
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        self._mode: str = __float_mode__  # 计算模式
        self.float_out: Tensor  # float模式下的输出，记录下来用来确定量化输出时的scale和zero_point

        # 注册输入特征图、卷积核、偏置、输出的scale和zero_point、M的量化信息、量化的weight、bias
        self.quan_buffers = ["ins", "inz", "outs", "outz", "ws", "wz", "bs", "bz", "mn", "mqm0", "n_bits"]
        for it in self.quan_buffers:
            self.register_buffer(it, torch.tensor(0.))

        # 量化的weight、bias
        self.register_buffer("quan_weight", torch.randn(self.weight.size()))
        if self.bias is not None:
            self.register_buffer("quan_bias", torch.randn(self.bias.size()))

    def switch_mode(self, new_mode: str):
        """
        改变计算模式
        """
        self._mode = new_mode

    def relu(self, x: Tensor, inplace: bool = False) -> Tensor:
        """
        支持量化的relu
        """
        if self._mode == __quan_mode__:
            outz = float(self.outz)
            return torch.where(x > outz, x, torch.ones(x.size()).to(x.device) * outz)
        elif self._mode == __float_mode__:
            return F.relu(x, inplace)
        else:
            raise RuntimeError("Unknown mode {}".format(self._mode))

    def forward(self, x: Tensor) -> Tensor:
        if self._mode == __quan_mode__:
            return self._quan_forward(x)
        elif self._mode == __float_mode__:
            return self._float_forward(x)
        else:
            raise RuntimeError("Unknown mode {}".format(self._mode))

    def _float_forward(self, x: Tensor) -> Tensor:
        """
        float模式下的前向
        """
        if self._mode != __float_mode__:
            raise RuntimeError("_float_forward must be called in {} mode".format(__float_mode__))
        out = super().forward(x)
        # 记录输出值
        self.float_out = out
        return out

    def _quan_forward(self, x: Tensor) -> Tensor:
        """
        quan模式下的前向
        """
        if self._mode != __quan_mode__:
            raise RuntimeError("_quan_forward must be called in {} mode".format(__quan_mode__))
        if self.quan_weight is None or self.quan_weight.size() != self.weight.size():
            raise RuntimeError("Please call QuanNet.quantize() first")

        inz = int(self.inz)
        outz = float(self.outz)
        wz = int(self.wz)
        mn = int(self.mn)
        mqm0 = int(self.mqm0)
        n_bits = int(self.n_bits)

        # 整数Mv
        out = F.linear(x, self.quan_weight, self.quan_bias)
        N = self.in_features
        out = torch.add(out, N * inz * wz)

        # 权重置为1
        weight = torch.ones(self.quan_weight.size()).to(x.device)
        bias = torch.zeros(self.quan_bias.size()).to(x.device)
        xsum = F.linear(x, weight, bias)
        xsum = torch.mul(xsum, wz)
        out = torch.sub(out, xsum)

        # 输入置为1
        ones = torch.ones(x.size()).to(x.device)
        bias = torch.zeros(self.quan_bias.size()).to(x.device)
        wsum = F.linear(ones, self.quan_weight, bias)
        wsum = torch.mul(wsum, inz)
        out = torch.sub(out, wsum)

        # 乘M
        out = torch.mul(out, mqm0)  # 乘qm0
        out = out >> (mn + n_bits)

        # 加零点
        out = torch.add(out, outz)

        return out

if __name__ == '__main__':
    pass
