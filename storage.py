import torch
import yaml
import utils
from typing import List
from collections import OrderedDict

class Model:
    def __init__(self):
        self.head = ""
        self.layers = []

    def map(self):
        return {
            "head": self.head,
            "layers": [layer.map() for layer in self.layers]
        }

class QuanModel(Model):
    def __init__(self):
        super().__init__()
        self.ins = 0.
        self.inz = 0
        self.outs = 0.
        self.outz = 0

    def map(self):
        return dict(super(QuanModel, self).map(), **{
            "ins": self.ins,
            "inz": self.inz,
            "outs": self.outs,
            "outz": self.outz,
        })

    def load_from(self, state_dict: 'OrderedDict'):
        self.ins = float(state_dict["_ins"])
        self.inz = int(state_dict["_inz"])
        self.outs = float(state_dict["_outs"])
        self.outz = int(state_dict["_outz"])

class Tensor:
    def __init__(self):
        self.size = []
        self.data = []

    def map(self):
        return {
            "size": self.size,
            "data": self.data
        }

    @staticmethod
    def convert(t: torch.Tensor) -> 'Tensor':
        ret = Tensor()
        if t is not None:
            ret.size = list(t.size())
            ret.data = t.tolist()
        return ret

class COOTensor:
    def __init__(self):
        self.size = []
        self.indices = []  # 二维
        self.values = []
        self.nnz = 0

    def map(self):
        return {
            "size": self.size,
            "indices": self.indices,
            "values": self.values,
            "nnz": self.nnz
        }

    @staticmethod
    def convert(t: torch.Tensor, quan=False) -> 'COOTensor':
        ret = COOTensor()
        if t is not None:
            ret.size = list(t.size())
            ret.indices = t.indices().tolist()
            ret.values = t.values().int().tolist() if quan else t.values().tolist()
            ret.nnz = int(t._nnz())
        return ret

class Layer:
    def __init__(self):
        self.name = ""
        self.weight = None
        self.bias = None

    def map(self):
        ret = {
            "name": self.name,
            "weight": self.weight.map(),
        }
        if self.bias is not None:
            ret = dict(ret, **{
                "bias": self.bias.map()
            })
        return ret

    def load_from(self, state_dict: 'OrderedDict', layer_name: str):
        self.name = layer_name
        self.weight = Tensor.convert(state_dict[layer_name + ".weight"])
        self.bias = Tensor.convert(state_dict.get(layer_name + ".bias", None))

class QuanLayer(Layer):
    def __init__(self):
        super().__init__()
        self.ins = 0.
        self.inz = 0
        self.outs = 0.
        self.outz = 0
        self.ws = 0.
        self.wz = 0
        self.bs = 0.
        self.bz = 0
        self.mn = 0
        self.mqm0 = 0
        self.n_bits = 0

    def map(self):
        ret = dict(super(QuanLayer, self).map(), **{
            "ins": self.ins,
            "inz": self.inz,
            "outs": self.outs,
            "outz": self.outz,
            "ws": self.ws,
            "wz": self.wz,
            "bs": self.bs,
            "bz": self.bz,
            "mn": self.mn,
            "mqm0": self.mqm0,
            "n_bits": self.n_bits
        })
        return ret

    def load_from(self, state_dict: 'OrderedDict', layer_name: str):
        self.name = layer_name
        self.weight = Tensor.convert(state_dict[layer_name + ".quan_weight"].int())
        bias = state_dict.get(layer_name + ".quan_bias", None)
        bias = bias.int() if bias is not None else bias
        self.bias = Tensor.convert(bias)
        self.ins = float(state_dict[layer_name + ".ins"])
        self.inz = int(state_dict[layer_name + ".inz"])
        self.outs = float(state_dict[layer_name + ".outs"])
        self.outz = int(state_dict[layer_name + ".outz"])
        self.ws = float(state_dict[layer_name + ".ws"])
        self.wz = int(state_dict[layer_name + ".wz"])
        self.bs = float(state_dict[layer_name + ".bs"])
        self.bz = int(state_dict[layer_name + ".bz"])
        self.mn = int(state_dict[layer_name + ".mn"])
        self.mqm0 = int(state_dict[layer_name + ".mqm0"])
        self.n_bits = int(state_dict[layer_name + ".n_bits"])

def save_quan_model(model_path: str, float_model_path: str, int_model_path: str, binary_int: bool = False):
    """
    经过量化工具得到的模型中包含float参数和integer参数，这个方法将两者分开，并分别保存在两个.yml文件中
    :param model_path: 通过torch.save(net.state_dict(), f)保存的.pt文件
    :param float_model_path: float参数的模型文件
    :param int_model_path: integer参数的模型文件
    :param binary_int: integer是否保存为二进制，默认是十进制
    :return: None
    """
    state_dict = torch.load(model_path)
    # 网络中的所有层
    layer_name_list = _get_layer_name_list(state_dict)

    float_model_data = Model()
    float_model_data.head = "float_value"
    for layer_name in layer_name_list:
        layer = Layer()
        layer.load_from(state_dict, layer_name)
        float_model_data.layers.append(layer)
    with open(float_model_path, mode="w", encoding="utf-8") as f:
        yaml.dump(float_model_data.map(), f)

    int_model_data = QuanModel()
    int_model_data.head = "int_values"
    int_model_data.load_from(state_dict)
    for layer_name in layer_name_list:
        layer = QuanLayer()
        layer.load_from(state_dict, layer_name)
        int_model_data.layers.append(layer)
    with open(int_model_path, mode="w", encoding="utf-8") as f:
        yaml.dump(int_model_data.map(), f)

def save_sparse_model(model_path: str, float_model_path: str, int_model_path: str, binary_int: bool = False, form: str = "coo"):
    """
    保存稀疏模型，目前支持coo格式
    """
    if form == "coo":
        _coo_model(model_path, float_model_path, int_model_path, binary_int)
    else:
        raise ValueError("Unsupported form {}".format(form))

def _coo_model(model_path: str, float_model_path: str, int_model_path: str, binary_int: bool = False):
    state_dict = torch.load(model_path)
    layer_name_list = _get_layer_name_list(state_dict)

    float_model_data = Model()
    float_model_data.head = "float_value"
    for layer_name in layer_name_list:
        layer = Layer()
        weight = utils.dense_2_coo(state_dict[layer_name + ".weight"], 0.)
        layer.weight = COOTensor.convert(weight)
        bias = state_dict.get(layer_name + ".bias", None)
        if bias is not None:
            bias = utils.dense_2_coo(bias, 0.)
        layer.bias = COOTensor.convert(bias)
        float_model_data.layers.append(layer)
    with open(float_model_path, mode="w", encoding="utf-8") as f:
        yaml.dump(float_model_data.map(), f)

    int_model_data = QuanModel()
    int_model_data.head = "int_values"
    int_model_data.load_from(state_dict)
    for layer_name in layer_name_list:
        layer = QuanLayer()
        layer.load_from(state_dict, layer_name)
        weight = utils.dense_2_coo(state_dict[layer_name + ".quan_weight"], layer.wz)
        layer.weight = COOTensor.convert(weight, quan=True)
        bias = state_dict.get(layer_name + ".quan_bias", None)
        if bias is not None:
            bias = utils.dense_2_coo(bias, layer.bz)
        layer.bias = COOTensor.convert(bias, quan=True)
        int_model_data.layers.append(layer)
    with open(int_model_path, mode="w", encoding="utf-8") as f:
        yaml.dump(int_model_data.map(), f)

# 网络中的所有层，按照堆叠的顺序
def _get_layer_name_list(state_dict) -> List[str]:
    layer_name_list = []
    for k in state_dict.keys():
        if "." not in k:  # 排除_ins, _inz, _outs, _outz
            continue
        layer_name = k.split(".")[0]
        if layer_name not in layer_name_list:
            layer_name_list.append(layer_name)
    return layer_name_list

