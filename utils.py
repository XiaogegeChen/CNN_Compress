import torch

# 计算缩放系数S和零点Z
def scale_zero_point(num_min: float, num_max: float, n_bits: int):
    S = (num_max - num_min) / ((2 ** n_bits) - 1)
    Z = round(-num_min / S)
    return S, Z

# 浮点数转定点数
def float2fixed_point(num: float, n_bits: int, num_min: float, num_max: float) -> (int, float, int, str):
    num = _clamp(num, num_min, num_max)
    S, Z = scale_zero_point(num_min, num_max, n_bits)
    fixed_point = round(num / S) + Z
    return fixed_point, S, Z, _fixed_point2str(fixed_point, n_bits)

# 定点数转浮点数
def fixed_point2float(fixed_point: int, S: float, Z: int) -> float:
    return S * (fixed_point - Z)

# 浮点数转定点数(tensor形式)
def float2fixed_point_tensor(t: torch.Tensor, n_bits: int, num_min: float = None, num_max: float = None) -> (torch.Tensor, float, int):
    if num_min is None:
        num_min = t.min().tolist()
    if num_max is None:
        num_max = t.max().tolist()
    S, Z = scale_zero_point(num_min, num_max, n_bits)
    new_t = torch.clamp(t, num_min, num_max)
    new_t = torch.div(new_t, S)
    new_t = torch.round(new_t)
    new_t = torch.add(new_t, Z)
    return new_t, S, Z

def float2fixed_point_SZ_tensor(t: torch.Tensor, S: float, Z: int):
    new_t = torch.div(t, S)
    new_t = torch.round(new_t)
    new_t = torch.add(new_t, Z)
    return new_t

# 定点数转浮点数(tensor形式)
def fixed_point2float_tensor(t: torch.Tensor, S: float, Z: int) -> torch.Tensor:
    new_t = torch.sub(t, Z)
    new_t = torch.mul(new_t, S)
    return new_t

# 量化收缩系数的比值M=S1*S2/S3
def quantize_M(M: float, n_m0_bits: int) -> (int, int):
    n = 0
    M0 = M
    while M0 < 0.5:
        M0 = M0 * 2
        n += 1
    qM0 = round(M0 * (2 ** n_m0_bits))
    return n, qM0

# 两个量化后的值相乘得到新的量化后的值，q3=q1*q2
def quantize_mul(q1: int, q2: int, m: int, q_bits: int, m_bits: int, z1: int, z2: int, z3: int) -> int:
    return (int((q1 - z1) * (q2 - z2) * m) >> (q_bits + m_bits)) + z3

# 两个量化的1维张量点乘得到新的量化的值,q3=v1 dotproduct v2
def quantize_dot(t1: torch.Tensor, t2: torch.Tensor, m: int, q_bits: int, m_bits: int, z1: int, z2: int, z3: int) -> int:
    idot = torch.dot(torch.sub(t1, z1), torch.sub(t2, z2))
    return (int(idot.tolist() * m) >> (q_bits + m_bits)) + z3

def _clamp(num, num_min, num_max):
    if num < num_min:
        return num_min
    if num > num_max:
        return num_max
    return num

def _fixed_point2str(fixed_point: int, n_bits: int) -> str:
    if fixed_point < 0:
        fixed_point += 2 ** n_bits
    fp_str = bin(fixed_point)
    fp_str = fp_str[2:]

    if len(fp_str) == n_bits:
        return fp_str

    blank_bit = n_bits - len(fp_str)
    fp_str = "0" * blank_bit + fp_str
    return fp_str
