from dataclasses import dataclass
from torch import nn
import torch


@dataclass
class ModelArgs:
    dim = 4096
    n_layers = 12
    n_heads = 32
    n_kv_heads = None
    vacab_size = -1
    multiple_of = 256
    ffn_dim_multiplier = None
    norm_eps = 1e-5
    max_batch_size = 32
    max_seq_len = 2048


# RMS = Root Mean Square
# 均方根
class RMSNorm(torch.nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 初始化神经网络的权重参数
        # weight 是 nn.Parameter 类型，是一个可学习的参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x.pow(2) [1,2,3] -> [1,4,9]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 预计算给点维度和结束索引的复杂指数的频率张量
# dim 频率张量的维度，end 预计算频率的结束索引 theta 频率计算的缩放因子
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2))[:(dim // 2)].float / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs, out=freqs)
    return freqs_cis


# 调整一个频率张量（freqs_cis）的形状，使其可以与另一个张量（x）进行广播
# 频率张量 -> 目标张量 x
def reshape_for_broadcast(frqes_cis, x):
    ndim = x.ndim
