"""
    author: W J-H (jiangh_wu@163.com)
    time: Feb 1, 2020
    -----------------------------------
    定义残差连接
"""
import torch.nn as nn
from misc.norm import LayerNorm


class SublayerConnection(nn.Module):
    """
        残差连接层
        注意标准化是第一个进行的
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
