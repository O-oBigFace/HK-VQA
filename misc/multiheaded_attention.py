"""
    author: W J-H (jiangh_wu@163.com)
    time: Feb 2, 2020
    -----------------------------------
    实现Multi-headed Attention
"""
import torch.nn as nn
from misc.helper import clones, attention


class MultiHeadedAttenton(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttenton, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 将一个batch中的所有特征做映射：d_model => h * d_k
        query, key, value = [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linears, (query, key, value))]

        # 2) 计算attention
        x, self.attn = attention(query, key, value, mask, self.dropout)

        # 3) 利用view来实现 concat
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
