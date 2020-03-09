"""
    author: W J-H (jiangh_wu@163.com)
    time: Feb 1, 2020
    -----------------------------------
    实现一些Normalization算法
"""
import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, feat_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(feat_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(feat_size), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
