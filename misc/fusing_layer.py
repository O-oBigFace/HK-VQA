"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 9, 2020 at 12:07:21 AM
    -----------------------------------
    融合层
"""
import torch
import torch.nn as nn


class FusingLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(FusingLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout= dropout
        self.gate = torch.tensor([0.5] * self.hidden_size, requires_grad=True)

    def forward(self, feat_q, facts: list):
        bank = facts[0] + facts[1] + facts[2]
        return nn.Dropout(self.dropout)(feat_q * self.gate + bank * (1-self.gate))

