"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 4, 2020 at 4:29:28 PM
    -----------------------------------
    存储知识嵌入方案
"""
import torch.nn as nn
import torch


class NormalGRU(nn.Module):
    def __init__(self, fact_len, rnn_layers, hidden_size, dropout):
        super(NormalGRU, self).__init__()

        self.fact_len = fact_len
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=rnn_layers,
                          batch_first=True,
                          dropout=dropout)

        self.alpha = torch.tensor(data=[0.5], requires_grad=True)  # 动态调节参数

    def forward(self, facts):
        batch_size = facts.size(0)

        # 双层RNN
        self.gru.flatten_parameters()
        facts = facts.view((-1, self.fact_len, self.hidden_size))
        feats = self.gru(facts)[1]

        # 加权和
        sum_feats = self.alpha * feats[0] + (1 - self.alpha) * feats[1]
        return sum_feats.view(batch_size, -1, self.hidden_size)

