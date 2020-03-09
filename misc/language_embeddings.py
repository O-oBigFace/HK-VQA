"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 3, 2020 at 11:11:32 AM
    -----------------------------------
    包含各类文本特征处理函数

    RandomEmbedding: 随机嵌入
"""

import torch.nn as nn


class RandomEmbedding(nn.Module):  # 随机嵌入
    def __init__(self, vocab_size, hidden_size, dropout):
        super(RandomEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sents):
        word_feat = self.embed(sents)
        return self.dropout(nn.Tanh()(word_feat))


def choose_w2v(type_):
    if type_ is "random":
        return RandomEmbedding
