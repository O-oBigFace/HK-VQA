"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 8, 2020 at 11:28:40 PM
    -----------------------------------
    提取短语特征
"""
import torch.nn as nn
import torch


class PhraseCNN(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super(PhraseCNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.conv_unigram = nn.Conv1d(in_channels=self.hidden_size,
                                      out_channels=self.hidden_size,
                                      kernel_size=1,
                                      stride=1)
        self.conv_bigram = nn.Conv1d(in_channels=self.hidden_size,
                                     out_channels=self.hidden_size,
                                     kernel_size=2,
                                     padding=1)
        self.conv_trigram = nn.Conv1d(in_channels=self.hidden_size,
                                      out_channels=self.hidden_size,
                                      kernel_size=3,
                                      padding=1)

    def forward(self, word_embeddings):
        word_embeddings = word_embeddings.permute(0, 2, 1)
        unigram = self.conv_unigram(word_embeddings)
        bigram = self.conv_bigram(word_embeddings).narrow(2, 0, self.seq_len)
        trigram = self.conv_trigram(word_embeddings)

        unigram = unigram.permute(0, 2, 1).view(-1, self.seq_len, self.hidden_size, 1)
        bigram = bigram.permute(0, 2, 1).view(-1, self.seq_len, self.hidden_size, 1)
        trigram = trigram.permute(0, 2, 1).view(-1, self.seq_len, self.hidden_size, 1)

        context_feat = torch.cat((unigram, bigram, trigram), dim=-1)
        return context_feat.max(dim=-1)[0]
