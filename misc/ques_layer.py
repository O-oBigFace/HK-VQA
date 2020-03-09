"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 8, 2020 at 11:42:18 PM
    -----------------------------------
    句子级别特征
"""
import torch.nn as nn
import torch
from misc.helper import fact_extract


class QuesLayer(nn.Module):
    def __init__(self, hidden_size, rnn, img_attn, word_attn, mlp, fact_attn, pooling, dropout):
        super(QuesLayer, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = rnn
        self.img_attn = img_attn
        self.word_attn = word_attn
        self.mlp = mlp
        self.fact_attn = fact_attn
        self.pooling = pooling
        self.dropout = dropout

    def forward(self, word_embeddings, word_masks, img_embeddings, fact_embeddings, fact_masks, bank_w, feat_p, bank_p):
        self.rnn.flatten_parameters()
        ques_feat, _ = self.rnn(word_embeddings)
        img_att_q = self.img_attn(img_embeddings, ques_feat, ques_feat)
        ques_att_q = self.word_attn(ques_feat, img_att_q, img_att_q)
        img_att_q, ques_att_q = self.pooling(img_att_q), self.pooling(ques_att_q)

        feat_q = nn.Dropout(self.dropout)(torch.cat((img_att_q + ques_att_q, feat_p), dim=1))
        feat_q = nn.Tanh()(self.mlp(feat_q))

        bank_q = fact_extract(fact_embeddings, fact_masks, [img_att_q, ques_att_q, feat_q, bank_w, bank_p], self.fact_attn, self.pooling)

        return feat_q, bank_q
