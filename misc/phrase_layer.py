"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 8, 2020 at 11:23:24 PM
    -----------------------------------
    短语级别
"""
import torch.nn as nn
import torch
from misc.helper import fact_extract


class PhraseLayer(nn.Module):
    def __init__(self, hidden_size, cnn, img_attn, word_attn, mlp, fact_attn, pooling, dropout):
        super(PhraseLayer, self).__init__()
        self.hidden_size = hidden_size
        self.cnn = cnn
        self.img_attn = img_attn
        self.word_attn = word_attn
        self.mlp = mlp
        self.fact_attn = fact_attn
        self.pooling = pooling
        self.dropout = dropout

    def forward(self, lang_embeddings, ques_masks, img_embeddings, fact_embeddings, fact_masks, feat_w, bank_w):
        context_feat = self.cnn(lang_embeddings)

        img_att_p = self.img_attn(img_embeddings, context_feat, context_feat, ques_masks)
        ques_att_p = self.word_attn(context_feat, img_att_p, img_att_p)

        img_att_p, ques_att_p = self.pooling(img_att_p), self.pooling(ques_att_p)

        feat_p = nn.Dropout(self.dropout)(torch.cat((img_att_p + ques_att_p, feat_w), dim=1))
        feat_p = nn.Tanh()(self.mlp(feat_p))    # 整合图像和文本特征

        bank_p = fact_extract(fact_embeddings, fact_masks, [img_att_p, ques_att_p, feat_p, bank_w],  self.fact_attn, self.pooling)

        return feat_p, bank_p
