"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 8, 2020 at 9:48:21 PM
    -----------------------------------
    单词级别特征层
"""
import torch.nn as nn
from misc.helper import fact_extract


class WordLayer(nn.Module):
    def __init__(self, hidden_size, img_attn, word_attn, mlp, fact_attn, pooling, dropout):
        super(WordLayer, self).__init__()
        self.hidden_size = hidden_size
        self.img_attn = img_attn
        self.word_attn = word_attn
        self.mlp = mlp
        self.fact_attn = fact_attn
        self.pooling = pooling
        self.dropout = dropout

    def forward(self, word_embeddings, word_masks, img_embeddings, fact_embeddings, fact_masks):
        # 计算词级别特征
        img_att_w = self.img_attn(img_embeddings, word_embeddings, word_embeddings, word_masks)  # 图像特征
        ques_att_w = self.word_attn(word_embeddings, img_att_w, img_att_w)  # 文本特征

        img_att_w, ques_att_w = self.pooling(img_att_w), self.pooling(ques_att_w)

        feat_w = nn.Dropout(self.dropout)(img_att_w + ques_att_w)
        feat_w = nn.Tanh()(self.mlp(feat_w))    # 整合图像和文本特征

        bank_w = fact_extract(fact_embeddings, fact_masks, [img_att_w, ques_att_w, feat_w], self.fact_attn, self.pooling)

        return feat_w, bank_w
