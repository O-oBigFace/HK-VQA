"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 3, 2020 at 10:59:18 AM
    -----------------------------------
    模型整合

    工作：记录需要的超参数
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from misc.language_embeddings import RandomEmbedding
from misc.image_embeddings import IMGLinear
from misc.knowledge_embeddings import NormalGRU
from misc.multiheaded_attention import MultiHeadedAttenton
from misc.helper import clones, fact_extract


def choosing_w2v(type_):
    if type_ is "random":
        return RandomEmbedding


def choosing_i2v(type_):
    if type_ is "linear":
        return IMGLinear


def cal_fact_mask(fact_count, mask):
    """
        计算fact mask，前N个为有效fact
        ------------------------------------------
        Args:
            fact_count: 每个样例的fact数量
            mask: mask值
        Returns:
    """
    batch_size = fact_count.size()[0]
    for i in range(batch_size):
        mask[i, 0:fact_count[i]] = False


class VQAModel(nn.Module):
    def __init__(self, opts):
        """
            模型整合类
            ------------------------------------------
            Args:
                opts:
                    hidden_size: 模型隐层特征维度
                    dropout

                    vocab_size*: lookup table 大小
                    w2v_type
                    seq_len: (最大)问题长度
                    rnn_layers

                    img_feat_dim*: 图像特征维度
                    i2v_type
                    img_len*: 图像分割区域数


                    fact_num: (最大)事实数
                    fact_len: (最大)事实长度

                    nheaders

                    nanswers
            Returns:
        """
        super(VQAModel, self).__init__()
        self.dropout = opts['dropout']
        self.seq_len = opts['seq_len']
        self.hidden_size = opts['hidden_size']
        self.batch_size = opts['batch_size']
        self.fact_num = opts['fact_num']

        # 基本特征转换
        self.layer_w2v = choosing_w2v(opts['w2v_type'])(vocab_size=opts['vocab_size'],
                                                        hidden_size=self.hidden_size,
                                                        dropout=self.dropout)

        self.layer_i2v = choosing_i2v(opts['i2v_type'])(img_feat_dim=opts['img_feat_dim'],
                                                        hidden_size=self.hidden_size,
                                                        dropout=self.dropout)

        self.fact_embedder = NormalGRU(fact_len=opts['fact_len'],
                                       rnn_layers=opts['rnn_layers'],
                                       hidden_size=self.hidden_size,
                                       dropout=self.dropout)
        self.dense_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooling1d = lambda x: x.permute(0, 2, 1).max(dim=-1)[0]

        # 注意力机制
        self.attens = clones(MultiHeadedAttenton(h=opts['nheaders'],
                                                 d_model=self.hidden_size,
                                                 dropout=self.dropout), 9)

        # 短语特征抽取
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

        self.dense_p = nn.Linear(2 * self.hidden_size, self.hidden_size)

        # 句子特征抽取
        self.bilstm = nn.LSTM(input_size=self.hidden_size,
                              hidden_size=self.hidden_size // 2,
                              num_layers=opts['rnn_layers'],
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)
        self.dense_q = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.gate = torch.tensor([0.5] * self.hidden_size, requires_grad=True)

        self.predict_layer = nn.Sequential(nn.Linear(self.hidden_size, opts['nanswers']),
                                           nn.Softmax(dim=1))

    def forward(self, questions, images, facts, fact_count):
        """
            前馈函数
            ------------------------------------------
            Args:
                questions: 问题，应转化为id形式 (batch_size, question_len)
                images: 图像此时接收的应是*图像特征*
                facts: 转换成id的事实，这里的事实搜索算法可以研究 (batch_size, fact_num, fact_len)
                fact_count: 每个训练用例对应的事实数量 (batch_size, )
            Returns:
        """

        ques_mask = torch.zeros_like(questions, dtype=torch.bool)  # 计算问句的mask
        ques_mask[torch.eq(questions, 0)] = True
        ques_mask.unsqueeze_(1)  # 计算atten时，需扩张一个维度

        fact_mask = torch.ones((self.batch_size, self.fact_num), dtype=torch.bool)  # 生成fact的mask
        cal_fact_mask(fact_count, fact_mask)
        fact_mask.unsqueeze_(1)

        word_embeds = self.layer_w2v(questions)  # 计算词级别问句特征
        img_embeds = self.layer_i2v(images)  # 图像特征后处理
        fact_embeds = self.layer_w2v(facts)
        fact_embeds = self.fact_embedder(fact_embeds)

        # 计算词级别特征
        img_att_w = self.attens[0](img_embeds, word_embeds, word_embeds, ques_mask)  # 图像特征
        ques_att_w = self.attens[1](word_embeds, img_att_w, img_att_w)  # 文本特征

        img_att_w, ques_att_w = self.pooling1d(img_att_w), self.pooling1d(ques_att_w)

        feat_w = nn.Dropout(self.dropout)(img_att_w + ques_att_w)
        feat_w = nn.Tanh()(self.dense_w(feat_w))    # 整合图像和文本特征

        mem_bank_w = fact_extract(fact_embeds, fact_mask, [img_att_w, ques_att_w, feat_w], self.attens[2], self.pooling1d)

        # 计算短语级别特征
        word_embeds_permute = word_embeds.permute(0, 2, 1)
        unigram = self.conv_unigram(word_embeds_permute)
        bigram = self.conv_bigram(word_embeds_permute).narrow(2, 0, self.seq_len)
        trigram = self.conv_trigram(word_embeds_permute)

        unigram = unigram.permute(0, 2, 1).view(-1, self.seq_len, self.hidden_size, 1)
        bigram = bigram.permute(0, 2, 1).view(-1, self.seq_len, self.hidden_size, 1)
        trigram = trigram.permute(0, 2, 1).view(-1, self.seq_len, self.hidden_size, 1)

        context_feat = torch.cat((unigram, bigram, trigram), dim=-1)
        context_feat = context_feat.max(dim=-1)[0]

        img_att_p = self.attens[3](img_embeds, context_feat, context_feat, ques_mask)
        ques_att_p = self.attens[4](context_feat, img_att_p, img_att_p)

        img_att_p, ques_att_p = self.pooling1d(img_att_p), self.pooling1d(ques_att_p)

        feat_p = nn.Dropout(self.dropout)(torch.cat((img_att_p + ques_att_p, feat_w), dim=1))
        feat_p = nn.Tanh()(self.dense_p(feat_p))    # 整合图像和文本特征

        mem_bank_p = fact_extract(fact_embeds, fact_mask, [img_att_p, ques_att_p, feat_p], self.attens[5], self.pooling1d)

        # 计算句子级别特征
        self.bilstm.flatten_parameters()
        ques_feat, _ = self.bilstm(word_embeds)

        img_att_q = self.attens[6](img_embeds, ques_feat, ques_feat)
        ques_att_q = self.attens[7](ques_feat, img_att_q, img_att_q)
        img_att_q, ques_att_q = self.pooling1d(img_att_q), self.pooling1d(ques_att_q)

        feat_q = nn.Dropout(self.dropout)(torch.cat((img_att_q + ques_att_q, feat_p), dim=1))
        feat_q = nn.Tanh()(self.dense_q(feat_q))

        mem_bank_q = fact_extract(fact_embeds, fact_mask, [img_att_q, ques_att_q, feat_q], self.attens[8], self.pooling1d)

        # 融合
        mem_bank = nn.Dropout(self.dropout)(mem_bank_w + mem_bank_p + mem_bank_q)

        feat_q = F.normalize(input=feat_q, p=2, dim=1)
        mem_bank = F.normalize(mem_bank, p=2, dim=1)

        feat = nn.Dropout(self.dropout)(feat_q * self.gate + mem_bank * (1-self.gate))

        out = self.predict_layer(feat)
        return out


if __name__ == '__main__':
    batch_size = 32
    seq_len = 26
    hidden_size = 512
    ques_feat = torch.randn(size=(batch_size, seq_len, hidden_size))
    ques_feat = ques_feat.permute(0, 2, 1).max(dim=-1)[0]

    mb = torch.randn((batch_size, hidden_size))
    print(mb)
    mb = F.normalize(mb, p=2, dim=1)
    print(mb)