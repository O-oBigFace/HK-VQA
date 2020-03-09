"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 8, 2020 at 10:09:46 PM
    -----------------------------------
    模型框架
"""
import torch.nn as nn
from misc.language_embeddings import choose_w2v
from misc.image_embeddings import choose_i2v
from misc.knowledge_embeddings import NormalGRU
from misc.word_layer import WordLayer
from misc.multiheaded_attention import MultiHeadedAttenton
from misc.phrase_cnn import PhraseCNN
from misc.phrase_layer import PhraseLayer
from misc.ques_layer import QuesLayer
from misc.generator import Generator
from misc.fusing_layer import FusingLayer
from copy import deepcopy


class VQAModel(nn.Module):
    def __init__(self, lang_embed, img_embed, fact_embed, word_layer, phrase_layer, ques_layer, fuse_layer, generator):
        super(VQAModel, self).__init__()
        self.lang_embed = lang_embed
        self.img_embed = img_embed
        self.fact_embed = fact_embed

        self.word_layer = word_layer
        self.phrase_layer = phrase_layer
        self.ques_layer = ques_layer

        self.fuse_layer = fuse_layer

        self.generator = generator

    def forward(self, ques, ques_masks, img, fact, fact_masks):
        """
            定义模型的输入输出
            ------------------------------------------
            Args:
            Returns:
        """
        lang_embeddings = self.lang_embed(ques)
        img_embeddings = self.img_embed(img)
        fact_embeddings = self.fact_embed(self.lang_embed(fact))

        feat_w, bank_w = self.word_layer(lang_embeddings,
                                         ques_masks,
                                         img_embeddings,
                                         fact_embeddings,
                                         fact_masks)

        feat_p, bank_p = self.phrase_layer(lang_embeddings,
                                           ques_masks,
                                           img_embeddings,
                                           fact_embeddings,
                                           fact_masks,
                                           feat_w,
                                           bank_w)

        feat_q, bank_q = self.ques_layer(lang_embeddings,
                                         ques_masks,
                                         img_embeddings,
                                         fact_embeddings,
                                         fact_masks,
                                         bank_w,
                                         feat_p,
                                         bank_p)

        feat = self.fuse_layer(feat_q, (bank_w, bank_p, bank_q))

        out = self.generator(feat)
        return out


def make_model(opts):
    attn = MultiHeadedAttenton(opts["nheaders"], opts["hidden_size"])
    pooling = lambda x: x.permute(0, 2, 1).max(dim=-1)[0]

    model = VQAModel(lang_embed=choose_w2v(opts['w2v_type'])(opts["vocab_size"], opts["hidden_size"], opts["dropout"]),
                     img_embed=choose_i2v(opts['i2v_type'])(opts["img_feat_dim"], opts["hidden_size"], opts["dropout"]),
                     fact_embed=NormalGRU(opts['fact_len'], opts['rnn_layers'], opts["hidden_size"], opts["dropout"]),
                     word_layer=WordLayer(opts["hidden_size"], deepcopy(attn), deepcopy(attn),
                                          nn.Linear(opts["hidden_size"], opts["hidden_size"]),
                                          deepcopy(attn), pooling, opts["dropout"]),
                     phrase_layer=PhraseLayer(opts["hidden_size"],
                                              PhraseCNN(opts["hidden_size"], opts["seq_len"]),
                                              deepcopy(attn), deepcopy(attn),
                                              nn.Linear(2*opts["hidden_size"], opts["hidden_size"]),
                                              deepcopy(attn), pooling, opts["dropout"]),
                     ques_layer=QuesLayer(opts["hidden_size"],
                                          nn.LSTM(input_size=opts["hidden_size"],
                                                  hidden_size=opts["hidden_size"] // 2,
                                                  num_layers=opts['rnn_layers'],
                                                  batch_first=True,
                                                  dropout=opts["dropout"],
                                                  bidirectional=True),
                                          deepcopy(attn), deepcopy(attn),
                                          nn.Linear(2 * opts["hidden_size"], opts["hidden_size"]),
                                          deepcopy(attn), pooling, opts["dropout"]),
                     fuse_layer=FusingLayer(opts["hidden_size"], opts["dropout"]),
                     generator=Generator(opts["hidden_size"], opts["nanswers"])
                     )
    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
