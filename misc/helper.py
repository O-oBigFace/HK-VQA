"""
    author: W J-H (jiangh_wu@163.com)
    time: Feb 1, 2020
    -----------------------------------
    一些辅助函数
"""

import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt


def clones(module, N):
    """
        生成N个相同的层次
        ------------------------------------------
        Args:
            module:
            N:
        Returns:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask == 1, -1e9)  # 值为1算是mask
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, value), scores


def fact_extract(fact_embeds, fact_masks, feats: list, atten, pooling):
    comb_feat = torch.cat([x.unsqueeze(1) for x in feats], dim=1)
    feat = atten(comb_feat, fact_embeds, fact_embeds, fact_masks)
    return pooling(feat)
