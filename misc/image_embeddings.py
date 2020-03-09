"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 4, 2020 at 10:35:07 AM
    -----------------------------------
    对图像特征后处理

    IMGLinear: 线性+tanh
"""
import torch.nn as nn


class IMGLinear(nn.Module):
    def __init__(self, img_feat_dim, hidden_size, dropout):
        super(IMGLinear, self).__init__()
        self.layer_dense = nn.Linear(img_feat_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        return self.dropout(nn.Tanh()(self.layer_dense(img)))


def choose_i2v(type_):
    if type_ is "linear":
        return IMGLinear
