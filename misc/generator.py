"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 9, 2020 at 12:03:46 AM
    -----------------------------------
    生成结果
"""
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
        结果生成器
    """
    def __init__(self, d_model, nanswers):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, nanswers)   # 将隐层向量映射到词汇上

    def forward(self, x):
        """
        :param x: (batch_size, d_model)
        :return:
        """
        return F.log_softmax(self.proj(x), dim=1)
