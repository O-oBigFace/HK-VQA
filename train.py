"""
    author: W J-H (jiangh_wu@163.com)
    time: Mar 7, 2020 at 10:37:14 AM
    -----------------------------------
    模型训练
"""
from argparse import ArgumentParser
from datetime import datetime
import json
import torch
from os.path import exists
import os
from misc.vqa_model import make_model


def args_process():
    arg_parser = ArgumentParser()
    # 模型信息
    arg_parser.add_argument("--dropout",
                            default=0.5,
                            type=float,
                            help="dropout")
    arg_parser.add_argument("--hidden_size",
                            default=512,
                            type=int,
                            help="the hidden layer size of the model.")
    # arg_parser.add_argument("--rnn_size",
    #                         default=512,
    #                         type=int,
    #                         help="size of the rnn in number of hidden nodes in each layer")
    arg_parser.add_argument("--batch_size",
                            default=32,
                            type=int,
                            help="what is the utils batch size in number of images per batch?"
                                 " (there will be x seq_per_img sentences)")
    arg_parser.add_argument("--output_size",
                            default=1000,
                            type=int,
                            help="number of output answers")
    arg_parser.add_argument("--rnn_layers",
                            default=2,
                            type=int,
                            help="number of the rnn layers")
    arg_parser.add_argument("--seq_len",
                            default=26,
                            type=int,
                            help="max length of input questions")
    arg_parser.add_argument("--fact_num",
                            default=10,
                            type=int,
                            help="max num of input facts")
    arg_parser.add_argument("--fact_len",
                            default=26,
                            type=int,
                            help="max length of input facts")
    arg_parser.add_argument("--nheaders",
                            default=8,
                            type=int,
                            help="the number of attention headers")
    arg_parser.add_argument("--nanswers",
                            default=1024,
                            type=int,
                            help="the number of answers")

    arg_parser.add_argument("--w2v_type",
                            default="random",
                            help="type of word embeddings, default: random")
    arg_parser.add_argument("--i2v_type",
                            default="linear",
                            help="type of image embeddings, default: linear")

    # # 图片
    # arg_parser.add_argument("--input_img_train_h5",
    #                         default='data/cocoqa_data_img_vgg_train.h5',
    #                         help="path to the h5file containing the image feature")
    # arg_parser.add_argument("--input_img_test_h5",
    #                         default='data/cocoqa_data_img_vgg_test.h5',
    #                         help="path to the h5file containing the image feature")
    # arg_parser.add_argument("--input_img_val_h5",
    #                         default='data/cocoqa_data_img_vgg_val.h5',
    #                         help="path to the h5file containing the image feature")

    # # 样例
    # arg_parser.add_argument("--input_prepro_train_h5",
    #                         default='data/cocoqa_prepro_train.h5',
    #                         help="path to the h5file containing the preprocessed dataset")
    # arg_parser.add_argument("--input_prepro_test_h5",
    #                         default='data/cocoqa_prepro_test.h5',
    #                         help="path to the h5file containing the preprocessed dataset")
    # arg_parser.add_argument("--input_prepro_val_h5",
    #                         default='data/cocoqa_prepro_val.h5',
    #                         help="path to the h5file containing the preprocessed dataset")

    # # 处理信息
    # arg_parser.add_argument("--input_json",
    #                         default='data/cocoqa_data_prepro.json',
    #                         help="path to the json file containing additional info and vocab")
    #
    # arg_parser.add_argument("--start_from",
    #                         default='',
    #                         help="path to a model checkpoint to initialize model weights from. Empty = don't")
    # arg_parser.add_argument("--co_atten_type",
    #                         default='Alternating',
    #                         help="co_attention type. Parallel or Alternating")
    # arg_parser.add_argument("--feature_type",
    #                         default='VGG',
    #                         help="VGG or Residual")

    # 训练信息
    arg_parser.add_argument("--learning_rate",
                            default=1e-3,
                            type=float,
                            help="learning rate")
    arg_parser.add_argument("--learning_rate_decay_start",
                            default=10,
                            type=int,
                            help="at what epoch to start decaying learning rate? (-1 = dont)")
    arg_parser.add_argument("--learning_rate_decay_every",
                            default=10,
                            type=int,
                            help="every how many epoch there after to drop LR by 0.1?")
    arg_parser.add_argument("--max_epoch",
                            default=100,
                            type=int,
                            help="max number of iterations to run for (-1 = run forever)")
    arg_parser.add_argument("--save_dir",
                            default='save',
                            help="folder to save checkpoints into (empty = this folder)")
    arg_parser.add_argument("--id",
                            default=datetime.now().strftime('%y%m%d%H%M%S'),
                            help="an id identifying this run/job. "
                                 "used in cross-val and appended when writing progress files")
    arg_parser.add_argument("--seed",
                            default=123,
                            type=int,
                            help="random number generator seed to use")

    return arg_parser.parse_args().__dict__


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


def main(opts):
    opts["vocab_size"] = 2000
    opts["img_feat_dim"] = 512
    opts["img_len"] = 196
    print(json.dumps(opts, indent=2))

    model = make_model(opts)

    ques = torch.randint(low=0, high=opts["vocab_size"],
                         size=(opts["batch_size"], opts["seq_len"]))
    ques_mask = torch.zeros_like(ques, dtype=torch.bool)  # 计算问句的mask
    ques_mask[torch.eq(ques, 0)] = True
    ques_mask.unsqueeze_(1)  # 计算atten时，需扩张一个维度

    images = torch.randn(size=(opts["batch_size"], opts["img_len"], opts["hidden_size"]))

    facts = torch.randint(low=0, high=opts["vocab_size"],
                          size=(opts["batch_size"], opts["fact_num"], opts["fact_len"]))

    facts_count = torch.randint(low=0, high=opts["fact_num"], size=(opts["batch_size"],))
    fact_mask = torch.ones((opts["batch_size"], opts["fact_num"]), dtype=torch.bool)  # 生成fact的mask
    cal_fact_mask(facts_count, fact_mask)
    fact_mask.unsqueeze_(1)

    predict = model(ques, ques_mask, images, facts, fact_mask)
    print(predict.size())


if __name__ == '__main__':
    opts = args_process()

    torch.manual_seed = opts["seed"]

    if not exists(opts["save_dir"]):
        os.makedirs(opts["save_dir"])

    main(opts)
