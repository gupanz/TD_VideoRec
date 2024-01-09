from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# 下面是模型的超参
class Settings(object):
    def __init__(self):
        self.phase = 'train'
        self.model_name = "td"

        self.batch_size = 1024    #self-attention 64   lstm 2048   128 256 512
        self.max_epoch = 2
        self.display = 200  # print info during training,test

        self.max_length = 300    #lstm 300
        self.block_size = 1
        # self.n_block = self.max_length//self.block_size

        self.recent_click_num = 300 # 暂时设定为block_size*2
        self.short_block_size = 1
        # self.short_n_block = self.recent_click_num // self.n_block  # 2

        self.item_dim = 64
        self.dnn_size = 128
        self.interest_dim = 64
        self.user_dim = 64
        self.gnn_layer = 2
        self.keep_prob = 1.0    # keep_prob  0.8
        self.learning_rate = 0.001
        self.decay_steps = 1000
        self.decay_rate = 0.98

        self.losspos_w = 2.0
        self.losspos_flag = True
        self.in_block_flag = 'average'  # bi_lstm lstm cumsum self_attention maxpool   block
        self.dataset_flag = "dataset1"  # dataset1  dataset2
        self.filter_thre = 200    # 0是不设置过滤，200，400

        self.restore = False
        self.out_dir = './data/output'
        # self.data_dir = './data/input'
        self.data_dir = "./data/dataset1"
        self.restore_model_dir = './data/output'
