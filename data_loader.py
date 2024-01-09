from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random, logging, os
import pickle
from tqdm import tqdm


class DataLoader(object):
    def __init__(self, params):
        self.data_dir = params.data_dir
        self.batch_size = params.batch_size
        self.max_length = params.max_length
        self.block_size = params.block_size
        self.short_block_size = params.short_block_size

        self.n_block = params.n_block
        self.filter_thre = params.filter_thre

        self.recent_click_num = params.recent_click_num  # 暂时设定为block_size*2
        self.short_n_block = params.short_n_block  # 2

        self.dataset_flag = params.dataset_flag
        if self.dataset_flag == 'dataset1':
            self.preload_feat_into_memory()
        elif self.dataset_flag == 'dataset2':
            self.load_data2()

    def load_data2(self):
        file_root = "/data/gp/Dataset1/dataset2"
        # 下面是视觉特征
        train_visual_dir = os.path.join(file_root, "visual_feature_0711_notwhiten.npy")
        self.train_visual_feature = np.load(train_visual_dir)
        self.train_visual_feature = np.concatenate([self.train_visual_feature, [[0.0] * 64]], axis=0)
        print(self.train_visual_feature.shape)

        self.user_em = np.random.normal(loc=0, scale=0.01, size=[10986, 64])

        file_name = "./new_train_dataset_0623_100_negall.pkl"
        pkl_dir = os.path.join(file_root, file_name)

        with open(pkl_dir, 'rb') as f:
            self.train_interaction_data = pickle.load(f)
            self.val_interaction_data = pickle.load(f)
            self.pos_his_data = pickle.load(f)
            self.pos_his_data300 = pickle.load(f)

        self.epoch_train_length = len(self.train_interaction_data)
        self.epoch_test_length = len(self.val_interaction_data)
        self.user_num = len(self.pos_his_data)
        self.train_item_num = self.train_visual_feature.shape[0]

        logging.info(train_visual_dir)
        logging.info('{} train samples'.format(self.epoch_train_length))
        logging.info('{} test samples'.format(self.epoch_test_length))
        logging.info('{} user_num'.format(self.user_num))
        logging.info('{} item_num'.format(self.train_item_num))

    def preload_feat_into_memory(self):
        # 下面是视觉特征
        logging.info('load train visual feature')
        train_visual = os.path.join(self.data_dir, "visual64_select.npy")
        self.train_visual_feature = np.load(train_visual)
        self.train_item_num = self.train_visual_feature.shape[0]
        logging.info("train_visual_feature : {}".format(self.train_visual_feature.shape))

        self.user_em = np.random.normal(loc=0, scale=0.01, size=[10000, 64])

        file_name = "./new_train_dataset1_thr{}_negall.pkl".format(self.filter_thre)
        pkl_dir = os.path.join(self.data_dir, file_name)

        with open(pkl_dir, 'rb') as f:
            self.train_interaction_data = pickle.load(f)
            self.val_interaction_data = pickle.load(f)
            self.pos_his_data = pickle.load(f)  # 用户全部的行为序列
            self.pos_his_data300 = pickle.load(f)  # 用户sample出的300个行为序列

            random.shuffle(self.train_interaction_data)  # 0815 add
            self.epoch_train_length = len(self.train_interaction_data)
            self.epoch_test_length = len(self.val_interaction_data)
            self.user_num = len(self.pos_his_data)

            logging.info('new_train_dataset_pos_0617.pkl')
            logging.info('{} train samples'.format(self.epoch_train_length))
            logging.info('{} test samples'.format(self.epoch_test_length))
            logging.info('{} user_num'.format(self.user_num))

    def getBatchDataset2(self, num, train_test_flag):
        if train_test_flag == 'train':
            self.epoch_data = self.train_interaction_data
        elif train_test_flag == 'test':
            self.epoch_data = self.val_interaction_data

        label = np.zeros([self.batch_size], dtype=np.int)
        item_input = np.zeros([self.batch_size], dtype=np.int)
        user_id_input = np.zeros([self.batch_size], dtype=np.int)

        # long-term
        pos_input = np.zeros([self.batch_size, self.max_length], dtype=np.int)
        pos_mask_input = np.zeros([self.batch_size], dtype=np.int)
        pos_inter_mask = np.zeros([self.batch_size], dtype=np.int)

        # short-term
        short_pos_input = np.zeros([self.batch_size, self.recent_click_num], dtype=np.int)
        short_pos_mask_input = np.zeros([self.batch_size], dtype=np.int)
        short_pos_inter_mask = np.zeros([self.batch_size], dtype=np.int)

        for i in range(num * self.batch_size, (num + 1) * self.batch_size):

            interaction = self.epoch_data[i][0]
            user_id = interaction[0]
            if train_test_flag == 'train':
                pos_len_alpine = self.epoch_data[i][1][0]
            elif train_test_flag == 'test':
                pos_len_alpine = self.epoch_data[i][1][0]

            pos_len = self.epoch_data[i][-1][1]

            user_id_input[i % self.batch_size] = user_id
            label[i % self.batch_size] = interaction[3]
            item_input[i % self.batch_size] = interaction[1]

            # long-term
            pos_pad = (max(self.max_length, pos_len_alpine) - pos_len_alpine)
            pos_input[i % self.batch_size] = self.pos_his_data300[user_id][:pos_len_alpine] + [self.train_item_num - 1] * pos_pad
            pos_mask_input[i % self.batch_size] = pos_len_alpine
            pos_pad_n_block = pos_pad // self.block_size
            pos_inter_mask[i % self.batch_size] = self.n_block - pos_pad_n_block

            # short-term,取最近self.recent_click_num个行为
            # ------------------------shortterm取自pos_his_data中-----------------------------
            short_pos_start = pos_len - self.recent_click_num if pos_len > self.recent_click_num else 0
            short_pos_pad = self.recent_click_num - (pos_len - short_pos_start)
            short_pos_input[i % self.batch_size] = self.pos_his_data[user_id][short_pos_start:pos_len] + [self.train_item_num - 1] * short_pos_pad

            short_pos_mask_input[i % self.batch_size] = pos_len - short_pos_start
            short_pos_pad_n_block = short_pos_pad // self.short_block_size  # 向下取整，11//3 = 3
            short_pos_inter_mask[i % self.batch_size] = self.short_n_block - short_pos_pad_n_block  # block序列的mask，值小于等于self.n_block
        return user_id_input, label, pos_input, pos_mask_input, item_input, pos_inter_mask, short_pos_input, short_pos_mask_input, short_pos_inter_mask

    def getBatch(self, num, train_test_flag):
        if train_test_flag == 'train':
            self.epoch_data = self.train_interaction_data
        elif train_test_flag == 'test':
            self.epoch_data = self.val_interaction_data

        user_ids = np.zeros([self.batch_size], dtype=np.int)
        label = np.zeros([self.batch_size], dtype=np.int)
        item_input = np.zeros([self.batch_size], dtype=np.int)

        # long-term
        pos_input = np.zeros([self.batch_size, self.max_length], dtype=np.int)
        pos_mask_input = np.zeros([self.batch_size], dtype=np.int)  # 整个序列的mask，值小于等于max_length
        pos_inter_mask = np.zeros([self.batch_size], dtype=np.int)
        # short-term
        short_pos_input = np.zeros([self.batch_size, self.recent_click_num], dtype=np.int)
        short_pos_mask_input = np.zeros([self.batch_size], dtype=np.int)
        short_pos_inter_mask = np.zeros([self.batch_size], dtype=np.int)

        for i in range(num * self.batch_size, (num + 1) * self.batch_size):
            # (user_id, photo_id, click, like, follow, time, playing_time, duration_time),click_end_idx
            user_id = self.epoch_data[i][0][0]
            pos_len_alpine = self.epoch_data[i][1][0]  # 原先的end_idx

            # short_pos_start = self.epoch_data[i][2][0]   #按照recent_num = 100过滤的
            pos_len = self.epoch_data[i][2][1]  # [short_start_idx,click_end_idx, unclick_start_idx,notclick_end_idx]

            user_ids[i % self.batch_size] = user_id
            item_input[i % self.batch_size] = self.epoch_data[i][0][1]
            label[i % self.batch_size] = self.epoch_data[i][0][2]

            # long-term
            # print("self.pos_his_data300[user_id]", self.pos_his_data300[user_id])
            # print("len(self.pos_his_data300[user_id])",len(self.pos_his_data300[user_id]))
            # print("pos_len_alpine",pos_len_alpine)
            pos_pad = (max(self.max_length, pos_len_alpine) - pos_len_alpine)
            pos_input[i % self.batch_size] = self.pos_his_data300[user_id][:pos_len_alpine] + [self.train_item_num - 1] * pos_pad
            pos_mask_input[i % self.batch_size] = pos_len_alpine
            pos_pad_n_block = pos_pad // self.block_size
            pos_inter_mask[i % self.batch_size] = self.n_block - pos_pad_n_block

            # short-term,取最近self.recent_click_num个行为
            # ------------------------shortterm取自pos_his_data中-----------------------------
            short_pos_start = pos_len - self.recent_click_num if pos_len > self.recent_click_num else 0
            short_pos_pad = self.recent_click_num - (pos_len - short_pos_start)
            short_pos_input[i % self.batch_size] = self.pos_his_data[user_id][short_pos_start:pos_len] + [self.train_item_num - 1] * short_pos_pad
            short_pos_mask_input[i % self.batch_size] = pos_len - short_pos_start

            short_pos_pad_n_block = short_pos_pad // self.short_block_size  # 向下取整，11//3 = 3
            short_pos_inter_mask[i % self.batch_size] = self.short_n_block - short_pos_pad_n_block  # block序列的mask，值小于等于self.n_block

        return user_ids, label, pos_input, pos_mask_input, item_input, pos_inter_mask, short_pos_input, short_pos_mask_input, short_pos_inter_mask
