from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import tqdm
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import evaluation2_item
from utils import evaluation2


class Solver(object):
    def __init__(self, model, data, params):
        self.model = model
        self.data = data

        self.restore = params.restore
        self.max_epoch = params.max_epoch
        self.batch_size = params.batch_size
        self.model_name = params.model_name
        self.display = params.display
        self.restore_model_dir = params.restore_model_dir
        self.lr = params.learning_rate
        self.keep_prob = params.keep_prob
        self.out_dir = params.out_dir
        # self.summary_dir = os.path.join(self.out_dir, 'summary')
        self.model_dir = os.path.join(self.out_dir, 'ckpt')
        self.dataset_flag = params.dataset_flag

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # if not os.path.exists(self.summary_dir):
        #     os.makedirs(self.summary_dir)

    def train(self):

        # config = tf.ConfigProto(log_device_placement=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        if self.restore:
            has_ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if has_ckpt and has_ckpt.model_checkpoint_path:
                model_path = has_ckpt.model_checkpoint_path
                saver.restore(sess, model_path)
                logging.info("Load model from {}".format(model_path))
            else:
                raise ValueError("No checkpoint file found in {}".format(self.model_dir))

        # saver.restore(sess,  "./model_save/model.ckpt")
        best_auceva, stop_training_counter = 0, 0

        logging.info('start train phase')

        loop_num = int(self.data.epoch_train_length / self.batch_size)
        test_frequency = (int(loop_num * 0.1) // 100) * 100
        test_start_step = (int(loop_num * 0.7) // 1000) * 1000
        logging.info("test_frequency:" + str(test_frequency))
        logging.info("test_start_step:" + str(test_start_step))

        all_num = 0
        for epoch in range(self.max_epoch):
            logging.info('all train iterations: {}, this epoch: {}'.format(loop_num, epoch))

            avg_loss, avg_auc = 0.0, 0.0
            for num in range(loop_num):
                all_num += 1
                if self.dataset_flag == "dataset1":
                    ret_vals = self.data.getBatch(num, 'train')
                elif self.dataset_flag == "dataset2":
                    ret_vals = self.data.getBatchDataset2(num, 'train')

                user_ids, label, pos_input, pos_mask_input, item_input, pos_inter_mask, short_pos_input, short_pos_mask_input, short_pos_inter_mask = ret_vals
                _, loss, y_pred = sess.run([self.model.opt_op, self.model.joint_loss, self.model.joint_evaoutput],
                                           feed_dict={
                                               self.model.user_id_inputs: user_ids,
                                               self.model.item_inputs: item_input,
                                               self.model.y_inputs: label,

                                               self.model.pos_inputs: pos_input,
                                               self.model.pos_mask_inputs: pos_mask_input,
                                               self.model.pos_block_mask: pos_inter_mask,

                                               self.model.short_pos_inputs: short_pos_input,
                                               self.model.short_pos_mask_inputs: short_pos_mask_input,
                                               self.model.short_pos_block_mask: short_pos_inter_mask,

                                               self.model.keep_prob: self.keep_prob,
                                               self.model.tst: False
                                           })
                if len(np.unique(label)) > 1:
                    auc = roc_auc_score(label, y_pred)
                    avg_auc += auc
                avg_loss += loss

                if (num + 1) % self.display == 0:
                    logging.info("epoch:" + str(epoch) + ' loop:' + str(num) + "/" + str(loop_num) + ' loss:' + str(avg_loss / self.display) + ' auc:' + str(avg_auc / self.display))
                    avg_loss, avg_auc = 0.0, 0.0

                if all_num == 101 or (all_num >= test_start_step and (all_num - test_start_step) % test_frequency == 0):
                    logging.info('all_num:' + str(all_num))
                    auc_eva = self.joint_eva(sess)
                    if (auc_eva > best_auceva):
                        # save model ----先暂时注释掉，太占用空间
                        # save_path = os.path.join(self.model_dir, 'model-{:.4f}.ckpt'.format(auc_eva))
                        # saver.save(sess, save_path, global_step=epoch)
                        # logging.info("save model into {}".format(self.model_dir))
                        # save model ----先暂时注释掉，太占用空间
                        best_auceva = auc_eva
                    else:
                        stop_training_counter += 1
                    logging.info('epoch:' + str(epoch) + ' auc on val:' + str(auc_eva) + ' best_auceva:' + str(best_auceva) + "\n")
                    logging.info('stop_training_counter' + str(stop_training_counter))

                    if stop_training_counter > 1:
                        logging.info("stop_training_counter = {}, stop trainning".format(stop_training_counter))
                        break

            # validation
            if stop_training_counter > 1:
                logging.info("stop_training_counter = {}, stop trainning".format(stop_training_counter))
                break

            auc_eva = self.joint_eva(sess)
            if (auc_eva > best_auceva):
                # saver.save(sess, self.model_dir)
                # save model ----先暂时注释掉，太占用空间
                # save_path = os.path.join(self.model_dir, 'model-{:.4f}.ckpt'.format(auc_eva))
                # saver.save(sess, save_path, global_step=epoch)
                # logging.info("save model into {}".format(self.model_dir))
                # save model ----先暂时注释掉，太占用空间
                best_auceva = auc_eva
            else:
                stop_training_counter += 1

            logging.info('epoch:' + str(epoch) + ' auc on val:' + str(auc_eva) + ' best_auceva:' + str(best_auceva) + "\n")
            logging.info('stop_training_counter' + str(stop_training_counter))

            if stop_training_counter > 1:
                logging.info("stop_training_counter = {}, stop trainning".format(stop_training_counter))
                break
        sess.close()

    def test(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        has_ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if has_ckpt and has_ckpt.model_checkpoint_path:
            model_path = has_ckpt.model_checkpoint_path
            saver.restore(sess, model_path)
            logging.info("Load model from {}".format(model_path))
        else:
            raise ValueError("No checkpoint file found in {}".format(self.model_dir))

        auc_eva = self.joint_eva(sess)
        logging.info(' auc on test:' + str(auc_eva))

        sess.close()

    def joint_eva(self, sess):
        val_loop = int(self.data.epoch_test_length / self.batch_size)
        result = []
        result_ans = []

        logging.info('"start evaluation, all evaluation iterations: {}'.format(val_loop))
        pred_dict = {}
        for valNum in range(val_loop):
            if self.dataset_flag == "dataset1":
                ret_vals = self.data.getBatch(valNum, 'test')
            elif self.dataset_flag == "dataset2":
                ret_vals = self.data.getBatchDataset2(valNum, 'test')

            user_ids, label, pos_input, pos_mask_input, item_input, pos_inter_mask, short_pos_input, short_pos_mask_input, short_pos_inter_mask = ret_vals

            loss, y_predVal = \
                sess.run([self.model.joint_loss, self.model.joint_evaoutput], feed_dict={
                    self.model.pos_inputs: pos_input,
                    self.model.pos_mask_inputs: pos_mask_input,
                    self.model.item_inputs: item_input,
                    self.model.y_inputs: label,
                    self.model.user_id_inputs: user_ids,
                    self.model.pos_block_mask: pos_inter_mask,

                    self.model.short_pos_inputs: short_pos_input,
                    self.model.short_pos_mask_inputs: short_pos_mask_input,
                    self.model.short_pos_block_mask: short_pos_inter_mask,

                    self.model.tst: True,
                    self.model.keep_prob: 1.0})  # test中不进行dropout
            result.extend(y_predVal)
            result_ans.extend(label)
            for i in range(self.batch_size):
                if pred_dict.get(user_ids[i]) is None:
                    pred_dict[user_ids[i]] = []
                pred_dict[user_ids[i]].append([y_predVal[i], int(label[i]), int(item_input[i])])

            if (valNum + 1) % (7 * self.display) == 0:
                logging.info("all test loop：{}, this loop: {}".format(val_loop, valNum))
                logging.info("======y_predVal======")
                logging.info("   ".join([str(i) for i in y_predVal[:20]]))

        auc = roc_auc_score(result_ans, result)
        for top_k in [10, 20, 30, 40, 50, 60]:
            if self.dataset_flag == "dataset1":
                precision, recall, f1 = evaluation2(pred_dict, top_k)
            elif self.dataset_flag == "dataset2":
                precision, recall, f1 = evaluation2_item(pred_dict, self.model_name, top_k)
            logging.info('test auc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f} in top {}'.format(auc, precision, recall, f1, top_k))

        return auc
