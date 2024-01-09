# -*- coding:utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

import warnings
import nets

warnings.filterwarnings("ignore")

epsilon = 1e-9


class Model(object):

    def __init__(self, W_embedding, user_embedding, settings):
        self.max_length = settings.max_length
        self.item_dim = settings.item_dim
        self.gnn_layer = settings.gnn_layer
        self.dnn_size = settings.dnn_size

        self.interest_dim = settings.interest_dim
        self.hidden_size = self.interest_dim

        self.n_block = settings.n_block
        self.short_n_block = settings.short_n_block

        # self.block_size = settings.block_size
        # self.short_block_size = settings.short_block_size

        self.in_block_flag = settings.in_block_flag

        self.recent_click_num = settings.recent_click_num

        self.batch_size = settings.batch_size
        self.dataset_flag = settings.dataset_flag

        self.losspos_w = settings.losspos_w
        self.losspos_flag = settings.losspos_flag

        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(settings.learning_rate,
                                                              self.global_step,
                                                              settings.decay_steps,
                                                              settings.decay_rate,
                                                              staircase=True))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        if self.dataset_flag == "dataset1":
            self.visual_embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                                    initializer=tf.constant_initializer(W_embedding), trainable=True)
        elif self.dataset_flag == "dataset2":
            self.visual_embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                                    initializer=tf.constant_initializer(W_embedding), trainable=True)

        self.user_embedding = tf.get_variable(name='user_embedding', shape=user_embedding.shape,
                                              initializer=tf.constant_initializer(user_embedding), trainable=True)

        # placeholders
        self.tst = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('Inputs'):
            self.pos_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='pos_inputs')  # postive history
            self.pos_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='pos_mask_inputs')  # postive history sequence mask
            self.pos_block_mask = tf.placeholder(tf.int32, [self.batch_size], name='pos_block_mask')  # block mask

            self.y_inputs = tf.placeholder(tf.float32, [self.batch_size], name='y_input')  # label
            self.item_inputs = tf.placeholder(tf.int32, [self.batch_size], name='item_inputs')  # video id x_new
            self.user_id_inputs = tf.placeholder(tf.int32, [self.batch_size], name='user_id_inputs')

            self.short_pos_inputs = tf.placeholder(tf.int32, [self.batch_size, self.recent_click_num], name='short_pos_inputs')
            self.short_pos_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='short_pos_mask_inputs')
            self.short_pos_block_mask = tf.placeholder(tf.int32, [self.batch_size], name='short_pos_block_mask')

            self.item = tf.nn.embedding_lookup(self.visual_embedding, self.item_inputs)
            self.pos = tf.nn.embedding_lookup(self.visual_embedding, self.pos_inputs)
            self.short_pos = tf.nn.embedding_lookup(self.visual_embedding, self.short_pos_inputs)
            self.user_feature = tf.nn.embedding_lookup(self.user_embedding, self.user_id_inputs)

        with tf.name_scope('user_block_att'):
            print("user_block_att")
            item_block_emb = self.block_embedding(self.pos_mask_inputs, self.pos, self.max_length, self.n_block)  # # [B,10,64]  seq_len, block_len)
            keys = nets.dense(item_block_emb, self.interest_dim, ["w_l2"], keep_prob=self.keep_prob, activation=None)
            self.caps2 = self.vanilla_attention_qkv_mask(self.user_feature, keys, item_block_emb, self.pos_block_mask)  # keys [B,T,H]    queries=self.user_feature, keys, vals, keys_length
            self.caps2 = tf.expand_dims(self.caps2, 1)

        with tf.name_scope('lstm_block_last_att'):
            short_item_block_emb = self.block_embedding(self.short_pos_mask_inputs, self.short_pos, self.recent_click_num, self.short_n_block)  # # [B,10,64]
            self.short_caps2 = self.recent_dnn(short_item_block_emb, self.short_pos_block_mask)
            self.short_caps2 = tf.expand_dims(self.short_caps2, 1)  # [B,1,64]

        with tf.name_scope('normalize_ortho_user_fuse_atten_1_1_diver'):
            self.caps2 = nets.normalize(self.caps2, scope="caps2")
            self.short_caps2 = nets.normalize(self.short_caps2, scope="short_caps2")
            diver_gate = self.get_ortho_user_interest_attention(self.short_caps2, self.caps2)
            self.joint_vector = self.caps2 + self.short_caps2 - tf.multiply(diver_gate, self.short_caps2)  # [B,5,64]
            self.joint_output = self.predict_score(self.joint_vector)

        self.joint_evaoutput = tf.nn.sigmoid(self.joint_output)

        with tf.name_scope('joint_loss'):
            l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name])
            if not self.losspos_flag:
                self.joint_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.joint_output, labels=self.y_inputs)) + l2_norm * 0.00005
            if self.losspos_flag:
                self.joint_loss = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(logits=self.joint_output, targets=self.y_inputs, pos_weight=self.losspos_w)) + l2_norm * 0.00005

        grads_and_vars = self.optimizer.compute_gradients(self.joint_loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

    def get_ortho_user_interest_attention(self, short_interest, long_interest):  # [B,5,64]
        vec_squared_norm = tf.reduce_sum(tf.square(short_interest), -1, keepdims=True)  # [B,1]    [B,5,1]
        vec_project1 = long_interest * short_interest * short_interest / (vec_squared_norm + epsilon)  # [B,64]    [B,5,64]
        vec_ortho1 = long_interest - vec_project1  # [B,64]      [B,5,64]
        long_interest2 = tf.expand_dims(self.user_feature, 1)
        vec_project2 = long_interest2 * short_interest * short_interest / (vec_squared_norm + epsilon)  # [B,64]    [B,5,64]
        vec_ortho2 = long_interest2 - vec_project2  # [B,64]      [B,5,64]
        vec_ortho = tf.concat([vec_ortho1, vec_ortho2], axis=-1)
        concat_att1 = nets.dense(vec_ortho, self.interest_dim, ["w_con1", "b_con1"], keep_prob=self.keep_prob, activation=tf.nn.relu)
        diver_gate = nets.dense(concat_att1, 1, ["w_con2", "b_con2"], keep_prob=self.keep_prob, activation=tf.nn.sigmoid)

        return diver_gate

    def recent_dnn(self, inputs, inputs_mask):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        rnn_outputs, states1 = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                 inputs=inputs,
                                                 sequence_length=inputs_mask,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32)
        last_h = states1.h  # [B,hidden]

        rnn_outputs = tf.nn.dropout(rnn_outputs, self.keep_prob)
        rnn_outputs = self.vanilla_attention_mask(last_h, rnn_outputs, inputs_mask)
        return rnn_outputs

    def predict_score(self, interest_emb):  # input: [B,5,64]       newinput: [B,64]
        inter_shape = interest_emb.get_shape().as_list()
        if len(inter_shape) == 3 and inter_shape[1] == 1:
            interest_emb = tf.squeeze(interest_emb, axis=1)
        elif len(inter_shape) == 3 and inter_shape[1] != 1:
            interest_emb = self.vanilla_attention(self.item, interest_emb)

        w1 = self.weight_variable([self.item_dim + self.interest_dim, self.dnn_size], name='w1')  # [128,128]
        b1 = self.bias_variable([self.dnn_size], name='w_b1')

        output = tf.matmul(tf.concat([interest_emb, self.item], axis=1), w1) + b1  # [B,128]

        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.nn.relu(output)

        w2 = self.weight_variable([self.dnn_size, 1], name='w2')
        b2 = self.bias_variable([1], name='w_b2')
        output = tf.matmul(output, w2) + b2
        return tf.reshape(output, [-1])

    def block_embedding(self, mask_inputs, inputs, seq_len, block_len):
        # reduce_sum
        if block_len == 1:
            return inputs
        if self.in_block_flag == 'cumsum' or self.in_block_flag == 'maxpool':
            pos_key_masks = tf.sequence_mask(mask_inputs, seq_len)  # [B, self.timestep]
            inputs = inputs * tf.cast(tf.expand_dims(pos_key_masks, -1), tf.float32)
            item_block_emb = tf.concat(tf.split(tf.expand_dims(inputs, axis=1), block_len, axis=2), axis=1)  # [B,15,20,64]
            if self.in_block_flag == 'cumsum':
                item_block_emb = tf.reduce_sum(item_block_emb, axis=2)  # [B,15,64]
            elif self.in_block_flag == 'maxpool':
                item_block_emb = tf.reduce_max(item_block_emb, axis=2)  # [B,15,64]
            return item_block_emb


        elif self.in_block_flag == "average":
            pos_key_masks = tf.sequence_mask(mask_inputs, seq_len)  # [B, self.timestep]
            inputs = inputs * tf.cast(tf.expand_dims(pos_key_masks, -1), tf.float32)
            item_block_emb = tf.concat(tf.split(tf.expand_dims(inputs, axis=1), block_len, axis=2), axis=1)  # [B,15,20,64]

            pos_key_masks = tf.concat(tf.split(tf.expand_dims(pos_key_masks, axis=1), block_len, axis=2), axis=1)  # [B,15,20]
            pos_key_masks = tf.cast(pos_key_masks, tf.float32)
            pos_key_masks = tf.reduce_sum(pos_key_masks, axis=2, keep_dims=True)
            item_block_emb = tf.reduce_sum(item_block_emb, axis=2)  # [B,15,64]
            item_block_emb = item_block_emb / (pos_key_masks + epsilon)
            return item_block_emb

    def vanilla_attention(self, queries, keys):  # query [B,64],  keys [B,5,64]
        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.interest_dim])

    def vanilla_attention_mask(self, queries, keys, keys_length):  # keys [B,T,H]
        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.item_dim])

    def vanilla_attention_qkv_mask(self, queries, keys, vals, keys_length):  # keys [B,T,H]
        queries = tf.expand_dims(queries, 1)  # [B, 1, H]
        # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, vals)  # [B, 1, H]
        return tf.reshape(outputs, [-1, self.item_dim])

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def para_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.0001, shape=shape)
        return tf.Variable(initial, name=name)
