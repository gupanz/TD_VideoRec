import os
import scipy
import numpy as np
import tensorflow as tf
import time

# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return (shape)


def evaluation2(pred_dict,top_k=0):  # pred_dict[user] = y_pred, y_label
    precisions, recalls = [], []
    pos_cnt = 0

    for key in pred_dict:
        preds = pred_dict[key]
        preds.sort(key=lambda x: x[0], reverse=True)
        preds = np.array(preds)

        pos_num = sum(preds[:, 1])
        if pos_num == 0:
            pos_cnt += 1
            continue
        # precision and recall

        precisions.append([sum(preds[:top_k, 1]) / min(top_k, len(preds)), len(preds)])
        recalls.append([sum(preds[:top_k, 1]) / sum(preds[:, 1]), len(preds)])

    print("过滤掉的用户数量：pos_cnt: {}".format(pos_cnt))
    precisions = np.array(precisions)
    recalls = np.array(recalls)


    p_unweight = np.mean(precisions[:, 0])
    r_unweight = np.mean(recalls[:, 0])
    f_unweight = 2 * p_unweight * r_unweight / (p_unweight + r_unweight)


    return p_unweight, r_unweight, f_unweight



def evaluation2_item(pred_dict,model_name,top_k=0):  # pred_dict[user] = y_pred, y_label, item
    precisions, recalls = [], []
    pos_cnt = 0
    result_name = "../results/result_{}_topk{}_time{}".format(model_name,top_k, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    file = open(result_name, 'w')

    for key in pred_dict:
        preds = pred_dict[key]
        preds.sort(key=lambda x: x[0], reverse=True)
        preds = np.array(preds)

        pos_num = sum(preds[:, 1])
        neg_num = len(preds) - pos_num
        if pos_num == 0:
            pos_cnt += 1
            continue
        # precision and recall

        file.write(str(key)+"@")
        pred_items = "\t".join(map(str,preds[:top_k,2]))
        file.write(pred_items+"\n")

        precisions.append([sum(preds[:top_k, 1]) / min(top_k, len(preds)), len(preds)])
        recalls.append([sum(preds[:top_k, 1]) / sum(preds[:, 1]), len(preds)])


    print("过滤掉的用户数量：pos_cnt: {}".format(pos_cnt))
    precisions = np.array(precisions)
    recalls = np.array(recalls)


    p_unweight = np.mean(precisions[:, 0])
    r_unweight = np.mean(recalls[:, 0])
    f_unweight = 2 * p_unweight * r_unweight / (p_unweight + r_unweight)

    file.close()
    return p_unweight, r_unweight, f_unweight

