from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random, logging, os
import pickle
from tqdm import tqdm

# 数据2的category分析

item_cate_dic = dict()

def get_item_cate_dict(interacte_data):
    for vals in interacte_data:
        interacte = vals[0]
        item, cate = interacte[1], interacte[2]
        if item not in item_cate_dic:
            item_cate_dic[item] = cate


def generate_train_user_click(train_interaction_data, user_click_ids):
    # # user,item,category,click,time
    for item in tqdm(train_interaction_data):
        if item[0][3] != 1:
            continue
        user_click_ids[item[0][0]].append((item[0][1], item[0][-1]))   #user, [item, time]


def read_file(file_name):
    file = open(file_name, "r")
    pred_results = [[] for _ in range(10986)]
    for line in file.readlines():
        line = line.strip()
        userid, itemlist = line.split("@")
        userid = int(userid)
        items = itemlist.split("\t")
        items = [float(item) for item in items]
        items = [int(item) for item in items]
        # print(items)
        pred_results[userid] = items
    file.close()
    # print("read result_file done!!")
    return pred_results


def joint_items(alist, blist):
    ajointlist = []
    bjointlist = []
    for item in alist:
        if item in blist:
            ajointlist.append(item)

    for item in blist:
        if item in alist:
            bjointlist.append(item)
    return len(ajointlist), len(bjointlist)



def load_data2(pred_file_name):

    pred_results = read_file(pred_file_name)


    duplicate2_list = []
    duplicate3_list = []

    for userid in range(10986):
        user_his_cates = []
        user_fut_cates = []
        user_real_fut_cates = []
        his_len = len(sorted_user_click_ids[userid])
        his_fut = len(pred_results[userid])
        min_len = min(his_fut, his_len)
        if min_len < 1:
            continue
        for item, time in sorted_user_click_ids[userid][-min_len:]:
            user_his_cates.append(item_cate_dic[item])
        for item in pred_results[userid][:min_len]:
            user_fut_cates.append(item_cate_dic[item])
        for item, time in val_sorted_user_click_ids[userid][:min_len]:
            user_real_fut_cates.append(item_cate_dic[item])
        his_cnt, fut_cnt = joint_items(user_his_cates, user_fut_cates)
        duplicate2_list.append(fut_cnt/min_len)
        his_cnt, fut_cnt = joint_items(user_his_cates, user_real_fut_cates)
        duplicate3_list.append(fut_cnt/min_len)
    duplicate_rat2 = np.average(duplicate2_list)
    diversity_rat2 = 1 - duplicate_rat2
    print("预测的：", diversity_rat2)

    duplicate_rat3 = np.average(duplicate3_list)
    diversity_rat3 = 1 - duplicate_rat3
    print("实际的：", diversity_rat3)


def load_train_test():
    file_root = "/data/gp/Dataset1/dataset2"

    pkl_dir = os.path.join(file_root, "new_train_dataset_0623_100_1.pkl")
    with open(pkl_dir, 'rb') as f:
        train_interaction_data = pickle.load(f)
        val_interaction_data = pickle.load(f)
    get_item_cate_dict(train_interaction_data)
    get_item_cate_dict(val_interaction_data)
    print("item_cate_dic done!!")
    print("item个数一共：", len(item_cate_dic))

    user_click_ids = [[] for _ in range(10986)]
    generate_train_user_click(train_interaction_data, user_click_ids)
    sorted_user_click_ids = [sorted(item, key=lambda x: x[-1]) for item in user_click_ids]      #user整个历史序列

    val_user_click_ids = [[] for _ in range(10986)]
    generate_train_user_click(val_interaction_data, val_user_click_ids)
    val_sorted_user_click_ids = [sorted(item, key=lambda x: x[-1]) for item in val_user_click_ids]  # user整个历史序列
    return sorted_user_click_ids,val_sorted_user_click_ids


if __name__ == '__main__':
    sorted_user_click_ids,val_sorted_user_click_ids = load_train_test()

    names = ["result_topk40_time2022-08-05 16:49",\
             "result_topk40_time2022-08-05 06:47",\
             "result_topk40_time2022-08-05 04:06",\
             "result_topk40_time2022-08-05 01:23",\
             "result_topk50_time2022-08-05 16:49",\
             "result_topk50_time2022-08-05 06:47",\
             "result_topk50_time2022-08-05 04:07",\
             "result_topk50_time2022-08-05 01:23",\
             'result_topk10_time2022-08-05 01:22',\
             'result_topk20_time2022-08-05 01:22',\
             'result_topk30_time2022-08-05 01:23',\
             'result_topk40_time2022-08-05 01:23',\
             'result_topk50_time2022-08-05 01:23',\
             'result_topk10_time2022-08-05 04:06',\
             'result_topk20_time2022-08-05 04:06',\
             'result_topk30_time2022-08-05 04:06',\
             'result_topk40_time2022-08-05 04:06',\
             'result_topk50_time2022-08-05 04:07',\
             'result_topk10_time2022-08-05 16:48',\
             'result_topk20_time2022-08-05 16:48',\
             'result_topk30_time2022-08-05 16:48',\
             'result_topk40_time2022-08-05 16:49',\
             'result_topk50_time2022-08-05 16:49',\
             ]
    # pred_file_name = "/home/gp/python_projects/VideoRec20220426_ortho_0802/results/result_topk10_time2022-08-04 21:54"
    for name in names:
        pred_file_name = "/home/gp/python_projects/VideoRec20220426_ortho_0802/results/" + name
        print(name)
        load_data2(pred_file_name)



# l
# -rw-rw-r-- 1 gp gp 1142433 8月   5 01:22 'result_topk10_time2022-08-05 01:22'
# -rw-rw-r-- 1 gp gp 2201474 8月   5 01:23 'result_topk20_time2022-08-05 01:22'
# -rw-rw-r-- 1 gp gp 3225427 8月   5 01:23 'result_topk30_time2022-08-05 01:23'
# -rw-rw-r-- 1 gp gp 4213676 8月   5 01:23 'result_topk40_time2022-08-05 01:23'
# -rw-rw-r-- 1 gp gp 5171033 8月   5 01:23 'result_topk50_time2022-08-05 01:23'
# -rw-rw-r-- 1 gp gp 6097503 8月   5 01:23 'result_topk60_time2022-08-05 01:23'
# s
# -rw-rw-r-- 1 gp gp 1142457 8月   5 04:06 'result_topk10_time2022-08-05 04:06'
# -rw-rw-r-- 1 gp gp 2201452 8月   5 04:06 'result_topk20_time2022-08-05 04:06'
# -rw-rw-r-- 1 gp gp 3225453 8月   5 04:06 'result_topk30_time2022-08-05 04:06'
# -rw-rw-r-- 1 gp gp 4213754 8月   5 04:07 'result_topk40_time2022-08-05 04:06'
# -rw-rw-r-- 1 gp gp 5171006 8月   5 04:07 'result_topk50_time2022-08-05 04:07'
# -rw-rw-r-- 1 gp gp 6097463 8月   5 04:07 'result_topk60_time2022-08-05 04:07'
# ls
# -rw-rw-r-- 1 gp gp 1142293 8月   5 06:47 'result_topk10_time2022-08-05 06:47'
# -rw-rw-r-- 1 gp gp 2201234 8月   5 06:47 'result_topk20_time2022-08-05 06:47'
# -rw-rw-r-- 1 gp gp 3225201 8月   5 06:47 'result_topk30_time2022-08-05 06:47'
# -rw-rw-r-- 1 gp gp 4213497 8月   5 06:47 'result_topk40_time2022-08-05 06:47'
# -rw-rw-r-- 1 gp gp 5170791 8月   5 06:47 'result_topk50_time2022-08-05 06:47'
# -rw-rw-r-- 1 gp gp 6097220 8月   5 06:48 'result_topk60_time2022-08-05 06:47'
# td
# -rw-rw-r-- 1 gp gp 1142419 8月   5 16:48 'result_topk10_time2022-08-05 16:48'
# -rw-rw-r-- 1 gp gp 2201402 8月   5 16:48 'result_topk20_time2022-08-05 16:48'
# -rw-rw-r-- 1 gp gp 3225363 8月   5 16:49 'result_topk30_time2022-08-05 16:48'
# -rw-rw-r-- 1 gp gp 4213738 8月   5 16:49 'result_topk40_time2022-08-05 16:49'
# -rw-rw-r-- 1 gp gp 5171041 8月   5 16:49 'result_topk50_time2022-08-05 16:49'
