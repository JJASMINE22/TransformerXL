# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os

# ===annotation===
model_data = "./model_data"
file_dir = "数据集根目录"

# ===data generator===
verbose = True
dataset = "数据集名称"
batch_size = 16
sequence_len = 70
warning_len = 500000

# ===model===
n_head = 10
d_head = 50
d_embed = 500
cutoffs = []
tie_projs = [True, True, True, True]
embed_size = 500
divide_rate = 1
num_layers = 6
member_len = 70

# ===training===
Epoches = 400
learning_rate = 6e-5
warmup_learning_rate = 1e-7
min_learning_rate = 1e-8
cosine_scheduler = True
per_sample_interval = 10
ckpt_path = "./checkpoint"
