import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
import argparse
import json
from sklearn.model_selection import StratifiedKFold
import os


def truncate_sequences(maxlen, index, *sequences):
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


def load_data(filename):
    """
    加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    data = open(filename, encoding='UTF-8')
    D = []
    for text in data.readlines():
        text_list = text.strip().split('\t')
        text1 = text_list[0]
        text2 = text_list[1]
        if len(text_list) > 2:
            label = int(text_list[2])
        else:
            label = -100
        D.append((text1, text2, label))
    return D


# 统计序列长度大于某个值的数量
def get_len(data):
    lens = []
    for line in data:
        lens.append(len(line[0] + line[1]))
    count = 0
    for i in lens:
        if i > 64:
            count += 1
    print(count)


if __name__ == '__main__':
    a = load_data('./data/train_dataset/BQ/train')
    b = load_data('./data/train_dataset/BQ/test')
    c = load_data('./data/train_dataset/BQ/dev')
    d = load_data('./data/train_dataset/LCQMC/train')
    e = load_data('./data/train_dataset/LCQMC/test')
    f = load_data('./data/train_dataset/LCQMC/dev')
    g = load_data('./data/train_dataset/OPPO/train')
    h = load_data('./data/train_dataset/OPPO/dev')
    i = load_data('./data/test_A.tsv')
    get_len(i)
