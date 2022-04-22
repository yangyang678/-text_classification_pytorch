# -*- encoding: utf-8 -*-

"""赛题数据预处理"""

import time
from tqdm import tqdm
import jsonlines
import json

def timer(func):
    """ time-consuming decorator
    """

    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f"function: `{func.__name__}` running time: {te - ts:.4f} secs")
        return res

    return wrapper

@timer
def comp_preprocess(src_path: str, dst_path: str) -> None:
    """处理原始的比赛数据
    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """
    # 组织数据
    data_entailment = []
    data_contradiction = []
    data = open(src_path, "r+", encoding="utf8")
    for text in data.readlines():
        text_list = text.strip().split('\t')
        sent1 = text_list[0]
        sent2 = text_list[1]
        label = text_list[2]

        if not sent1:
            continue
        if int(label) == 1:
            data_entailment.append([sent1,sent2])
        elif int(label) == 2:
            data_contradiction.append([sent1,sent2])
            # 筛选
    data_entailment = sorted(data_entailment, key=lambda x: x[0])
    data_contradiction = sorted(data_contradiction, key=lambda x: x[0])
    i = 0
    j = 0
    out_data = []
    while i < len(data_entailment):
        origin = data_entailment[i][0]
        for index in range(j,len(data_contradiction)):
            if  data_entailment[i][0] == data_contradiction[index][0]:
                out_data.append({'origin':origin,'entailment':data_entailment[i][1],'contradiction':data_contradiction[index][1]})
                j = index + 1
                break
        while i < len(data_entailment) and data_entailment[i][0] == origin:
            i += 1
    # 写文件

    with open(dst_path,'w') as f:
        for d in out_data:
            f.write(json.dumps(d) + '\n')

if __name__ == '__main__':
    dev_src, dev_dst = './data/train_dataset/OPPO/dev', './data/train_dataset/OPPO/dev.txt'
    train_src, train_dst = './data/train_dataset/OPPO/train', './data/train_dataset/OPPO/train.txt'

    comp_preprocess(train_src, train_dst)
    comp_preprocess(dev_src, dev_dst)

    dev_src, dev_dst = './data/train_dataset/LCQMC/dev', './data/train_dataset/LCQMC/dev.txt'
    train_src, train_dst = './data/train_dataset/LCQMC/train', './data/train_dataset/LCQMC/train.txt'

    comp_preprocess(train_src, train_dst)
    comp_preprocess(dev_src, dev_dst)

    dev_src, dev_dst = './data/train_dataset/BQ/dev', './data/train_dataset/BQ/dev.txt'
    train_src, train_dst = './data/train_dataset/BQ/train', './data/train_dataset/BQ/train.txt'

    comp_preprocess(train_src, train_dst)
    comp_preprocess(dev_src, dev_dst)
