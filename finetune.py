import pandas as pd
from finetune_args import args, model_map
from time import time
from tqdm import tqdm
from transformers.models.roberta.configuration_roberta import RobertaConfig
from pretrain_model_utils.nezha.configuration_nezha import NeZhaConfig
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from common_utils.optimizer import Lookahead
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch
import torch.nn as nn
import os
from common_utils.util import load_data, get_save_path, get_logger, set_seed, FGM, compute_kl_loss
from common_utils.Datasets import DatasetForSentencePair, BlockShuffleDatasetForSentencePair
from common_utils.DataLoaders import BlockShuffleDataLoader
from models.finetune_model import Model
import random

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

MODEL_CONFIG = {'nezha_wwm': 'NeZhaConfig', 'nezha_base': 'NeZhaConfig', 'roberta': 'RobertaConfig'}

#读取训练数据
train_data, valid_data = [], []
for train_path in args.train_data_path[:2]:
    train_data += load_data(train_path,"STS-B")

for train_path in args.train_data_path[2:6]:
    train_data += load_data(train_path, "else")

for train_path in args.train_data_path[6:]:
    train_data += load_data(train_path,"default")

for valid_path in args.valid_data_path[:1]:
    valid_data += load_data(valid_path,"STS-B")

for valid_path in args.valid_data_path[1:3]:
    valid_data += load_data(valid_path,"else")

for valid_path in args.valid_data_path[3:]:
    valid_data += load_data(valid_path,"default")
def train(model):
    # 创建模型保存路径以及记录配置的参数
    get_save_path(args)
    # 创建日志对象，写入训练效果
    logger = get_logger(args.model_save_path + '/finetune.log')
    # 创建数据集
    train_dataset = BlockShuffleDatasetForSentencePair(train_data, args.maxlen)
    valid_dataset = BlockShuffleDatasetForSentencePair(valid_data, args.maxlen)
    # 读取数据集
    train_loader = BlockShuffleDataLoader(train_dataset, batch_size=args.batch_size, is_shuffle=True,
                                          sort_key=lambda x: len(x[0]) + len(x[1]),
                                          collate_fn=train_dataset.collate)
    valid_loader = BlockShuffleDataLoader(valid_dataset, batch_size=args.batch_size, is_shuffle=False,
                                          sort_key=lambda x: len(x[0]) + len(x[1]),
                                          collate_fn=valid_dataset.collate)

    # 训练优化参数设置
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # 优化器设置
    if args.lookahead:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                      eps=args.adam_epsilon)
        optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                      eps=args.adam_epsilon)
    # 学习率衰减
    total_steps = len(train_loader) * args.epoch
    warmup_steps = int(total_steps * 0)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # 损失函数
    criterion = nn.BCELoss()

    # 是否使用对抗生成网络
    fgm = None
    if args.use_fgm:
        fgm = FGM(model)
    model.zero_grad()

    # 训练
    best_acc = 0.
    early_stop = 0
    for epoch in range(args.epoch):
        model.train()
        losses, acc_list = [], []
        pbar = tqdm(train_loader, ncols=150, desc='训练中')
        # 一个batch的训练
        for data in pbar:
            # 梯度是累积计算而不是被替换，因此每个batch将梯度初始化为零
            optimizer.zero_grad()
            # 组织待输入模型的数据，转到GPU上
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long(),
            }
            data['label'] = data['label'].to(args.device).float()


            if args.use_rDrop:
                # keep dropout and forward twice
                outputs = model.forward(inputs)
                outputs2 = model.forward(inputs)
                # cross entropy loss for classifier
                ce_loss = 0.5 * (criterion(outputs, data['label']) + criterion(outputs2, data['label']))
                kl_loss = compute_kl_loss(outputs, outputs2)
                # carefully choose hyper-parameters
                loss = ce_loss + args.rDrop_coef * kl_loss
            else:
                outputs = model.forward(inputs)
                loss = criterion(outputs, data['label'])

            loss.backward()

            if args.use_fgm:
                fgm.attack(epsilon=args.fgm_epsilon)
                outputs_adv = model(inputs)
                loss_adv = criterion(outputs_adv, data['label'])
                loss_adv.backward()
                fgm.restore()

            optimizer.step()
            if args.warmup:
                scheduler.step()

            losses.append(loss.cpu().detach().numpy())
            output_array = outputs.cpu().detach().numpy()
            label_array = data['label'].cpu().detach().numpy()
            acc_list.extend(np.argmax(output_array, axis=1) == np.argmax(label_array, axis=1))
            pbar.set_description(
                f'epoch:{epoch + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

        # 评估验证集
        acc, report = evaluate(model, valid_loader)
        if acc > best_acc:
            early_stop = 0
            best_acc = acc
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            # 保存最优效果模型
            torch.save(model.state_dict(), args.model_save_path + f'/{args.model_type}_{args.struc}_best_model.pth',
                       _use_new_zipfile_serialization=False)
        # 效果增长不起时候停掉训练
        else:
            early_stop += 1
            if early_stop > 3:
                break

        logger.info(f'epoch:{epoch + 1}/{args.epoch}, vaild acc: {acc}, best_acc: {best_acc}')
        logger.info(f'{report}')


def evaluate(model, data_loader):
    model.eval()
    true, preds = [], []
    pbar = tqdm(data_loader, ncols=150)
    with torch.no_grad():
        for data in pbar:
            data['label'] = data['label'].float()
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long(),
            }
            outputs = model(inputs)
            pred = np.argmax(outputs.cpu().numpy(), axis=-1)
            true.extend(np.argmax(data['label'].cpu().numpy(), axis=-1))
            preds.extend(pred)
    acc = accuracy_score(true, preds)
    report = classification_report(true, preds)

    return acc, report


def predict(model):
    time_start = time()
    set_seed(args.seed)

    test_data = load_data(args.test_data_path,"default")

    test_dataset = DatasetForSentencePair(test_data, args.maxlen)
    test_loader = DataLoader(test_dataset, args.batch_size)

    model.eval()
    preds = []
    # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=150):
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long()
            }
            outputs = model(inputs)
            pred = np.argmax(outputs.cpu().numpy(), axis=-1)
            preds.extend(pred)

    pd.DataFrame({'label': [i for i in preds]}).to_csv(
        args.model_save_path + args.model_timestamp + "/result.csv", index=False, header=None)
    time_end = time()
    print(f'finish {time_end - time_start}s')


def main():
    set_seed(args.seed)

    # 把MODEL_CONFIG列表里的字符串转为变量名
    # 读取预训练模型配置

    config = globals()[MODEL_CONFIG[args.model_type]].from_json_file(model_map[args.model_type]['config_path'])
    model = Model(bert_config=config, args=args)
    # 训练模式
    if args.do_train:
        # 读取已有预训练模型的参数
        state_dict = torch.load(model_map[args.model_type]['model_path'], map_location='cuda')
        # 加载预训练参数
        model.load_state_dict(state_dict, strict=False)
        model = model.to(args.device)
        train(model)

    elif args.do_train_after_pretrain:
        file_dir = args.pre_model_path + args.pre_model_timestamp
        # 读取路径目录
        file_list = os.listdir(file_dir)
        for name in file_list:
            if name == f'{args.model_type}.pth' or name.split('.')[-1] != 'pth':
                continue
            model_path = os.path.join(file_dir, name)
            if os.path.isfile(model_path) and name.split('-')[1] == f'epoch{args.pre_epoch}.pth':
                print('pretrain model: ', name)
                state_dict = torch.load(model_path, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device)
                train(model)
    # 预测
    elif args.do_predict:
        state_dict = torch.load(
            args.model_save_path + args.model_timestamp + f'/{args.model_type}_{args.struc}_best_model.pth',
            map_location='cuda')
        model.load_state_dict(state_dict)
        model = model.to(args.device)
        predict(model)


if __name__ == '__main__':
    main()
