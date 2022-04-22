import os
import pandas as pd
import logging
import datetime
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional



def load_data(filename,style):
    """
    加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    if style == "default":
        data = open(filename, encoding='UTF-8')
        D = []
        for text in data.readlines():
            text_list = text.strip().split('\t')
            text1 = text_list[0]
            text2 = text_list[1]

            if len(text1) == 0 or len(text2) == 0:
              continue

            if len(text_list) > 2:
                label = int(text_list[2])
                # label转为one-hot向量
                label = [1, 0] if label == 0 else [0, 1]
            else:
                label = [-100, -100]
            # 逐字存入
            D.append((list(text1), list(text2), label))
        return D
    elif style == "STS-B":
        D = []
        with open(filename, encoding='UTF-8') as f:
            for l in f.readlines():
                text_list = l.split('\t')
                text1 = text_list[0]
                text2 = text_list[1]

                if len(text1) == 0 or len(text2) == 0:
                    continue

                if len(text_list) > 2:
                    label = int(text_list[2])
                    # label转为one-hot向量
                    label = [1, 0] if label < 3 else [0, 1]
                else:
                    label = [-100, -100]
                # 逐字存入
                D.append((list(text1), list(text2), label))
            return D
    else :
        D = []
        with open(filename, encoding='UTF-8') as f:
            for l in f.readlines():
                text_list = l.split('\t')
                text1 = text_list[0]
                text2 = text_list[1]

                if len(text1) == 0 or len(text2) == 0:
                    continue

                if len(text_list) > 2:
                    label = int(text_list[2])
                    # label转为one-hot向量
                    label = [1, 0] if label == 0 else [0, 1]
                else:
                    label = [-100, -100]
                # 逐字存入
                D.append((list(text1), list(text2), label))
            return D






def get_save_path(args):
    # 获取开始时间
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    # 模型保存路径
    args.model_save_path = args.model_save_path + "{}/".format(timestamp)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # 记录模型配置参数
    argsDict = args.__dict__
    with open(args.model_save_path + 'args.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def get_logger(log_file, display2console=True):
    """
    定义日志方法
    :param log_file:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为DEBUG
    logger.setLevel(logging.DEBUG)
    # 创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 创建一个控制台的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    if display2console:
        logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_sequences(maxlen, index, *sequences):
    """
    截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


def compute_kl_loss(p, q, pad_mask = None):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# 对抗训练FGM
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: float = 0.0,
                 alpha: float = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position=True) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target),
                                                                               -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(target,
                                num_classes=logits_size).float() if self.index_label_position else target.float()
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:,
                                                                           label_idx] >= threshold) | pos_example.view(
                        -1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num + 1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)
