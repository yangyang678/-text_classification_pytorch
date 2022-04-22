import argparse

parser = argparse.ArgumentParser()

# 微调参数
parser.add_argument('--gpu_id', default=2, type=int,
                    help="使用的GPU id")
parser.add_argument("--maxlen", default=128, type=int,
                    help="最长句长")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--model_save_path", default='./finetune_model/', type=str,
                    help="微调模型保存路径")
parser.add_argument("--model_timestamp", default='2021-10-16_17_12_03', type=str,
                    help="微调模型保存时间戳")
parser.add_argument("--device", default='cuda', type=str,
                    help="使用GPU")
parser.add_argument("--lookahead", action='store_true',default=False,
                    help="是否使用lookahead.")
parser.add_argument("--lstm_dim", default=256, type=int,
                    help="lstm隐藏状态维度")
parser.add_argument("--gru_dim", default=256, type=int,
                    help="gru隐藏状态维度")
parser.add_argument("--do_train", action='store_true', default=False,
                    help="是否微调")
parser.add_argument("--do_predict", action='store_true', default=False,
                    help="是否预测")
parser.add_argument("--do_train_after_pretrain", action='store_true', default=False,
                    help="是否预训练后再微调")
parser.add_argument("--warmup", action='store_true', default=True,
                    help="是否采用warmup学习率策略")
parser.add_argument("--pre_model_path", default='./pretrain_model/', type=str,
                    help="预训练模型保存路径")
parser.add_argument("--pre_model_timestamp", default='2021-09-24_18_31_25', type=str,
                    help="预训练模型保存时间戳")
parser.add_argument("--lr", default=2e-5, type=float,
                    help="初始学习率")
parser.add_argument('--rDrop_coef', default=0.0, type=float,
                    help='R-Drop系数')
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--epoch", default=3, type=int,
                    help="训练轮次")
parser.add_argument('--seed', type=int, default=42,
                    help="随机种子")
parser.add_argument('--struc', default='cls', type=str,
                    choices=['cls', 'bilstm', 'bigru', 'idcnn'],
                    help="下接结构")
parser.add_argument("--dropout_num", default=1, type=int,
                    help="dropout数量")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size")
parser.add_argument("--avg_size", default=16, type=int,
                    help="平均池化窗口尺寸")
parser.add_argument("--use_avg", action='store_true', default=True,
                    help="是否使用平均池化")
parser.add_argument("--use_fgm", action='store_true', default=False,
                    help="是否使用对抗训练")
parser.add_argument("--use_rDrop", action='store_true', default=False,
                    help="是否使用r_Drop")
parser.add_argument("--fgm_epsilon", default=0.1, type=float,
                    help="fgm epsilon")
parser.add_argument("--num_classes", default=2, type=int,
                    help="类别数目")
parser.add_argument("--pre_epoch", default=-1, type=int,
                    help="选取哪个epoch的预训练模型")
parser.add_argument("--model_type", default='nezha_base', type=str, choices=['roberta', 'nezha_wwm', 'nezha_base'],
                    help="预训练模型类型")
parser.add_argument("--train_data_path", default=['./data/train_dataset/STS-B/train.data',
                                                  './data/train_dataset/STS-B/valid.data',
                                                  './data/train_dataset/PAWSX/train.data',
                                                  './data/train_dataset/PAWSX/valid.data',
                                                  './data/train_dataset/ATEC/train.data',
                                                  './data/train_dataset/ATEC/valid.data',
                                                  './data/train_dataset/OPPO/train',
                                                  './data/train_dataset/LCQMC/train',
                                                  # './data/train_dataset/LCQMC/test',
                                                  './data/train_dataset/BQ/train',
                                                  # './data/train_dataset/BQ/test',
                                                  ],
                                                  type=list)
parser.add_argument("--valid_data_path", default=[
                                                  './data/train_dataset/STS-B/test.data',
                                                  './data/train_dataset/PAWSX/test.data',
                                                  './data/train_dataset/ATEC/test.data',
                                                  './data/train_dataset/LCQMC/dev',
                                                  './data/train_dataset/BQ/dev',
                                                  './data/train_dataset/OPPO/dev'
                                                  ], type=list)
parser.add_argument("--test_data_path", default='./data/test_A.tsv', type=str)

args = parser.parse_args()

# 开源预训练模型路径
model_map = dict()

model_map['roberta'] = {
    'model_path': "/home/wangzhili/YangYang/pretrainModel/chinese-roberta-wwm-ext/pytorch_model.bin",
    'config_path': "/home/wangzhili/YangYang/pretrainModel/chinese-roberta-wwm-ext/config.json"
}
model_map['nezha_wwm'] = {
    'model_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/pytorch_model.bin",
    'config_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/config.json"
}
model_map['nezha_base'] = {
    'model_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/pytorch_model.bin",
    'config_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/config.json"
}

