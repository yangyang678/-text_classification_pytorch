import argparse

parser = argparse.ArgumentParser()

# 预训练参数
parser.add_argument("--gpu_id", type=int, default=1)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument('--attack_type', default='none', type=str, choices=['fgm', 'pgd', 'none'])
parser.add_argument("--fgm_epsilon", default=0.2, type=float, help="fgm epsilon")
parser.add_argument("--model_save_path", default='./pretrain_model/', type=str)
parser.add_argument("--do_resume", action='store_true', default=False)
parser.add_argument("--resume_epoch", default=0, type=int, help="the epoch where to resume")
parser.add_argument("--resume_model_path",
                    default='', type=str,
                    help="The path of the pretrained model checkpoints to resume.")
parser.add_argument("--max_seq_length", default=64, type=int)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--fp16_opt_level", type=str, default="O1")
parser.add_argument("--use_multi_gpu", action='store_true', default=False)
parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_rate", default=0.0, type=int)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--model_type", default='roberta', type=str, choices=['roberta', 'nezha_wwm', 'nezha_base'])
parser.add_argument("--train_data_path", default=['./data/train_dataset/BQ/train',
                                                  './data/train_dataset/BQ/test',
                                                  './data/train_dataset/LCQMC/train',
                                                  './data/train_dataset/LCQMC/test',
                                                  './data/train_dataset/OPPO/train'], type=list)
parser.add_argument("--valid_data_path", default=['./data/train_dataset/BQ/dev',
                                                  './data/train_dataset/LCQMC/dev',
                                                  './data/train_dataset/OPPO/dev'], type=list)
parser.add_argument("--test_data_path", default='./data/test_A.tsv', type=str)

args = parser.parse_args()

model_map = dict()

model_map['roberta'] = {'model_path': "/home/wangzhili/YangYang/pretrainModel/chinese_roberta_wwm_ext_pytorch/pytorch_model.bin",
                        'config_path': "/home/wangzhili/YangYang/pretrainModel/chinese_roberta_wwm_ext_pytorch/bert_config.json"}
model_map['nezha_wwm'] = {'model_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/pytorch_model.bin",
                          'config_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/config.json"}
model_map['nezha_base'] = {'model_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/pytorch_model.bin",
                           'config_path': "/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/config.json"}
