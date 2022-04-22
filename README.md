## 总体流程
预训练脚本:
sh pretrain.sh

微调脚本：
sh finetune.sh

预测脚本:
sh predict.sh

## 代码目录及功能介绍
1.common_utils              -- 常用工具包
 -- convert_nezha_original_tf_checkpoint_to_pytorch.py   -- tf模型转torch模型
 -- MyDataset.py                 -- 定义预训练以及微调环节的dataset类
 -- optimizer.py                 -- 优化器函数
 -- util.py                      -- 包含常用的加载数据、获取文件存储目录、日志、设置随机种子、文本截断、FGM对抗训练、Focal loss损失函数、Dice loss损失函数等功能性工具包

2.data                      -- 数据文件

3.models                    -- 模型定义
 -- downstream_model.py          -- bert下接模型结构，如idcnn
 -- finetune_model.py            -- 微调用的Model类

4.pretrain_model_utils      --预训练常用工具包
 -- nezha                        -- nezha模型

5.finetune.py               -- 微调、实现train、eval、predict

6.finetune_args.py          -- 微调代码的超参数设置

7.preprocess.py             -- 文本预处理代码

8.pretrain.py               -- 语料预训练代码

9.pretrain_args.py          -- 预训练代码的超参数设置

10.vote.py                  -- 投票融合代码






