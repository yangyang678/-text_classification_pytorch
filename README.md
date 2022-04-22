## 代码框架介绍
该代码框架复用性与解耦性比较高。在这里大致说明一下怎么去使用这个框架。对于一个问题，我们首先想的是解决问题的办法，也就是模型构建部分models。当模型确定了，那我们就要构建数据迭代器（util.py）给模型喂数据了，而util.py读入的数据是preprocess.py清洗干净的数据。

当构建以上这几部分之后，便是模型训练部分finetune.py，这个部分包含训练、验证F1和保存每一个epoch训练模型的过程。一开始我们训练单模得先确定单模是否有效，可以通过finetune.py将训练集和验证集都用验证集去表示，看一下验证集F1是否接近90%，若接近则说明我们的模型构建部分没有出错，但不保证我们的F1评估公式是否写错。因此，我们使用刚刚用验证集训练得到的模型，通过predict.py来预测验证集，人工检验预测的结果是否有效，这样子就能保证我们整体的单模流程完全没问题了。

最后就是后处理规则postprocess和融合vote两部分，这里的主观性比较强，一般都是根据具体问题具体分析来操作。

其中，util.py可以用来检验我们构造的Batch数据是否有误，可以直接打印人工检验一下输入的情况。

整个框架的超参数在finetune_args.py以及pretrain_args.py处设置，加强框架的解耦性，避免了一处修改，处处修改的情况。

整体的框架也可复用到其他问题上，只需要根据我们修改的model.py来确定喂入的Batch数据格式，其他的代码文件也只是根据具体问题去修改相应部分，降低了调试成本。

### 整体流程
预训练脚本: sh pretrain.sh

微调脚本: sh finetune.sh

预测脚本: sh predict.sh

### 代码目录及功能介绍
1.common_utils              -- 常用工具包
```
   -- convert_nezha_original_tf_checkpoint_to_pytorch.py   -- tf模型转torch模型

   -- MyDataset.py                 -- 定义预训练以及微调环节的dataset类

   -- optimizer.py                 -- 优化器函数

   -- util.py                      -- 包含常用的加载数据、获取文件存储目录、日志、设置随机种子、文本截断、FGM对抗训练、Focal loss损失函数、Dice loss损失函数等功能性工具包
```

2.data                      -- 数据文件


3.models                    -- 模型定义

```
   -- downstream_model.py          -- bert下接模型结构，如idcnn

   -- finetune_model.py            -- 微调用的Model类
 ```
 

4.pretrain_model_utils      --预训练常用工具包
```

   -- nezha                        -- nezha模型
 
```

5.finetune.py               -- 微调、实现train、eval、predict


6.finetune_args.py          -- 微调代码的超参数设置


7.preprocess.py             -- 文本预处理代码


8.pretrain.py               -- 语料预训练代码


9.pretrain_args.py          -- 预训练代码的超参数设置


10.vote.py                  -- 投票融合代码



### 模型方法
* Nezha + BILSTM + CRF
* Nezha + IDCNN + CRF
* 动态权重Nezha + IDCNN + CRF
* 动态权重Nezha + BILSTM + CRF

![](https://github.com/yangyang678/text_classification_pytorch/blob/master/__pycache__/BILSTM.png?raw=true)
![](https://github.com/yangyang678/text_classification_pytorch/blob/master/__pycache__/IDCNN.png)
![](https://github.com/yangyang678/text_classification_pytorch/blob/master/__pycache__/%E5%8A%A8%E6%80%81%E6%9D%83%E9%87%8D%E8%9E%8D%E5%90%88.png)


