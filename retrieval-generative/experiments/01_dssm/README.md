# 01 DSSM 双塔召回

## 当前实现

- 使用用户历史物品 Embedding 的平均池化表示用户
- 使用独立物品塔生成物品向量
- 使用余弦相似度计算用户和物品匹配分数
- 支持 Batch 内负采样和随机负采样
- 使用最后一次行为作为测试目标
- 评估时排除用户历史物品
- 使用全物品库计算 Recall@10、Recall@50、NDCG 和 Item Coverage

## 运行方式

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_dssm.py --epochs 2 --max-users 1000
```

参数说明：

- `--max-users`：限制用户数，用于快速验证
- `--max-train-samples`：限制训练样本数
- `--temperature`：Batch 内对比损失的温度参数
- `--embedding-dim`：用户塔和物品塔向量维度

## 当前验证状态

- 数据读取和序列切分：已实现
- 模型代码语法检查：已通过
- 训练运行：已完成

## 实验结果：Batch 内负采样

配置：

- 数据：MovieLens 1M 前 1,000 个用户
- 训练样本：153,177
- 物品数：3,706
- Batch Size：512
- Embedding 维度：64
- Temperature：0.07
- 设备：CPU
- Epoch：2

| Epoch | Loss | Recall@10 |
|---:|---:|---:|
| 1 | 6.2120 | 0.0080 |
| 2 | 5.5345 | 0.0370 |

完整指标：

| Epoch | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0110 | 0.0480 | 0.0055 | 0.0129 | 0.2558 |
| 2 | 0.0370 | 0.1420 | 0.0169 | 0.0397 | 0.7275 |

## 负采样对照

同样使用 1,000 个用户、153,177 个训练样本、Batch Size 512 和 2 个 Epoch。完整负采样专项记录见 [`../02_negative_sampling/README.md`](../02_negative_sampling/README.md)：

| 负采样方式 | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---|---:|---:|---:|---:|---:|
| Batch 内，Epoch 2 | 0.0370 | 0.1420 | 0.0169 | 0.0397 | 0.7275 |
| 随机 10 个，Epoch 2 | 0.0350 | 0.1360 | 0.0184 | 0.0399 | 0.1811 |

初步结论：两种方法的 Top-K 命中效果接近，但 Batch 内负采样的 Item Coverage 明显更高。这个结论仍需要更多随机种子和更长训练验证，不能仅凭一次运行下定论。

## 超参数探索

以下实验仅使用 300 个用户和 1 个 Epoch，用于观察训练早期行为，不与主实验直接比较：

| Batch Size | Temperature | Recall@10 | Recall@50 | Item Coverage@50 |
|---:|---:|---:|---:|---:|
| 256 | 0.07 | 0.0000 | 0.0000 | 0.0219 |
| 512 | 0.20 | 0.0000 | 0.0033 | 0.0283 |

## 结论边界

当前实验完成了指标、负采样和基础超参数链路，但还没有完成多随机种子、完整用户规模和线上指标验证。后续应固定测试集，至少重复 3 个随机种子，再比较采样策略。

## 下一步实验

- [x] 运行 DSSM 基线
- [x] 增加随机负采样作为对照
- [x] 比较 Batch Size 和温度参数
- [x] 记录 Recall@10、Recall@50、NDCG 和 Item 覆盖率
