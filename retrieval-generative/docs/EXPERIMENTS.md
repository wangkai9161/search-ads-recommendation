# 实验汇总

## 统一评估

- `Recall@K`：目标物品是否进入候选集。
- `NDCG@K`：目标物品在候选集中的位置质量。
- `Item Coverage@K`：所有用户候选覆盖的物品比例。
- 生成式召回额外记录候选唯一率和长尾比例。

## 负采样对照

| 策略 | Recall@10 | Recall@50 | NDCG@10 | Coverage@50 |
|---|---:|---:|---:|---:|
| Batch 内负采样 | 0.0370 | 0.1420 | 0.0169 | 0.7275 |
| 随机 10 个负样本 | 0.0350 | 0.1360 | 0.0184 | 0.1811 |

配置和完整讨论见 [`../experiments/02_negative_sampling/README.md`](../experiments/02_negative_sampling/README.md)。

## 其他模型

公平对比中，改进后的多兴趣 Router 在 300 用户、2 Epoch 下达到 Recall@10 `0.0300`、Recall@50 `0.0967`、Coverage@50 `0.3546`；同条件 DSSM 为 `0.0067`、`0.0467`、`0.1349`。

在相同的 300 用户、2 Epoch 配置下，FM 为 Recall@10 `0.0200`、Recall@50 `0.0767`、NDCG@10 `0.0117`、Coverage@50 `0.6384`；DeepFM 为 `0.0200`、`0.1100`、`0.0067`、`0.5046`。生成式模型因 CPU Transformer 评估成本较高，采用 100 用户、32 维、1 Epoch 的教学配置，结果见 [`../experiments/06_generative_recall/README.md`](../experiments/06_generative_recall/README.md)。
