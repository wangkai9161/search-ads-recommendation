# 04 MIND / 多兴趣召回

## 实现

本实验使用一个用户级 Router 将行为分配到 4 个兴趣槽位；训练时使用温度 `0.2` 的 `logsumexp` 聚合多个兴趣，并加入较小权重的兴趣多样性辅助 Loss。它用于理解 MIND 的核心思想；不是论文中完整的 Capsule Routing 复现。

## 运行

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_multi_interest.py --epochs 1 --max-users 300 --batch-size 512 --num-interests 4
```

公平对比配置：

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_multi_interest.py --epochs 2 --max-users 300 --batch-size 512 --num-interests 4 --diversity-weight 0.01
```

## 改进前结果

配置：300 用户、43,145 个训练样本、3,706 个物品、4 个兴趣、1 Epoch、CPU。

| Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---:|---:|---:|---:|---:|
| 0.0000 | 0.0033 | 0.0000 | 0.0006 | 0.0165 |

这组结果出现明显候选塌缩，因此没有直接作为多兴趣模型结论。

## Router 改进结果

改动：用用户级 Router 将每个历史行为分配到 4 个兴趣槽位；使用 `logsumexp` 聚合兴趣；温度设为 `0.2`，多样性 Loss 权重设为 `0.01`。

配置：300 用户、43,145 个训练样本、3,706 个物品、4 个兴趣、2 Epoch、CPU。

| Epoch | Loss | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.2742 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0194 |
| 2 | 6.0890 | 0.0300 | 0.0967 | 0.0145 | 0.0289 | 0.3546 |

同条件 DSSM 基线（300 用户、2 Epoch）为 Recall@10 `0.0067`、Recall@50 `0.0467`、NDCG@10 `0.0019`、Coverage@50 `0.1349`。改进后的多兴趣模型已明显缓解候选塌缩，但仍需要更多随机种子和更大规模实验验证。
