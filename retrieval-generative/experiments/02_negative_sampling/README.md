# 02 负采样专项

负采样代码复用 DSSM 入口，通过 `--negative-mode` 切换策略：

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_dssm.py --negative-mode in_batch --epochs 2 --max-users 1000
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_dssm.py --negative-mode random --num-negatives 10 --epochs 2 --max-users 1000
```

当前对比重点：Recall@10/50、NDCG@10/50、Item Coverage@50。本文件记录该专项的完整结果。

## 实验配置

- 数据：MovieLens 1M 前 1,000 个用户
- 训练样本：153,177
- 物品数：3,706
- Batch Size：512
- Embedding 维度：64
- Epoch：2
- 设备：CPU

## 实验结果

| 负采样方式 | Loss | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---|---:|---:|---:|---:|---:|---:|
| Batch 内负采样 | 5.5345 | 0.0370 | 0.1420 | 0.0169 | 0.0397 | 0.7275 |
| 随机 10 个负样本 | 1.0873 | 0.0350 | 0.1360 | 0.0184 | 0.0399 | 0.1811 |

## 结论

- 两种策略的 Recall@10 和 Recall@50 接近。
- Batch 内负采样的 Item Coverage 明显更高，召回结果没有过度集中在少数物品上。
- Loss 数值不能直接横向比较，因为两种 Loss 的定义和尺度不同。
- 当前结果是单次、有限用户规模实验，需要多个随机种子和更长训练进一步验证。
