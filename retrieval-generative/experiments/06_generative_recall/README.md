# 06 生成式召回

## 实现

本实验使用带因果 Mask 的 Transformer 做下一物品预测。模型读取用户历史序列，输出最后一个有效位置的隐藏向量，再预测整个物品词表；这是 Decoder-only 生成式召回的最小教学版本。

## 运行

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_generative.py --epochs 1 --max-users 100 --batch-size 128 --embedding-dim 32
```

## 结果

配置：100 用户、12,776 个训练样本、3,706 个物品、1 Epoch、CPU，32 维 Embedding。为控制 CPU 评估耗时，生成式模型单独使用教学规模，不与 300 用户判别式模型横向比较。

| Loss | Recall@10 | Recall@50 | Coverage@50 | Unique Ratio | Tail Ratio |
|---:|---:|---:|---:|---:|---:|
| 8.1677 | 0.0300 | 0.0700 | 0.1819 | 0.1348 | 0.0242 |

评估已改为按 batch 推理，并保留历史物品屏蔽、候选覆盖、唯一率和长尾比例；当前结果主要验证自回归训练与生成式候选质量统计链路。

## 补充指标

`train_generative.py` 额外输出：

- `item_coverage@50`：所有用户 Top-50 候选覆盖的物品比例
- `candidate_unique_ratio`：跨用户候选去重后的比例
- `tail_ratio`：候选中低于测试集物品频次中位数的物品比例
