# 03 FM 风格召回

## 实现

本实验使用用户历史物品和候选物品的二阶 Embedding 交互，配合 Batch 内负采样训练。它是 FM 的教学版召回实现，重点观察特征交叉思路，不等同于完整 DeepFM 排序模型。

## 运行

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_fm.py --epochs 2 --max-users 300 --batch-size 512
```

## 结果

配置：300 用户、43,145 个训练样本、3,706 个物品、2 Epoch、CPU。

| Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---:|---:|---:|---:|---:|
| 0.0200 | 0.0767 | 0.0117 | 0.0242 | 0.6384 |

## DeepFM 对照

DeepFM 教学实现位于 `src/models/deepfm.py`，复用 DSSM 的用户历史样本和 Batch 内负采样，增加 FM 交叉项与 MLP。

运行：

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/train_deepfm.py --epochs 2 --max-users 300
```

配置：300 用户、43,145 个训练样本、3,706 个物品、2 Epoch、CPU。

| Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Item Coverage@50 |
|---:|---:|---:|---:|---:|
| 0.0200 | 0.1100 | 0.0067 | 0.0261 | 0.5046 |
