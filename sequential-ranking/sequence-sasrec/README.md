# 项目三：SASRec / Transformer 序列推荐模型

## 项目目标

基于 MovieLens 风格用户行为序列，实现 SASRec 序列推荐：

- 用户历史行为序列构造
- 序列截断与 padding
- Item Embedding + Position Embedding
- Transformer Encoder 建模用户动态兴趣
- 下一物品预测
- Recall@K / NDCG@K 离线评估

## 运行

```bash
pip install -r requirements.txt
bash run_demo.sh
```

## 输出

```text
outputs/
├── best_model.pth
├── train_history.csv
└── result.csv
```

## 简历表述

基于用户历史交互序列实现 SASRec 序列推荐模型，通过 Item Embedding、Position Embedding 与 Self-Attention 机制建模用户动态兴趣，完成下一物品预测任务，并使用 Recall@K、NDCG@K 对序列推荐效果进行离线评估。
