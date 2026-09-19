# 项目二：Two-Tower 双塔召回模型

## 项目目标

基于 MovieLens 风格用户-物品交互数据，实现双塔召回模型：

- 用户 ID 和物品 ID 编码
- 按用户留一法划分训练/验证
- 用户塔和物品塔 Embedding 表示学习
- 负采样训练
- 全量候选物品打分
- Recall@K / NDCG@K 评估

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

基于 MovieLens 风格数据实现双塔召回模型，分别构建用户塔与物品塔进行向量表示学习，通过内积相似度完成候选物品召回，并采用负采样训练策略提升召回效率，使用 Recall@K、NDCG@K 对 Top-K 推荐效果进行离线评估。
