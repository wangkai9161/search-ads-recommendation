# MovieLens Recommendation

推荐系统辅助项目，使用 MovieLens-1M 构造用户-物品交互和行为序列，覆盖召回、
排序和序列推荐。它用于补充搜索广告 CVR 项目中缺少的用户历史建模经验，不把
MovieLens 实验包装成广告 CTR/CVR 结果。

## 项目定位

- 双塔召回：用户塔、物品塔、负采样、全量候选检索。
- 序列推荐：SASRec/Transformer、下一物品预测、Recall@K 和 NDCG@K。
- 排序基线：DeepFM 和 DIN，练习稀疏特征交互与目标感知兴趣建模。

## 目录

```text
movielens-recommendation/
|-- data/movielens/ml-1m/   MovieLens-1M 原始数据
|-- train_movielens_deepfm.py
|-- train_movielens_din.py
|-- train_movielens_din_v2.py
|-- recall-twotower/         双塔召回和负采样
|-- sequence-sasrec/         SASRec、GRU4Rec 和 PopRec 对比
`-- outputs/                 训练结果、指标和图表
```

## 快速运行

安装项目依赖。双塔和 SASRec 的 shell 入口会使用当前环境中的 `python`：

```bash
python -m pip install -r requirements.txt
```

排序和 DIN 实验：

```bash
python train_movielens_deepfm.py
python train_movielens_din.py
```

双塔召回：

```bash
cd recall-twotower
bash run_demo.sh
```

序列推荐：

```bash
cd sequence-sasrec
bash run_compare.sh
```

双塔和 SASRec 目录自带小规模样例数据生成脚本，不需要把完整数据复制到
实验输出目录。真实 MovieLens-1M 数据位于项目根目录的 `data/` 下。

### GPU 全量管线验证

以下结果均使用外部 `py310` 环境和 GPU 0（RTX 5080），MovieLens-1M 全量数据，
每个模型运行 1 个 epoch；checkpoint 和召回结果归档在 `outputs/full-gpu/`：

- DeepFM：测试集 AUC `0.785817`，LogLoss `0.554593`（1,000,209 条样本）。
- DIN：测试集 AUC `0.750190`，LogLoss `0.592270`（994,169 条序列样本）。
- DIN-V2：测试集 AUC `0.778170`，LogLoss `0.587020`（988,310 条序列样本）。
- TwoTower：Recall@20 `0.068389`，NDCG@20 `0.024591`。
- SASRec：Recall@20 `0.016231`，NDCG@20 `0.005155`。

统一序列比较入口（`train_sequence_models.py --model all`）另外得到：
PopRec Recall@20 `0.042722` / NDCG@20 `0.014712`，GRU4Rec Recall@20
`0.020205` / NDCG@20 `0.007084`，SASRec Recall@20 `0.020371` /
NDCG@20 `0.006478`。

召回模型采用每个用户最后一条交互做 leave-one-out 验证；这些是管线验证结果，
不是多轮调参后的最终模型结论。

## 与 CVR 项目的分工

```text
search-ads-cvr             MovieLens Recommendation
广告点击后转化预测         用户-物品召回和序列推荐
Sale / 转化延迟             Recall@K / NDCG@K
DeepFM / FM                DSSM / DIN / SASRec
```

MovieLens 中的评分阈值标签是教学用的偏好标签，不等价于广告曝光、点击或购买。
它适合用来解释用户历史序列、负采样和召回评估；广告 CVR 结论统一放在
`search-ads-cvr` 项目中。

旧的 `recommender-systems/ctr-demo` 和 `recommender-suite` 路径保留兼容链接，
新的项目入口统一使用本目录。
