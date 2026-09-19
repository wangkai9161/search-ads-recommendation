# 搜广推召回与生成式推荐实践

实践项目：从双塔召回出发，逐步实现负采样、FM/DeepFM、多兴趣、离散化和生成式推荐。
研究重点：负采样策略如何影响 Top-K 召回、候选覆盖和生成结果质量。

## 项目介绍

- 从 DSSM 双塔基线开始，统一比较随机负采样和 Batch 内负采样。
- 使用 `Recall@10/50`、`NDCG@10/50`、`Item Coverage` 等召回指标，避免只看 AUC。
- 实现 FM、DeepFM、多兴趣召回和 Decoder-only 下一物品预测教学版本。
- 增加 MiniBatch K-Means / VQ 风格 Item Code 实验。
- 实验使用 MovieLens-1M 与 HetRec LastFM 2K，训练入口可由 `py310` 环境直接运行。

## 简历与仓库对应关系

简历中的相关内容分布在统一仓库的三个子项目。下面的状态以面试官当前能够打开的公开代码和证据为准；标记为“待上传”的内容暂不应描述为已经完成。

| 简历内容 | 对应仓库 | 当前状态 |
| --- | --- | --- |
| MovieLens-1M DSSM、Batch 内/随机负采样、FM/DeepFM、多兴趣和生成式召回 | 本仓库 | 已上传 |
| LastFM 用户--艺术家双塔、负样本 0~5、BCE/BPR 和长尾权重 | 本仓库 | 已上传：`src/data/lastfm.py`、`scripts/train_lastfm_ablation.py` 与统一证据目录 |
| MovieLens Two-Tower、GRU4Rec、SASRec、Popularity baseline | [`sequential-ranking`](../sequential-ranking/) | 已合并为同一仓库的独立子项目 |
| Criteo Sponsored Search 约 1,600 万条点击日志、LR/FM/Wide&Deep/DeepFM CVR | [`ads-cvr`](../ads-cvr/) | 已合并为同一仓库的独立子项目 |
| Criteo Attribution CVR toy 基线 | 本仓库 | 待上传：当前本地有扩展代码，公开仓库以本 README 状态为准 |

详细状态记录见 [`docs/UPLOAD_STATUS.md`](docs/UPLOAD_STATUS.md)。

## 3 分钟快速判断

| 你可能关心的问题 | 当前仓库证据 |
| --- | --- |
| 是否只是写了 README？ | `src/models/` 有 DSSM、FM、DeepFM、多兴趣、离散化和生成式模型实现；`scripts/` 有训练入口。 |
| 是否有可比实验？ | `experiments/` 按阶段记录 DSSM、负采样、FM/DeepFM、多兴趣和生成式召回结果。 |
| 是否只看 AUC？ | 统一使用 `Recall@10/50`、`NDCG@10/50`、`Item Coverage@50`，更贴近召回阶段。 |
| 是否夸大成工业广告系统？ | 没有。当前使用 MovieLens-1M，定位为可复现离线实验；广告 CVR 扩展路径在下方单独说明。 |

## LastFM 双塔消融

在 HetRec LastFM 2K 的 92,826 条有效用户--艺术家交互上，对每个用户固定随机留出
一个测试目标。原数据不含时间戳，因此该切分不描述为时间切分。所有配置训练 10
轮并在过滤历史交互后的完整物品库上评估。

| 配置 | 最佳轮次 | Recall@20 | NDCG@20 | Coverage@20 |
| --- | ---: | ---: | ---: | ---: |
| BCE，0 个负样本 | 9 | 0.018047 | 0.006980 | 0.121014 |
| BCE，3 个负样本 | 2 | 0.074841 | 0.025292 | 0.047487 |
| BPR，5 个负样本 | 1 | 0.078556 | 0.030690 | 0.002326 |
| BCE，5 个负样本，长尾加权 | 1 | **0.085987** | 0.031289 | 0.003574 |

长尾加权提高了本轮总体 Recall，但各配置的稀有艺术家 Tail Recall@20 仍为 0，
不能据此声称已经解决长尾召回。完整 9 组结果见
[`../evidence/rtx5080-20260920/lastfm/report.md`](../evidence/rtx5080-20260920/lastfm/report.md)。

## 项目路线

```text
MovieLens-1M 行为序列
  -> DSSM 双塔召回基线
  -> 随机负采样 / Batch 内负采样对照
  -> FM / DeepFM 特征交互实验
  -> 多兴趣 Router 召回
  -> MiniBatch K-Means / VQ 风格离散表示
  -> Decoder-only 下一物品生成式召回

Criteo Attribution 日志（待上传）
  -> post-click CVR / impression-level CVR 样本
  -> Logistic Regression / DeepFM CVR 基线
  -> LogLoss / AUC 评估
```

## 结果展示

### 对比结果：300 用户、2 Epoch

| 实验 | Recall@10 | Recall@50 | NDCG@10 | Item Coverage@50 |
| --- | ---: | ---: | ---: | ---: |
| DSSM + Batch 内负采样 | 0.0067 | 0.0467 | 0.0019 | 0.1349 |
| 多兴趣 Router，4 兴趣 | 0.0300 | 0.0967 | 0.0145 | 0.3546 |

结论：在相同训练规模下，多兴趣 Router 相比单向量 DSSM 明显提升 Top-K 命中和候选覆盖。这个结果用于说明用户行为序列中存在多兴趣表达需求，而不是声称已经达到工业线上指标。

### 其他教学规模结果

FM、DeepFM 使用 300 用户、2 Epoch；负采样主对照使用 1,000 用户、2 Epoch；Decoder-only 使用 100 用户、32 维、1 Epoch。不同规模结果不直接横向比较。

| 实验 | Recall@10 | Recall@50 | NDCG@10 | Item Coverage@50 |
| --- | ---: | ---: | ---: | ---: |
| DSSM + Batch 内负采样 | 0.0370 | 0.1420 | 0.0169 | 0.7275 |
| DSSM + 随机 10 负样本 | 0.0350 | 0.1360 | 0.0184 | 0.1811 |
| FM 风格召回 | 0.0200 | 0.0767 | 0.0117 | 0.6384 |
| DeepFM 风格召回 | 0.0200 | 0.1100 | 0.0067 | 0.5046 |
| Decoder-only 生成式召回 | 0.0300 | 0.0700 | - | 0.1819 |

初步观察：Batch 内负采样与随机负采样的 Top-K 命中接近，但候选覆盖差异明显；Router 改进后，多兴趣模型的候选得到缓解。FM、DeepFM 已统一到 300 用户、2 Epoch；生成式模型因 Transformer 需求大评估成本较高，所以使用 100 用户、32 维、1 Epoch 的训练配置，不能与上表直接横向比较。

## 和搜索广告 CVR 的关系

这个仓库当前完成的是 **召回、行为序列建模、负采样、离线评估和公开广告转化数据上的 CVR 最小链路**，不是完整广告 CVR 生产系统。它对搜索广告 CVR 岗位的价值在于：

- 候选召回和粗排前置建模：DSSM、多兴趣召回、Top-K 检索、候选覆盖分析。
- 用户行为序列表征：从历史行为构造下一物品预测样本，可迁移到用户-查询-广告上下文。
- 特征交互基础：FM/DeepFM 为 CTR/CVR 稀疏特征交叉做铺垫。
- CVR 扩展入口：Criteo Attribution 代码目前待上传；已公开的完整 Criteo Sponsored Search CVR 实验见 [`ads-cvr`](../ads-cvr/)。
- 离线评估意识：区分 Recall/NDCG/覆盖率，避免只看单一指标。

如果继续扩展为更完整的广告 CTR/CVR 项目，下一步会补充：

1. 曝光-点击-转化样本构造，区分 `CTR = P(click | impression)`、`CVR = P(conversion | click)`、`CTCVR = P(click, conversion | impression)`。
2. 用户、query、ad、context、time 等稀疏/稠密特征。
3. Wide&Deep、DeepFM、DCN、ESMM/MMoE 等 CTR/CVR 基线。
4. 延迟转化、负采样校正、校准和线上/离线指标偏差分析。

## 快速复现

建议 Python 3.10。原始 MovieLens-1M 数据不提交到仓库，下载和放置方式见 [`data/raw/README.md`](data/raw/README.md)。

```bash
pip install -r requirements.txt
```

运行 DSSM Batch 内负采样：

```bash
python scripts/train_dssm.py --epochs 2 --max-users 1000 --negative-mode in_batch
```

运行随机负采样对照：

```bash
python scripts/train_dssm.py --epochs 2 --max-users 1000 --negative-mode random --num-negatives 10
```

运行多兴趣召回：

```bash
python scripts/train_multi_interest.py --epochs 2 --max-users 300 --num-interests 4
```

运行 LastFM 消融：

```bash
python scripts/train_lastfm_ablation.py \
  --data-file data/raw/lastfm/user_artists.dat \
  --output-dir outputs/lastfm-ablation \
  --epochs 10 --device cuda
```

> Criteo Attribution CVR toy 基线目前标记为“待上传”，因此暂不提供公开仓库内的运行命令。

## 仓库结构

```text
data/          MovieLens 数据说明与原始数据占位
configs/       实验配置说明
docs/          项目结构、学习路线和边界说明
experiments/   分阶段实验记录和结果
notebooks/     探索性笔记
scripts/       MovieLens 训练与实验入口
src/data/      MovieLens、LastFM、MIND 数据读取
src/evaluation/Recall、NDCG、Item Coverage
src/models/    DSSM、FM、DeepFM、多兴趣、生成式、离散化模型
tests/         基础测试
```

## 待上传扩展

以下是本地扩展代码预期采用的相对路径结构，目前尚未包含在公开仓库中：

```text
model/       模型注册与入口
prepare/     数据读取、清洗、归一化
train/       训练入口与模型选择
output/      模型文件和可视化产物
result/      训练结果、指标和可视化数据
```

其中 Criteo Attribution toy 基线、上述扩展目录以及对应脚本均标记为“待上传”。

## 已完成与边界

已完成：

- MovieLens-1M 行为序列和下一物品预测样本。
- DSSM、负采样对照、FM/DeepFM、多兴趣召回、离散表示和生成式召回最小链路。
- Full-catalog Top-K 检索、历史物品过滤、Recall/NDCG/Item Coverage 评估。
- Criteo Attribution CVR 的扩展代码当前标记为待上传；已公开的完整 Criteo Sponsored Search CVR 实验位于 [`ads-cvr`](../ads-cvr/)。

当前边界：

- 不是工业推荐/广告线上系统。
- 未声称完成真实 MIND 全量曝光日志训练。
- 未包含工业搜索广告全链路、延迟反馈校正、校准或线上 A/B 验证。
- 当前结果仍需要更多随机种子和更大训练规模验证。
