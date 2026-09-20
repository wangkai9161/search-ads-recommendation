# Search Ads CVR

搜索广告点击后转化率主项目，使用 Criteo Sponsored Search Conversion Log。每行是一条广告点击，预测目标是：

```text
P(Sale = 1 | click context)
```

数据下载、许可、校验值和字段边界见 [`DATA.md`](DATA.md)。

## 严格数据边界

- 使用 15,995,634 条点击，按时间 70%/15%/15% 切分。
- `SalesAmountInEuro` 和 `time_delay_for_conversion` 是转化后字段，始终排除。
- 审计发现 `product_price` 在测试段中与标签近确定相关：90.85% 为 0，非零价格样本的转化率为 100%，非转化样本的非零价格比例为 0%。因此主实验也排除 `product_price`。
- clean 特征包含 16 个类别字段及 `click_timestamp`、`nb_clicks_1week`；数值统计量只由训练段估计。
- 类别哈希使用字段命名空间，0 号桶只表示缺失，避免跨字段同值产生系统性碰撞。

## 运行

```bash
python main.py \
  --models lr,fm,wide_deep,deepfm \
  --feature-sets clean \
  --embedding-dims 16 --learning-rates 0.001 \
  --epochs 10 --max-rows 0 --stats-rows 0 \
  --cache-dir ./cache/criteo-full --device cuda
```

DeepFM clean 消融：

```bash
python main.py \
  --models deepfm \
  --feature-sets clean,clean_no_user_id,clean_no_product_id,clean_no_entity_ids,clean_coarse_context \
  --seeds 41,42,43 --epochs 10 --max-rows 0 --stats-rows 0 \
  --cache-dir ./cache/criteo-full --device cuda
```

主指标按验证 LogLoss 选择 checkpoint，测试集只评估一次；同时记录 PR-AUC、ROC-AUC、Brier 和 ECE。

## Clean 四模型结果

| 模型 | Test LogLoss | Test PR-AUC | Test AUC | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| LR | 0.279342 | 0.268036 | 0.765749 | **0.005451** |
| FM | 0.278196 | 0.276872 | 0.770010 | 0.008619 |
| Wide & Deep | 0.315669 | 0.243251 | 0.740691 | 0.044198 |
| DeepFM | **0.274910** | **0.285382** | **0.779100** | 0.011312 |

## DeepFM 三随机种子消融

| 特征 | Test LogLoss | Test PR-AUC | Test AUC |
| --- | ---: | ---: | ---: |
| clean | 0.275297 +/- 0.000761 | 0.285394 +/- 0.000128 | **0.777870 +/- 0.001254** |
| clean_no_user_id | **0.274713 +/- 0.000570** | 0.288641 +/- 0.001203 | 0.777521 +/- 0.000649 |
| clean_no_product_id | 0.275063 +/- 0.000780 | 0.285660 +/- 0.001777 | 0.776302 +/- 0.001271 |
| clean_no_entity_ids | 0.274787 +/- 0.000716 | **0.290042 +/- 0.002098** | 0.776285 +/- 0.001836 |
| clean_coarse_context | 0.295249 +/- 0.000103 | 0.197438 +/- 0.000204 | 0.712745 +/- 0.000447 |

去掉用户/商品实体 ID 后基本不下降，说明主结果不是 ID 记忆。删除更多品牌、卖家和受众上下文后明显下降，表明这些上下文仍有有效信号。

价格字段只保留作诊断：`price_only` AUC 为 `0.965585 +/- 0.000259`，不能作为可信模型能力指标。完整审计见 [`../evidence/rtx5080-20260920/EXPERIMENT_REPORT_CN.md`](../evidence/rtx5080-20260920/EXPERIMENT_REPORT_CN.md)。

## 项目边界

该数据只有点击样本，因此当前项目是 post-click CVR，不是 CTR、CTCVR 或 ESMM。结果均为公开数据集离线证据，不代表线上收入提升。
