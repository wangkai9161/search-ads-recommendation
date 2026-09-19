# Search Ads CVR

搜索广告点击后转化率（CVR）主项目，使用 Criteo Sponsored Search
Conversion Log 数据。每一行是一条广告点击，目标是预测点击之后是否产生
`Sale`，而不是曝光级 CTR：

```text
P(Sale = 1 | click context)
```

下载、许可、文件位置和校验值见 [`DATA.md`](DATA.md)。

## 项目定位

- 面向搜索广告粗排/转化预估：LR、FM、Wide & Deep、DeepFM。
- 研究稀疏特征交互、历史统计特征、时间切分、概率校准和样本不均衡。
- 使用 `time_delay_for_conversion` 做延迟转化分析；该字段和
  `SalesAmountInEuro` 不作为当前 CVR 输入，避免转化后信息泄漏。
- 后续扩展用户历史序列、延迟反馈修正和 CVR 实验 Agent。

## 目录

```text
search-ads-cvr/
|-- data/criteo-sponsored-search/  原始 Criteo 数据（本地大文件）
|-- prepare/                       流式读取、哈希和数值归一化
|-- model/                         LR、FM、Wide & Deep、DeepFM 等
|-- train/                         训练和超参数搜索实现
|-- result/                        指标、配置和模型 checkpoint
|-- output/                        可视化结果
`-- main.py                        统一实验入口
```

## 快速运行

```bash
cd search-ads-cvr
python main.py \
  --models deepfm \
  --embedding-dims 16 \
  --learning-rates 0.001 \
  --epochs 8 \
  --max-rows 200000 \
  --stats-rows 1000000
```

默认使用时间切分（70% train、15% validation、15% test），数值归一化统计量只从
训练段估计，并按 validation LogLoss 做 Early Stopping。完整数据运行时，将
`--max-rows 200000` 改为 `--max-rows 0`。数据文件约 6.4 GB，训练前先用小规模
smoke run 验证环境。

重复进行全量实验时建议启用一次性 mmap 缓存。缓存保存 hash 后的稀疏特征、训练
集归一化后的数值特征、标签和时间 split，后续 trial 不再解析 TSV：

```bash
python main.py \
  --models deepfm --feature-sets full,no_time,numeric_only \
  --epochs 8 --max-rows 0 --stats-rows 0 \
  --cache-dir ./cache/criteo-full \
  --device cuda
```

缓存目录已加入 `.gitignore`，不会提交到 GitHub。

特征消融可以通过同一入口完成：

```bash
python main.py \
  --models deepfm --epochs 8 --max-rows 200000 \
  --feature-sets full,no_history,no_price,no_time,sparse_only,numeric_only
```

可选特征集合分别表示完整特征、去历史点击次数、去价格、去时间、仅稀疏特征和
仅数值特征。结果会同时记录 validation 与 chronological test 的 LogLoss、ROC-AUC、
PR-AUC、Brier Score 和 ECE。

## 当前结果

### 全量四模型统一对比（2026-09-20）

在 15,995,634 行完整数据上，固定时间切分、完整特征、embedding 维度 16、学习率
0.001、随机种子 42，并设置最多 10 epoch 与 validation LogLoss Early Stopping：

| 模型 | 最佳/结束轮次 | Test LogLoss | Test PR-AUC | Test AUC | Test ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR | 10/10 | 0.196421 | 0.689681 | 0.920590 | 0.037857 |
| FM | 10/10 | 0.206346 | 0.584846 | 0.891373 | 0.007406 |
| Wide & Deep | 8/10 | 0.039846 | 0.973750 | 0.994344 | 0.004650 |
| DeepFM | 3/6 | **0.035910** | **0.977485** | **0.995453** | **0.001592** |

DeepFM 按验证集 LogLoss 选为最佳模型。Temperature Scaling 得到温度 1.0，测试
LogLoss 与 ECE 不变，未带来额外校准收益。公开的路径无关证据见
[`../evidence/rtx5080-20260920/cvr-full-four-model-report.md`](../evidence/rtx5080-20260920/cvr-full-four-model-report.md)。

在严格时间切分的 20 万行子集实验中，DeepFM 使用 embedding 维度 16、学习率
0.001，训练 8 个 epoch 后验证集 `LogLoss=0.321006`、`PR-AUC=0.479342`、
`ROC-AUC=0.804541`；独立时间外测试集 `LogLoss=0.295269`、`PR-AUC=0.464898`、
`ROC-AUC=0.817913`。该结果比旧的周期性行号切分更保守，也更接近真实上线场景。

在相同时间切分和训练预算下，对 `full`、`no_history`、`no_time`、`numeric_only`
进行了 3 个随机种子重复。测试集均值如下：`full` AUC `0.859840 ± 0.044064`、
LogLoss `0.317296 ± 0.072511`；`no_time` AUC `0.885931 ± 0.049773`、LogLoss
`0.308095 ± 0.035199`；`numeric_only` AUC `0.928432 ± 0.012586`、LogLoss
`0.252670 ± 0.006475`。当前数据上数值特征更稳定，完整稀疏 ID 交互反而带来较大
随机波动；该结论仍需在更长时间窗口和全量数据上复核。

### 全量时间切分复核

在 15,995,634 行完整数据上使用相同配置训练 2 个 epoch，结果保存在
`result/full-time-2epoch/`：`full` 的 Test LogLoss/AUC/PR-AUC 为
`0.036665 / 0.996125 / 0.980274`，`no_time` 为
`0.035076 / 0.996189 / 0.980348`，`numeric_only` 为
`0.061701 / 0.979530 / 0.952694`。因此 20 万行实验中 `numeric_only` 的优势
不能直接推广到全量数据，最终结论应以完整数据和后续多 seed 复核为准。

启用 mmap 缓存后，三组配置以最多 8 个 epoch 重新训练，结果保存在
`result/experiments/full-cache-8ep-gpu2/`：`full` 在第 3 轮达到最佳验证 LogLoss，
Test LogLoss/AUC/PR-AUC 为 `0.035917 / 0.995463 / 0.977513`；`no_time` 在第 2 轮
达到最佳，指标为 `0.034990 / 0.996200 / 0.980408`；`numeric_only` 第 8 轮仍在
改善，指标为 `0.047038 / 0.980400 / 0.953623`。完整任务（含缓存构建、三组训练、
评估和校准）耗时约 549 秒。

对验证集拟合 Temperature Scaling 后，最终 `no_time` 测试集 LogLoss 从 `0.034990`
降至 `0.034929`，ECE 从 `0.003695` 降至 `0.003067`；校准文件位于实验目录的
`calibration/` 子目录。

### GPU 全量管线验证

使用外部 py310 环境、GPU 2 在
15,995,634 行完整数据上运行 DeepFM 2 个 epoch 的旧基线结果仍保存在
`result/full-gpu-deepfm-e2/`。该结果使用旧的周期性 holdout，仅作历史对照；新的
默认入口已经改为时间切分并同时保存 validation/test 指标。

## 实验路线

1. LR、FM、DeepFM 基线和特征交互对比。
2. 随机/周期切分与严格时间切分对比。
3. 去掉 `nb_clicks_1week` 等历史统计特征的消融实验。
4. 从 `user_id`、`product_id`、`click_timestamp` 重建点击历史，接入 DIN/DIEN。
5. 预测转化延迟，比较 1 天、7 天、30 天观察窗口。
6. Agent 读取 `result/` 中的指标，分析问题并生成下一组实验配置。

## 数据边界

当前文件是点击级转化日志，没有曝光未点击样本。因此本项目不把自己包装成
完整 CTR/ESMM 系统；如果要严格建模 CTR 或 ESMM，需要曝光级日志。

旧实验路径仅作历史说明，新的代码和
文档入口统一使用本目录。
