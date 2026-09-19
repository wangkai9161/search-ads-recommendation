# 配置目录

建议将数据路径、模型超参数、负采样策略和评估参数写入 YAML 配置。

推荐字段：

- `data`: 数据集和切分配置
- `model`: 模型类型和维度
- `sampling`: 负采样策略、数量、温度
- `train`: Batch Size、学习率、Epoch
- `eval`: K 值和评估指标

