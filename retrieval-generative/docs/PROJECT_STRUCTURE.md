# 项目结构

```text
搜广推/
├─ data/
│  ├─ README.md            数据目录入口
│  ├─ raw/                 原始数据和数据来源说明
│  └─ processed/           处理后的数据，不提交大文件
├─ src/
│  ├─ data/                MovieLens、MIND 数据读取与样本构造
│  ├─ models/              DSSM、FM、DeepFM、多兴趣、生成式、离散化
│  ├─ evaluation/          Recall、NDCG、Coverage 等指标
│  ├─ losses/              可复用损失函数
│  └─ utils/               通用工具
├─ scripts/                从项目根目录执行的训练和分析入口
├─ experiments/            按路线阶段保存实验说明和结果
├─ docs/                   学习路线、数据说明、项目结构和笔记
├─ configs/                配置说明和后续 YAML/JSON 配置
├─ notebooks/              探索性分析，不放核心逻辑
└─ tests/                  单元测试和回归测试
```

## 代码边界

`notebooks/`、`tests/`、`src/losses/` 和 `src/utils/` 当前保留为后续扩展入口，核心实验代码不依赖这些空白模块。

- 数据读取只放在 `src/data/`，训练脚本不直接解析原始文件格式。
- 模型只放在 `src/models/`，评估逻辑统一调用 `src/evaluation/`。
- `scripts/` 负责组装数据、模型、训练和输出，不承载可复用算法。
- 每个阶段的结论放在 `experiments/<阶段>/README.md`。

## 阶段入口

| 阶段 | 入口 | 说明 |
|---|---|---|
| 01 | `scripts/train_dssm.py` | DSSM 与负采样基线 |
| 02 | `scripts/train_dssm.py --negative-mode random` | 随机负采样对照 |
| 03 | `scripts/train_fm.py` / `scripts/train_deepfm.py` | FM / DeepFM |
| 04 | `scripts/train_multi_interest.py` | 多兴趣召回 |
| 05 | `scripts/run_discrete.py` | K-Means / VQ 离散化 |
| 06 | `scripts/train_generative.py` | Decoder-only 下一物品预测 |
