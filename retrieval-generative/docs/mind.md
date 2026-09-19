# MIND 数据适配

项目已加入 [`src/data/mind.py`](../src/data/mind.py)，支持解析 MIND 的 `behaviors.tsv` 和 `news.tsv`。

当前状态：

- 数据格式解析：已实现
- 曝光未点击负样本构造：已实现
- MIND 实际下载和训练：待获取数据后执行

MIND 实验不能用 MovieLens 结果替代，因为两者的曝光机制和负样本含义不同。

