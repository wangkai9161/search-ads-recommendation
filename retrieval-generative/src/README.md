# 源码结构

- `models/`：DSSM、FM、DeepFM、多兴趣、离散化和 Decoder-only 模型
- `data/`：MovieLens 序列构造与 MIND 数据解析
- `evaluation/`：Recall@K、HitRate@K、NDCG 和 Item Coverage
- `losses/`：预留的可复用损失函数目录
- `utils/`：预留的通用工具目录

训练入口不放在 `src/`，统一位于项目根目录的 `scripts/`。模型和评估模块应保持可被多个实验脚本复用。
