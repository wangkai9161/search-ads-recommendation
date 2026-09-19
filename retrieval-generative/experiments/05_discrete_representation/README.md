# 05 离散化表示

本阶段使用电影 Genre 多热向量做 MiniBatch K-Means，并用同一套码本演示 Vector Quantization。它先验证“连续特征 -> 离散 Item Code”的数据流，后续再将输入替换为 DSSM/Transformer 学到的 Item Embedding。

运行：

```powershell
C:\Users\wangx\miniconda3\Scripts\conda.exe run -n py310 python scripts/run_discrete.py --clusters 16
```

## 结果

配置：3,883 部电影、18 个 Genre 特征、16 个离散码。

```text
kmeans_inertia=1875.5035
codebook_utilization=1.0000
vq_unique_codes=16
```

当前实验验证了离散码生成和码本利用率统计；还没有把离散码接入生成式 Transformer。
