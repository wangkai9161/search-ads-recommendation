# 脚本入口

脚本统一从项目根目录执行，使用 `py310` 环境：

```powershell
$conda = 'C:\Users\wangx\miniconda3\Scripts\conda.exe'
& $conda run -n py310 python scripts/<script>.py
```

| 脚本 | 用途 |
|---|---|
| `train_dssm.py` | DSSM、随机负采样、Batch 内负采样 |
| `train_fm.py` | FM 风格召回 |
| `train_deepfm.py` | DeepFM 风格召回 |
| `train_multi_interest.py` | 多兴趣召回 |
| `run_discrete.py` | MiniBatch K-Means / VQ 风格离散化 |
| `train_generative.py` | Decoder-only 下一物品预测 |
| `run_seed_sweep.ps1` | DSSM 多随机种子入口 |

