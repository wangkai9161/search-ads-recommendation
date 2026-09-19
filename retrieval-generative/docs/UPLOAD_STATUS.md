# Resume and Repository Upload Status

更新时间：2026-09-07

本文档用于核对简历描述与公开 GitHub 仓库，不代表尚未上传的内容已经完成。

## Public repositories

| Repository | Scope |
| --- | --- |
| [`search-recommendation-learning`](https://github.com/wangkai9161/search-recommendation-learning) | MovieLens DSSM、负采样、FM/DeepFM、多兴趣、离散表示和生成式召回 |
| [`movielens-recommendation`](https://github.com/wangkai9161/movielens-recommendation) | MovieLens Two-Tower、GRU4Rec、SASRec 和 Popularity baseline |
| [`search-ads-cvr`](https://github.com/wangkai9161/search-ads-cvr) | Criteo Sponsored Search 点击后 CVR 预估 |

## Status for current resume

| Experiment | Status | Evidence or next action |
| --- | --- | --- |
| MovieLens DSSM and next-item samples | Uploaded | `src/data/movielens.py`, `src/models/dssm.py`, `scripts/train_dssm.py` |
| Batch negatives versus random 10 negatives | Uploaded | `experiments/02_negative_sampling/` and `docs/EXPERIMENTS.md` |
| FM/DeepFM recall | Uploaded | `src/models/fm.py`, `src/models/deepfm.py` |
| Multi-interest Router | Uploaded | `src/models/multi_interest.py` and `experiments/04_multi_interest/` |
| LastFM user-artist experiment | Pending upload | Add dataset preparation, train entry, fixed configurations and result table before claiming it is in this repository |
| Two-Tower, GRU4Rec, SASRec and Popularity comparison | Uploaded separately | Use `movielens-recommendation` as the resume link; do not imply these files are in this repository |
| Criteo Sponsored Search full CVR experiment | Uploaded separately | Use `search-ads-cvr` as the resume link |
| Criteo Attribution toy CVR extension | Pending upload | The local working tree contains an extension, but it is not part of the current public commit until pushed |

## Resume wording rule

Only describe an experiment as “已实现” or “已上传” when the interviewer can open the linked repository and find its code, run command and result record. For pending items, use “待上传” or omit them from the current resume.
