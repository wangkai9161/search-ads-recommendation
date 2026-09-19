# Resume and Repository Upload Status

更新时间：2026-09-20

本文档用于核对简历描述与公开 GitHub 仓库，不代表尚未上传的内容已经完成。

## Public repositories

| Repository | Scope |
| --- | --- |
| [`retrieval-generative`](../..) | MovieLens DSSM、负采样、FM/DeepFM、多兴趣、离散表示和生成式召回 |
| [`sequential-ranking`](../../sequential-ranking/) | MovieLens Two-Tower、GRU4Rec、SASRec 和 Popularity baseline |
| [`ads-cvr`](../../ads-cvr/) | Criteo Sponsored Search 点击后 CVR 预估 |

## Status for current resume

| Experiment | Status | Evidence or next action |
| --- | --- | --- |
| MovieLens DSSM and next-item samples | Uploaded | `src/data/movielens.py`, `src/models/dssm.py`, `scripts/train_dssm.py` |
| Batch negatives versus random 10 negatives | Uploaded | `experiments/02_negative_sampling/` and `docs/EXPERIMENTS.md` |
| FM/DeepFM recall | Uploaded | `src/models/fm.py`, `src/models/deepfm.py` |
| Multi-interest Router | Uploaded | `src/models/multi_interest.py` and `experiments/04_multi_interest/` |
| LastFM user-artist experiment | Uploaded | `src/data/lastfm.py`, `scripts/train_lastfm_ablation.py`, and `../../evidence/rtx5080-20260920/lastfm/` |
| Two-Tower, GRU4Rec, SASRec and Popularity comparison | Uploaded | Unified repository subproject `sequential-ranking/`; current 10-epoch result has SASRec, not Two-Tower, as best |
| Criteo Sponsored Search full CVR experiment | Uploaded | Unified repository subproject `ads-cvr/`; four-model 15,995,634-row evidence is under `evidence/rtx5080-20260920/` |
| Criteo Attribution toy CVR extension | Pending upload | The local working tree contains an extension, but it is not part of the current public commit until pushed |

## Resume wording rule

Only describe an experiment as “已实现” or “已上传” when the interviewer can open the linked repository and find its code, run command and result record. For pending items, use “待上传” or omit them from the current resume.
