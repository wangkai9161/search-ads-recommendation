# 原始数据说明

## MovieLens 1M

- 来源：[GroupLens MovieLens](https://grouplens.org/datasets/movielens/1m/)
- 下载地址：`https://files.grouplens.org/datasets/movielens/ml-1m.zip`
- 下载日期：2026-07-26
- 用户数：6,040
- 电影记录数：3,883
- 评分数：1,000,209

文件格式：

- `users.dat`：`UserID::Gender::Age::Occupation::Zip-code`
- `movies.dat`：`MovieID::Title::Genres`
- `ratings.dat`：`UserID::MovieID::Rating::Timestamp`

原始压缩包和官方 README 保留在本目录。使用前请阅读 `ml-1m/README` 中的许可说明；不要将原始数据重新分发到公开代码仓库。

2026-09-20 复现实验使用的 `ratings.dat`：

- 文件大小：`24,594,131` bytes
- SHA-256：`506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f`

## HetRec 2011 LastFM 2K

- 来源：<https://files.grouplens.org/datasets/hetrec2011/>
- 下载地址：`https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip`
- 原始交互文件：`user_artists.dat`

解压后放置为：

```text
data/raw/lastfm/user_artists.dat
```

2026-09-20 复现实验使用的文件大小为 `1,296,455` bytes，SHA-256 为
`001400dc3c7d2667fca6e4ea6dc6acc31a9dd28ad5cd0f74cea988c019934d3b`。
原始文件有约 9.3 万条隐式反馈；固定留出前会去重，并过滤不足 2 条交互的用户，
最终纳入实验 92,826 条交互。使用前请阅读压缩包中的官方 README 和许可说明。
