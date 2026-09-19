"""将 MovieLens 电影 Genre 向量聚类为离散 Item Code。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.discrete import MiniBatchKMeans, VectorQuantizer


def load_genre_features(path):
    rows = []
    genres = set()
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        item_id, title, genre_text = line.split('::')
        current = genre_text.split('|')
        genres.update(current); rows.append((int(item_id), current))
    genre_list = sorted(genres); index = {name: i for i, name in enumerate(genre_list)}
    values = np.zeros((len(rows), len(genre_list)), dtype=np.float32)
    item_ids = []
    for row, (item_id, current) in enumerate(rows):
        item_ids.append(item_id)
        values[row, [index[name] for name in current]] = 1.0
    return item_ids, values, genre_list


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-file', default='data/raw/ml-1m/movies.dat')
    parser.add_argument('--clusters', type=int, default=16); args = parser.parse_args()
    item_ids, values, genres = load_genre_features(args.data_file)
    # 先用 Genre 多热特征验证连续特征到离散码本的完整链路。
    model = MiniBatchKMeans(args.clusters).fit(values); codes = model.predict(values)
    quantized, _ = model.quantize(values)
    inertia = float(((values - quantized) ** 2).sum())
    utilization = len(set(codes.tolist())) / args.clusters
    vq = VectorQuantizer(args.clusters).fit(values); _, vq_codes = vq.quantize(values)
    print(f'items={len(item_ids)} features={len(genres)} codebook={args.clusters}')
    print(f'kmeans_inertia={inertia:.4f} codebook_utilization={utilization:.4f}')
    print(f'vq_unique_codes={len(set(vq_codes.tolist()))}')


if __name__ == '__main__': main()
