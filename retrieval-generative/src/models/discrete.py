"""无额外依赖的 MiniBatch K-Means 和向量量化工具。"""

import numpy as np


class MiniBatchKMeans:
    def __init__(self, n_clusters=32, batch_size=256, n_iter=20, seed=42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        indices = self.rng.choice(len(values), self.n_clusters, replace=False)
        self.centers = values[indices].copy()
        counts = np.ones(self.n_clusters, dtype=np.float32)
        for _ in range(self.n_iter):
            batch_ids = self.rng.choice(len(values), min(self.batch_size, len(values)), replace=False)
            batch = values[batch_ids]
            assignments = self.predict(batch)
            # 只用当前小批量更新对应的中心，模拟 MiniBatch K-Means。
            for vector, cluster in zip(batch, assignments):
                counts[cluster] += 1
                rate = 1.0 / counts[cluster]
                self.centers[cluster] += rate * (vector - self.centers[cluster])
        return self

    def predict(self, values):
        values = np.asarray(values, dtype=np.float32)
        distances = ((values[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=-1)
        return distances.argmin(axis=1)

    def quantize(self, values):
        assignments = self.predict(values)
        return self.centers[assignments], assignments


class VectorQuantizer:
    def __init__(self, codebook_size=32, seed=42):
        self.codebook_size = codebook_size
        self.seed = seed

    def fit(self, embeddings):
        self.kmeans = MiniBatchKMeans(self.codebook_size, seed=self.seed).fit(embeddings)
        return self

    def quantize(self, embeddings):
        return self.kmeans.quantize(embeddings)
