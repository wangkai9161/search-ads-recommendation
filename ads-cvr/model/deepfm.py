"""DeepFM models for tabular prediction and MovieLens retrieval."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .interface import MLP, ModelInterface
from .wide_deep import SparseDenseEncoder


class DeepFM(ModelInterface):
    """FM low-order interactions combined with a DNN high-order branch."""

    model_name = "deepfm"
    task_type = "binary"

    def __init__(
        self,
        num_sparse_bins: int,
        num_sparse_fields: int,
        num_dense: int,
        embedding_dim: int = 16,
        hidden_dims=(128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SparseDenseEncoder(num_sparse_bins, num_sparse_fields, num_dense, embedding_dim)
        self.sparse_linear = nn.Embedding(num_sparse_bins, 1)
        self.dense_linear = nn.Linear(num_dense, 1) if num_dense else None
        self.deep = MLP(self.encoder.output_dim, hidden_dims, dropout)
        self.deep_out = nn.Linear(self.deep.output_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.sparse_linear.weight, std=0.01)
        nn.init.normal_(self.encoder.embedding.weight, std=0.01)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor | None = None) -> torch.Tensor:
        sparse = sparse.long()
        embeddings = self.encoder.embedding(sparse)
        summed = embeddings.sum(dim=1)
        fm = 0.5 * ((summed * summed) - (embeddings * embeddings).sum(dim=1)).sum(dim=1)
        linear = self.sparse_linear(sparse).sum(dim=1).squeeze(-1) + self.bias
        if self.dense_linear is not None:
            if dense is None:
                raise ValueError("dense features are required because num_dense > 0")
            linear = linear + self.dense_linear(dense.float()).squeeze(-1)
        deep = self.deep_out(self.deep(self.encoder(sparse, dense))).squeeze(-1)
        return linear + fm + deep

    def compute_loss(self, sparse, dense, labels):
        return F.binary_cross_entropy_with_logits(self(sparse, dense), labels.float())


class DeepFMRecall(ModelInterface):
    """DeepFM-style user/item scorer retained for the retrieval task."""

    model_name = "deepfm_recall"
    task_type = "retrieval"

    def __init__(self, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.deep = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)

    def encode_user(self, histories, mask=None):
        embeddings = self.item_embedding(histories)
        if mask is None:
            return embeddings.mean(dim=1)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode_item(self, items):
        return F.normalize(self.item_embedding(items), dim=-1)

    def in_batch_loss(self, histories, positives, mask=None, temperature=0.07):
        user = self.encode_user(histories, mask)
        items = self.item_embedding(positives)
        fm = user @ items.T
        user_for_deep = user.unsqueeze(1).expand(-1, items.size(0), -1)
        item_for_deep = items.unsqueeze(0).expand(user.size(0), -1, -1)
        deep = self.deep(torch.cat([user_for_deep, item_for_deep], dim=-1)).sum(dim=-1)
        logits = (fm + deep) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_loss(self, histories, positives, mask=None, temperature: float = 0.07):
        return self.in_batch_loss(histories, positives, mask=mask, temperature=temperature)


DeepFMCVR = DeepFM
DeepFMModel = DeepFM

__all__ = ["DeepFM", "DeepFMModel", "DeepFMCVR", "DeepFMRecall"]
