"""Factorization Machine models for tabular ranking and retrieval."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .interface import ModelInterface


class FM(ModelInterface):
    """Factorization Machine with first- and second-order interactions."""

    model_name = "fm"
    task_type = "binary"

    def __init__(self, num_sparse_bins: int, num_sparse_fields: int, num_dense: int, embedding_dim: int = 16):
        super().__init__()
        self.sparse_linear = nn.Embedding(num_sparse_bins, 1)
        self.dense_linear = nn.Linear(num_dense, 1) if num_dense else None
        self.embedding = nn.Embedding(num_sparse_bins, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.sparse_linear.weight, std=0.01)
        nn.init.normal_(self.embedding.weight, std=0.01)
        if self.dense_linear is not None:
            nn.init.normal_(self.dense_linear.weight, std=0.01)
            nn.init.zeros_(self.dense_linear.bias)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor | None = None) -> torch.Tensor:
        sparse = sparse.long()
        embeddings = self.embedding(sparse)
        summed = embeddings.sum(dim=1)
        interaction = 0.5 * ((summed * summed) - (embeddings * embeddings).sum(dim=1)).sum(dim=1)
        logit = self.sparse_linear(sparse).sum(dim=1).squeeze(-1) + interaction + self.bias
        if self.dense_linear is not None:
            if dense is None:
                raise ValueError("dense features are required because num_dense > 0")
            logit = logit + self.dense_linear(dense.float()).squeeze(-1)
        return logit

    def compute_loss(self, sparse, dense, labels):
        return F.binary_cross_entropy_with_logits(self(sparse, dense), labels.float())


class FMRecall(ModelInterface):
    """FM-style user/item scorer retained for the MovieLens retrieval task."""

    model_name = "fm_recall"
    task_type = "retrieval"

    def __init__(self, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)

    def encode_user(self, histories: torch.Tensor, mask: torch.Tensor | None = None):
        embeddings = self.item_embedding(histories)
        if mask is None:
            return embeddings.mean(dim=1)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def score(self, histories: torch.Tensor, items: torch.Tensor, mask=None):
        user = self.encode_user(histories, mask)
        item = self.item_embedding(items)
        return (user * item).sum(dim=-1) + self.item_bias(items).squeeze(-1)

    def in_batch_loss(self, histories, positives, mask=None, temperature=0.07):
        user = F.normalize(self.encode_user(histories, mask), dim=-1)
        items = F.normalize(self.item_embedding(positives), dim=-1)
        logits = user @ items.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_loss(self, histories, positives, mask=None, temperature: float = 0.07):
        return self.in_batch_loss(histories, positives, mask=mask, temperature=temperature)

    def encode_item(self, items):
        return F.normalize(self.item_embedding(items), dim=-1)


__all__ = ["FM", "FMRecall"]
