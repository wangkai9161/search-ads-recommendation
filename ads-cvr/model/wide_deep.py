"""Wide & Deep baseline and shared sparse/dense feature encoder."""

from __future__ import annotations

import torch
from torch import nn

from .interface import MLP, ModelInterface


class SparseDenseEncoder(nn.Module):
    """Embed hashed sparse fields and concatenate them with dense values."""

    def __init__(self, num_sparse_bins: int, num_sparse_fields: int, num_dense: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_sparse_bins, embedding_dim)
        self.output_dim = num_sparse_fields * embedding_dim + num_dense

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor | None = None) -> torch.Tensor:
        values = self.embedding(sparse.long()).flatten(start_dim=1)
        if dense is not None:
            values = torch.cat([values, dense.float()], dim=1)
        return values


class WideDeep(ModelInterface):
    """Linear memorization branch plus a deep embedding branch."""

    model_name = "wide_deep"
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
        self.deep = MLP(self.encoder.output_dim, hidden_dims, dropout)
        self.deep_out = nn.Linear(self.deep.output_dim, 1)
        self.wide_sparse = nn.Embedding(num_sparse_bins, 1)
        self.wide_dense = nn.Linear(num_dense, 1) if num_dense else None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor | None = None) -> torch.Tensor:
        sparse = sparse.long()
        deep = self.deep_out(self.deep(self.encoder(sparse, dense))).squeeze(-1)
        wide = self.wide_sparse(sparse).sum(dim=1).squeeze(-1) + self.bias
        if self.wide_dense is not None:
            if dense is None:
                raise ValueError("dense features are required because num_dense > 0")
            wide = wide + self.wide_dense(dense.float()).squeeze(-1)
        return wide + deep

    def compute_loss(self, sparse, dense, labels):
        return nn.functional.binary_cross_entropy_with_logits(self(sparse, dense), labels.float())

    @torch.no_grad()
    def predict_proba(self, sparse, dense=None):
        return torch.sigmoid(self.predict((sparse, dense)))


WideAndDeep = WideDeep
WideDeepModel = WideDeep

__all__ = ["SparseDenseEncoder", "WideDeep", "WideAndDeep", "WideDeepModel"]
