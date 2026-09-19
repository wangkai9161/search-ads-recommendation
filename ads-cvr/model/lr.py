"""Logistic-regression baseline for sparse and dense ranking features."""

from __future__ import annotations

import torch
from torch import nn

from .interface import ModelInterface


class LR(ModelInterface):
    """Linear model over hashed sparse fields and dense features."""

    model_name = "lr"
    task_type = "binary"

    def __init__(self, num_sparse_bins: int, num_sparse_fields: int | None = None, num_dense: int = 0):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.sparse_linear = nn.Embedding(num_sparse_bins, 1)
        self.dense_linear = nn.Linear(num_dense, 1) if num_dense else None
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.sparse_linear.weight, mean=0.0, std=0.01)
        if self.dense_linear is not None:
            nn.init.normal_(self.dense_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.dense_linear.bias)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor | None = None) -> torch.Tensor:
        logit = self.sparse_linear(sparse.long()).sum(dim=1).squeeze(-1) + self.bias
        if self.dense_linear is not None:
            if dense is None:
                raise ValueError("dense features are required because num_dense > 0")
            logit = logit + self.dense_linear(dense.float()).squeeze(-1)
        return logit

    def compute_loss(self, sparse, dense, labels):
        return nn.functional.binary_cross_entropy_with_logits(self(sparse, dense), labels.float())

    @torch.no_grad()
    def predict_proba(self, sparse, dense=None):
        return torch.sigmoid(self.predict((sparse, dense)))


LogisticCVR = LR

__all__ = ["LR", "LogisticCVR"]
