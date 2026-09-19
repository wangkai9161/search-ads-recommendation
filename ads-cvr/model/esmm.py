"""Entire Space Multi-task Model for CTR/CVR/CTCVR estimation."""

from __future__ import annotations

import torch
from torch import nn

from .interface import MLP, ModelInterface
from .wide_deep import SparseDenseEncoder


class ESMM(ModelInterface):
    """Jointly estimate CTR and CTCVR, deriving CVR from their probabilities."""

    model_name = "esmm"
    task_type = "multi_task"

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dims=(128, 64),
        num_sparse_bins: int | None = None,
        num_sparse_fields: int | None = None,
        num_dense: int | None = None,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.encoder = None
        if input_dim is None:
            if num_sparse_bins is None or num_sparse_fields is None or num_dense is None:
                raise ValueError("provide input_dim or all sparse/dense feature dimensions")
            self.encoder = SparseDenseEncoder(num_sparse_bins, num_sparse_fields, num_dense, embedding_dim)
            input_dim = self.encoder.output_dim
        self.backbone = MLP(input_dim, hidden_dims)
        self.ctr_head = nn.Linear(self.backbone.output_dim, 1)
        self.cvr_head = nn.Linear(self.backbone.output_dim, 1)

    def _features(self, features=None, dense=None, sparse=None):
        # Support model(features), model(sparse, dense), and keyword batches.
        if dense is not None and sparse is None and features is not None and not features.is_floating_point():
            sparse, features = features, None
        if self.encoder is not None:
            if sparse is None:
                sparse = features
            if sparse is None:
                raise ValueError("sparse features are required")
            return self.encoder(sparse, dense)
        if features is None:
            features = dense
        if features is None:
            raise ValueError("features are required")
        return features.float()

    def forward(self, features=None, dense=None, sparse=None):
        hidden = self.backbone(self._features(features, dense, sparse))
        ctr_logit = self.ctr_head(hidden).squeeze(-1)
        cvr_logit = self.cvr_head(hidden).squeeze(-1)
        ctr_prob = torch.sigmoid(ctr_logit)
        cvr_prob = torch.sigmoid(cvr_logit)
        ctcvr_prob = ctr_prob * cvr_prob
        return {
            "ctr_logit": ctr_logit,
            "cvr_logit": cvr_logit,
            "ctcvr_logit": torch.logit(ctcvr_prob.clamp(1e-7, 1.0 - 1e-7)),
            "ctr_prob": ctr_prob,
            "cvr_prob": cvr_prob,
            "ctcvr_prob": ctcvr_prob,
        }

    def compute_loss(self, features=None, ctr_labels=None, conversion_labels=None, dense=None, sparse=None):
        if ctr_labels is None or conversion_labels is None:
            raise ValueError("ESMM requires ctr_labels and conversion_labels")
        output = self(features, dense=dense, sparse=sparse)
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(output["ctr_logit"], ctr_labels.float())
        ctcvr_loss = nn.functional.binary_cross_entropy_with_logits(
            output["ctcvr_logit"], conversion_labels.float()
        )
        return ctr_loss + ctcvr_loss


ESMMModel = ESMM

__all__ = ["ESMM", "ESMMModel"]
