"""Deep Interest Network for target-aware user behavior modeling."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .interface import MLP, ModelInterface, masked_mean


def _mask(history, mask):
    if mask is not None:
        return mask.bool()
    return torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)


class DIN(ModelInterface):
    """Attention over history conditioned on the candidate item."""

    model_name = "din"
    task_type = "sequence_ranking"

    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dims=(128, 64)):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.attention = MLP(embedding_dim * 4, hidden_dims)
        self.attention_score = nn.Linear(self.attention.output_dim, 1)
        self.head = MLP(embedding_dim * 2, hidden_dims)
        self.output = nn.Linear(self.head.output_dim, 1)

    def encode_history(self, history, mask=None):
        return masked_mean(self.item_embedding(history.long()), _mask(history, mask))

    def _interest(self, history, target, mask):
        sequence = self.item_embedding(history.long())
        query = self.item_embedding(target.long())
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query_values = query.unsqueeze(2).expand(-1, -1, sequence.size(1), -1)
        history_values = sequence.unsqueeze(1).expand_as(query_values)
        features = torch.cat(
            [query_values, history_values, query_values - history_values, query_values * history_values],
            dim=-1,
        )
        scores = self.attention_score(self.attention(features)).squeeze(-1)
        scores = scores.masked_fill(~_mask(history, mask).unsqueeze(1), -torch.inf)
        return scores.softmax(dim=-1) @ sequence, query

    def forward(self, history, target_items=None, mask=None):
        if target_items is None:
            return self.encode_history(history, mask)
        interest, target = self._interest(history, target_items, mask)
        logits = self.output(self.head(torch.cat([interest, target], dim=-1))).squeeze(-1)
        return logits.squeeze(1) if target_items.dim() == 1 else logits

    def score_candidates(self, history, candidates, mask=None):
        return self(history, candidates, mask)

    def compute_loss(self, history, target_items, labels=None, mask=None):
        logits = self(history, target_items, mask)
        if labels is None:
            labels = torch.ones_like(logits)
        return F.binary_cross_entropy_with_logits(logits, labels.float())


DINModel = DIN

__all__ = ["DIN", "DINModel"]
