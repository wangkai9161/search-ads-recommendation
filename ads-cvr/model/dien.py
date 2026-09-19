"""Deep Interest Evolution Network with GRU-based interest dynamics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .din import DIN, _mask


class DIEN(DIN):
    """Evolve per-step interests before applying target-aware attention."""

    model_name = "dien"

    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dims=(128, 64)):
        super().__init__(num_items, embedding_dim, hidden_dims)
        self.evolution = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.auxiliary_head = nn.Linear(embedding_dim, num_items)

    def _interest(self, history, target, mask):
        sequence = self.item_embedding(history.long())
        evolved, _ = self.evolution(sequence)
        query = self.item_embedding(target.long())
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query_values = query.unsqueeze(2).expand(-1, -1, evolved.size(1), -1)
        history_values = evolved.unsqueeze(1).expand_as(query_values)
        features = torch.cat(
            [query_values, history_values, query_values - history_values, query_values * history_values],
            dim=-1,
        )
        scores = self.attention_score(self.attention(features)).squeeze(-1)
        scores = scores.masked_fill(~_mask(history, mask).unsqueeze(1), -torch.inf)
        return scores.softmax(dim=-1) @ evolved, query

    def auxiliary_loss(self, history, next_items, mask=None):
        """Optional next-item supervision for the interest extractor."""
        evolved, _ = self.evolution(self.item_embedding(history.long()))
        logits = self.auxiliary_head(evolved)
        valid = _mask(history, mask)
        if next_items.shape == history.shape:
            return F.cross_entropy(logits[valid], next_items.long()[valid])
        return F.cross_entropy(logits[:, -1], next_items.long())


DIENModel = DIEN

__all__ = ["DIEN", "DIENModel"]
