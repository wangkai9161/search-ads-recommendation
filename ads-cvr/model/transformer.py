"""Transformer sequence models: SASRec and decoder-style next-item prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .interface import ModelInterface


class SASRec(ModelInterface):
    """Self-attentive sequential recommendation with causal masking."""

    model_name = "transformer"
    task_type = "sequence_ranking"

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_len: int = 200,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)

    def encode_sequence(self, history, mask=None):
        history = history.long()[:, -self.max_len :]
        if mask is not None:
            mask = mask.bool()[:, -self.max_len :]
        if mask is None:
            mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        length = history.size(1)
        positions = torch.arange(length, device=history.device).unsqueeze(0)
        hidden = self.item_embedding(history) + self.position_embedding(positions)
        causal = torch.triu(torch.ones(length, length, device=history.device), diagonal=1).bool()
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=~mask)
        hidden = self.layer_norm(hidden)
        last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]

    def forward(self, history, target_items=None, mask=None):
        sequence = self.encode_sequence(history, mask)
        if target_items is None:
            return sequence
        target_items = target_items.long()
        target = self.item_embedding(target_items)
        bias = self.item_bias(target_items).squeeze(-1)
        if target.dim() == 2:
            return (sequence * target).sum(dim=-1) + bias
        return (sequence.unsqueeze(1) * target).sum(dim=-1) + bias

    def score_candidates(self, history, candidates, mask=None):
        return self(history, candidates, mask)

    def compute_loss(self, history, target_items, mask=None):
        sequence = self.encode_sequence(history, mask)
        logits = sequence @ self.item_embedding.weight.T + self.item_bias.weight.squeeze(-1)
        return F.cross_entropy(logits, target_items.long())


class DecoderOnlyRecall(ModelInterface):
    """Causal Transformer encoder used as a decoder-style next-item model."""

    model_name = "decoder_transformer"
    task_type = "sequence_generation"

    def __init__(self, num_items: int, embedding_dim: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(256, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(embedding_dim, num_items)

    def forward(self, histories, mask=None):
        positions = torch.arange(histories.size(1), device=histories.device).unsqueeze(0)
        hidden = self.item_embedding(histories.long()) + self.position_embedding(positions)
        length = histories.size(1)
        causal = torch.triu(torch.ones(length, length, device=histories.device), diagonal=1).bool()
        padding = None if mask is None else ~mask.bool()
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=padding)
        if mask is None:
            last = hidden[:, -1]
        else:
            last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
            last = hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]
        return self.output(last)

    def compute_loss(self, histories, targets, mask=None):
        return F.cross_entropy(self(histories, mask), targets.long())


Transformer = SASRec
TransformerSequence = SASRec
SASRecModel = SASRec

__all__ = ["SASRec", "SASRecModel", "Transformer", "TransformerSequence", "DecoderOnlyRecall"]
