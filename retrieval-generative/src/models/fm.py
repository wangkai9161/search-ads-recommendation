"""简化版 FM 风格召回打分器。"""

import torch
from torch import nn
from torch.nn import functional as F


class FMRecall(nn.Module):
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
        # 历史物品 Embedding 与候选物品 Embedding 的内积表示二阶交互。
        user = self.encode_user(histories, mask)
        item = self.item_embedding(items)
        return (user * item).sum(dim=-1) + self.item_bias(items).squeeze(-1)

    def in_batch_loss(self, histories, positives, mask=None, temperature=0.07):
        user = F.normalize(self.encode_user(histories, mask), dim=-1)
        items = F.normalize(self.item_embedding(positives), dim=-1)
        logits = user @ items.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def encode_item(self, items):
        return F.normalize(self.item_embedding(items), dim=-1)
