"""学习项目使用的简化版 DeepFM 召回模型。"""

import torch
from torch import nn
from torch.nn import functional as F


class DeepFMRecall(nn.Module):
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
        # FM 分支捕获二阶交互，Deep 分支学习非线性组合。
        fm = user @ items.T
        user_for_deep = user.unsqueeze(1).expand(-1, items.size(0), -1)
        item_for_deep = items.unsqueeze(0).expand(user.size(0), -1, -1)
        deep = self.deep(torch.cat([user_for_deep, item_for_deep], dim=-1)).sum(dim=-1)
        logits = (fm + deep) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
