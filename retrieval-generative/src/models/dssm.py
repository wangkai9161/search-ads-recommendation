"""简化版 DSSM 双塔召回模型。"""

import torch
from torch import nn
from torch.nn import functional as F


class DSSM(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_item_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def encode_user(self, histories: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.user_item_embedding(histories)
        if mask is None:
            pooled = embeddings.mean(dim=1)
        else:
            # padding 位置不能参与用户兴趣平均，否则会误计为第 0 个物品。
            weights = mask.unsqueeze(-1).to(embeddings.dtype)
            pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.user_projection(pooled), dim=-1)

    def encode_item(self, items: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(items), dim=-1)

    def in_batch_loss(
        self,
        histories: torch.Tensor,
        positives: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        user_vectors = self.encode_user(histories, mask)
        item_vectors = self.encode_item(positives)
        # 同一 Batch 中其他正样本作为当前用户的负样本。
        logits = user_vectors @ item_vectors.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def random_negative_loss(
        self,
        histories: torch.Tensor,
        positives: torch.Tensor,
        num_items: int,
        mask: torch.Tensor | None = None,
        num_negatives: int = 10,
    ) -> torch.Tensor:
        user_vectors = self.encode_user(histories, mask)
        positive_vectors = self.encode_item(positives)
        # 随机采样物品作为对照负样本，并避免直接采到当前正样本。
        negative_ids = torch.randint(
            0, num_items, (positives.size(0), num_negatives), device=positives.device
        )
        negative_ids = torch.where(
            negative_ids == positives.unsqueeze(1),
            (negative_ids + 1) % num_items,
            negative_ids,
        )
        negative_vectors = self.encode_item(negative_ids.reshape(-1)).reshape(
            positives.size(0), num_negatives, -1
        )
        positive_scores = (user_vectors * positive_vectors).sum(dim=-1)
        negative_scores = (user_vectors.unsqueeze(1) * negative_vectors).sum(dim=-1)
        positive_loss = F.softplus(-positive_scores).mean()
        negative_loss = F.softplus(negative_scores).mean()
        return positive_loss + negative_loss
