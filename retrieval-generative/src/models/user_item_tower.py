"""User/item two-tower model for implicit-feedback retrieval."""

import torch
from torch import nn
from torch.nn import functional as F


class UserItemTwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_projection(self.user_embedding(user_ids)), dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_projection(self.item_embedding(item_ids)), dim=-1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return (self.encode_user(user_ids) * self.encode_item(item_ids)).sum(dim=-1)
