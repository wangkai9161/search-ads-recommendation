"""使用因果 Mask 的 Decoder-only 风格下一物品预测模型。"""

import torch
from torch import nn


class DecoderOnlyRecall(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(256, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=heads, dim_feedforward=embedding_dim * 4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(embedding_dim, num_items)

    def forward(self, histories, mask=None):
        positions = torch.arange(histories.size(1), device=histories.device).unsqueeze(0)
        hidden = self.item_embedding(histories) + self.position_embedding(positions)
        length = histories.size(1)
        # 上三角为 True，禁止当前位置看到未来行为。
        causal = torch.triu(torch.ones(length, length, device=histories.device), diagonal=1).bool()
        padding = None if mask is None else ~mask
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=padding)
        if mask is None:
            last = hidden[:, -1]
        else:
            # 不同用户序列长度不同，只取每条序列最后一个有效位置。
            last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
            last = hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]
        return self.output(last)
