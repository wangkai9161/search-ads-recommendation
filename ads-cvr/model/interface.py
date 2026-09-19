"""Common interface shared by all callable PyTorch models in this folder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ModelInterface(nn.Module):
    """Small common contract for model discovery and later training code.

    Implementations keep their natural ``forward`` signatures, while callers
    can use ``forward_batch`` for a dictionary/tuple batch and ``predict`` for
    temporary evaluation mode without changing the model's original mode.
    """

    model_name = "model"
    task_type = "generic"

    def forward_batch(self, batch):
        if isinstance(batch, Mapping):
            return self(**batch)
        if isinstance(batch, (tuple, list)):
            return self(*batch)
        return self(batch)

    @torch.no_grad()
    def predict(self, batch=None, *args, **kwargs):
        was_training = self.training
        self.eval()
        if batch is None:
            output = self(*args, **kwargs)
        elif args or kwargs:
            output = self(batch, *args, **kwargs)
        else:
            output = self.forward_batch(batch)
        if was_training:
            self.train()
        return output

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not define compute_loss")


class MLP(nn.Sequential):
    """Compact configurable MLP used by tabular and multi-task models."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout: float = 0.0):
        layers: list[nn.Module] = []
        current = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current, hidden_dim), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        super().__init__(*layers)
        self.output_dim = current


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, dim: int = 1) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)
