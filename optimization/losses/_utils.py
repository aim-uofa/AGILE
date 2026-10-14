"""Shared helpers for loss modules."""
from __future__ import annotations

import torch


def huber_loss(x: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
    """Smooth L1 loss that is less sensitive to outliers."""
    abs_x = torch.abs(x)
    return torch.where(
        abs_x <= delta,
        0.5 * (x ** 2) / delta,
        abs_x - 0.5 * delta,
    )
