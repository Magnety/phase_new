from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, feature: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return feature.view_as(feature)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient.neg().mul(ctx.scale), None


def gradient_reverse(feature: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(feature, float(scale))


class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_domains: int,
        hidden_dim: int,
        dropout: float,
        *,
        class_conditional: bool,
        num_condition_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_domains = int(num_domains)
        self.class_conditional = bool(class_conditional)
        self.num_condition_classes = int(num_condition_classes)
        if self.class_conditional and self.num_condition_classes < 2:
            raise ValueError("Conditional domain heads require at least two classes")
        output_dim = self.num_domains * (
            self.num_condition_classes if self.class_conditional else 1
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, feature: torch.Tensor, *, reverse: bool, scale: float = 1.0
    ) -> torch.Tensor:
        if reverse:
            feature = gradient_reverse(feature, scale)
        logits = self.classifier(feature)
        if self.class_conditional:
            logits = logits.reshape(
                feature.shape[0], self.num_condition_classes, self.num_domains
            )
        return logits


class BinaryPrototypeHead(nn.Module):
    def __init__(self, feature_dim: int, initial_scale: float = 8.0) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(2, feature_dim))
        nn.init.orthogonal_(self.prototypes)
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.normalize(feature.float(), dim=-1)
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        similarity = feature @ prototypes.transpose(0, 1)
        scale = self.log_scale.exp().clamp(1.0, 40.0)
        return scale * (similarity[:, 1] - similarity[:, 0]) + self.bias

    def separation_loss(self, margin: float = 0.2) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        cosine = (prototypes[0] * prototypes[1]).sum()
        return F.relu(cosine - float(margin)).square()


class MulticlassPrototypeHead(nn.Module):
    def __init__(
        self, feature_dim: int, num_classes: int, initial_scale: float = 8.0
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("A multiclass head requires at least two classes")
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        nn.init.orthogonal_(self.prototypes)
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.normalize(feature.float(), dim=-1)
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        scale = self.log_scale.exp().clamp(1.0, 40.0)
        return scale * (feature @ prototypes.transpose(0, 1)) + self.bias

    def separation_loss(self, margin: float = 0.2) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        similarities = prototypes @ prototypes.transpose(0, 1)
        off_diagonal = ~torch.eye(
            self.num_classes, device=similarities.device, dtype=torch.bool
        )
        return F.relu(similarities[off_diagonal] - float(margin)).square().mean()


class CoxRiskHead(nn.Module):
    """Unbounded log-risk for Cox proportional-hazards optimization."""

    def __init__(self, feature_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.predictor(feature.float()).squeeze(-1)

    def separation_loss(self, margin: float = 0.2) -> torch.Tensor:
        del margin
        return self.predictor[-1].weight.sum() * 0.0


__all__ = [
    "BinaryPrototypeHead",
    "CoxRiskHead",
    "DomainDiscriminator",
    "MulticlassPrototypeHead",
    "gradient_reverse",
]
