from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from ..tasks import TaskSpec, task_specs
from .common import TASKS, task_value


def domain_classification_loss(
    logits: torch.Tensor | None,
    center_id: torch.Tensor | None,
) -> torch.Tensor | None:
    if logits is None or center_id is None:
        return None
    domains = center_id.to(logits.device).reshape(-1).long()
    valid = (domains >= 0) & (domains < logits.shape[-1])
    if not bool(valid.any()):
        return logits.float().sum() * 0.0
    return F.cross_entropy(logits[valid].float(), domains[valid])


def class_conditional_domain_loss(
    logits_by_task: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor] | torch.Tensor,
    label_mask: Mapping[str, torch.Tensor] | torch.Tensor,
    center_id: torch.Tensor | None,
    *,
    tasks: Sequence[str] = TASKS,
    specs: Mapping[str, TaskSpec] | None = None,
    survival_event: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    definitions = dict(specs or task_specs(tasks))
    values: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}
    if center_id is None:
        return None, details
    for task in tasks:
        logits = logits_by_task.get(task)
        spec = definitions.get(task)
        if logits is None:
            continue
        target = task_value(labels, task, logits.device)
        mask = task_value(label_mask, task, logits.device)
        if target is None or mask is None:
            continue
        domains = center_id.to(logits.device).reshape(-1).long()
        if spec is not None and spec.kind == "survival":
            if survival_event is None:
                continue
            classes = survival_event.to(logits.device).reshape(-1).long()
        else:
            classes = target.reshape(-1).long()
        valid = (
            (mask.reshape(-1) > 0)
            & (domains >= 0)
            & (domains < logits.shape[-1])
            & (classes >= 0)
            & (classes < logits.shape[1])
        )
        if bool(valid.any()):
            rows = torch.arange(logits.shape[0], device=logits.device)
            conditioned = logits[rows, classes.clamp(0, logits.shape[1] - 1)]
            value = F.cross_entropy(conditioned[valid].float(), domains[valid])
            values.append(value)
            details[f"domain/{task}"] = value
    return (torch.stack(values).mean() if values else None), details


def group_dro_loss(
    predictions: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor] | torch.Tensor,
    label_mask: Mapping[str, torch.Tensor] | torch.Tensor,
    center_id: torch.Tensor | None,
    *,
    tasks: Sequence[str] = TASKS,
    specs: Mapping[str, TaskSpec] | None = None,
    temperature: float = 0.2,
    min_group_samples: int = 2,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Smooth worst task × centre × class loss, not only mean accuracy."""

    definitions = dict(specs or task_specs(tasks))
    first = next(iter(predictions.values()), None)
    if first is None or center_id is None:
        return None, {}
    domains = center_id.to(first.device).reshape(-1).long()
    group_losses: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}
    for task in tasks:
        logits = predictions.get(task)
        spec = definitions.get(task)
        target = task_value(labels, task, first.device)
        mask = task_value(label_mask, task, first.device)
        if logits is None or target is None or mask is None:
            continue
        if spec is None or spec.kind == "survival":
            continue
        target = target.reshape(-1)
        valid = (mask.reshape(-1) > 0) & (domains >= 0)
        if spec.kind == "binary":
            target = target.float().clamp(0.0, 1.0)
            sample_loss = F.binary_cross_entropy_with_logits(
                logits.reshape(-1).float(), target, reduction="none"
            )
        else:
            target = target.long()
            valid &= (target >= 0) & (target < spec.num_classes)
            sample_loss = F.cross_entropy(
                logits.float(), target.clamp(0, spec.num_classes - 1), reduction="none"
            )
        task_groups: list[torch.Tensor] = []
        for domain in torch.unique(domains[valid]):
            for label in torch.unique(target[valid]):
                selected = valid & (domains == domain) & (target.long() == label)
                if int(selected.sum()) >= int(min_group_samples):
                    task_groups.append(sample_loss[selected].mean())
        if task_groups:
            values = torch.stack(task_groups)
            details[f"group_dro_worst/{task}"] = values.max()
            group_losses.extend(task_groups)
    if not group_losses:
        return first.float().sum() * 0.0, details
    values = torch.stack(group_losses)
    scale = max(float(temperature), 1e-4)
    robust = scale * (
        torch.logsumexp(values / scale, dim=0)
        - values.new_tensor(float(values.numel())).log()
    )
    details["group_dro_worst"] = values.max()
    details["group_dro_mean"] = values.mean()
    return robust, details


def cross_center_supervised_contrastive_loss(
    features: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor] | torch.Tensor,
    label_mask: Mapping[str, torch.Tensor] | torch.Tensor,
    center_id: torch.Tensor | None,
    *,
    tasks: Sequence[str] = TASKS,
    specs: Mapping[str, TaskSpec] | None = None,
    temperature: float = 0.15,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Pull same-class cases across centres and repel opposite classes."""

    definitions = dict(specs or task_specs(tasks))
    if center_id is None:
        return None, {}
    values: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}
    for task in tasks:
        spec = definitions.get(task)
        if spec is None or spec.kind == "survival":
            continue
        feature = features.get(task)
        if feature is None:
            continue
        target = task_value(labels, task, feature.device)
        mask = task_value(label_mask, task, feature.device)
        if target is None or mask is None:
            continue
        valid = mask.reshape(-1) > 0
        embedding = F.normalize(feature[valid].float(), dim=-1)
        target = target.reshape(-1)[valid].long()
        domains = center_id.to(feature.device).reshape(-1)[valid].long()
        if embedding.shape[0] < 3:
            continue
        similarities = embedding @ embedding.T / max(float(temperature), 1e-4)
        eye = torch.eye(embedding.shape[0], device=feature.device, dtype=torch.bool)
        positive = (
            (target[:, None] == target[None, :])
            & (domains[:, None] != domains[None, :])
            & ~eye
        )
        denominator = ~eye
        valid_anchor = positive.any(dim=1)
        if not bool(valid_anchor.any()):
            continue
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
        log_probability = similarities - torch.logsumexp(
            similarities.masked_fill(~denominator, float("-inf")), dim=1, keepdim=True
        )
        per_anchor = -(log_probability * positive.float()).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        value = per_anchor[valid_anchor].mean()
        values.append(value)
        details[f"contrastive/{task}"] = value
    return (torch.stack(values).mean() if values else None), details


def class_conditional_alignment_loss(
    features: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor] | torch.Tensor,
    label_mask: Mapping[str, torch.Tensor] | torch.Tensor,
    center_id: torch.Tensor | None,
    *,
    tasks: Sequence[str] = TASKS,
    specs: Mapping[str, TaskSpec] | None = None,
    min_group_samples: int = 2,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Align centre centroids only within a class, preserving task signal."""

    definitions = dict(specs or task_specs(tasks))
    if center_id is None:
        return None, {}
    values: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}
    for task in tasks:
        spec = definitions.get(task)
        if spec is None or spec.kind == "survival":
            continue
        feature = features.get(task)
        if feature is None:
            continue
        target = task_value(labels, task, feature.device)
        mask = task_value(label_mask, task, feature.device)
        if target is None or mask is None:
            continue
        domains = center_id.to(feature.device).reshape(-1).long()
        target = target.reshape(-1).long()
        valid = mask.reshape(-1) > 0
        task_terms: list[torch.Tensor] = []
        for label in torch.unique(target[valid]):
            class_selected = valid & (target == label)
            centers: list[torch.Tensor] = []
            for domain in torch.unique(domains[class_selected]):
                selected = class_selected & (domains == domain)
                if int(selected.sum()) >= int(min_group_samples):
                    centers.append(F.normalize(feature[selected].float().mean(dim=0), dim=0))
            if len(centers) >= 2:
                matrix = torch.stack(centers)
                task_terms.append((matrix - matrix.mean(dim=0)).square().mean())
        if task_terms:
            value = torch.stack(task_terms).mean()
            values.append(value)
            details[f"alignment/{task}"] = value
    return (torch.stack(values).mean() if values else None), details


def orthogonality_loss(
    phenotype: torch.Tensor | Mapping[str, torch.Tensor] | None,
    style: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    if phenotype is None or style is None:
        return None
    if isinstance(phenotype, Mapping):
        pairs = [
            (value, style.get(task) if isinstance(style, Mapping) else style)
            for task, value in phenotype.items()
        ]
    else:
        pairs = [(phenotype, style if torch.is_tensor(style) else None)]
    terms = []
    for clinical, nuisance in pairs:
        if clinical is None or nuisance is None:
            continue
        clinical = F.normalize(clinical.float(), dim=-1)
        nuisance = F.normalize(nuisance.detach().float(), dim=-1)
        terms.append((clinical * nuisance).sum(dim=-1).square().mean())
    return torch.stack(terms).mean() if terms else None


__all__ = [
    "class_conditional_alignment_loss",
    "class_conditional_domain_loss",
    "cross_center_supervised_contrastive_loss",
    "domain_classification_loss",
    "group_dro_loss",
    "orthogonality_loss",
]
