from __future__ import annotations

import torch
import torch.nn.functional as F


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg without batch-normalization or a minimum global batch size."""

    first = first.float()
    second = second.float()
    invariance = F.mse_loss(first, second)
    if first.shape[0] < 2:
        variance = invariance.new_zeros(())
        covariance = invariance.new_zeros(())
    else:
        first_centered = first - first.mean(dim=0)
        second_centered = second - second.mean(dim=0)
        first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
        second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-4)
        variance = 0.5 * (
            F.relu(float(variance_target) - first_std).mean()
            + F.relu(float(variance_target) - second_std).mean()
        )
        denominator = max(first.shape[0] - 1, 1)
        first_cov = first_centered.T @ first_centered / denominator
        second_cov = second_centered.T @ second_centered / denominator
        dimension = max(first.shape[1], 1)
        eye = torch.eye(dimension, device=first.device, dtype=torch.bool)
        covariance = 0.5 * (
            first_cov.masked_fill(eye, 0).square().sum() / dimension
            + second_cov.masked_fill(eye, 0).square().sum() / dimension
        )
    total = (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )
    return total, {
        "pretrain/vicreg_invariance": invariance,
        "pretrain/vicreg_variance": variance,
        "pretrain/vicreg_covariance": covariance,
        "pretrain/vicreg": total,
    }


def masked_voxel_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    voxel_mask: torch.Tensor,
    *,
    item_mask: torch.Tensor | None = None,
    anatomical_mask: torch.Tensor | None = None,
    outside_anatomy_weight: float = 0.10,
) -> torch.Tensor:
    """Reconstruct original intensities only at masked voxel locations."""

    selected = voxel_mask.bool()
    if item_mask is not None:
        expanded = item_mask.bool()
        while expanded.ndim < selected.ndim:
            expanded = expanded.unsqueeze(-1)
        selected = selected & expanded
    # Do not use ``prediction[selected]`` here.  Advanced boolean indexing of
    # a full [batch, phase, depth, height, width] reconstruction has caused
    # illegal CUDA memory accesses on DataParallel replicas in this runtime.
    # Masked reduction uses only broadcastable elementwise kernels and keeps
    # the same Smooth-L1 objective.
    selected_weights = selected.to(dtype=torch.float32)
    if anatomical_mask is not None:
        anatomy = anatomical_mask.bool()
        while anatomy.ndim < selected.ndim:
            anatomy = anatomy.unsqueeze(1)
        anatomical_weight = torch.where(
            anatomy,
            selected_weights.new_ones(()),
            selected_weights.new_full((), float(outside_anatomy_weight)),
        )
        selected_weights = selected_weights * anatomical_weight
    error = F.smooth_l1_loss(
        prediction.float(), target.detach().float(), reduction="none", beta=0.5
    )
    return (error * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


def pharmacokinetic_fitting_loss(
    outputs: dict[str, torch.Tensor],
    observed_curve: torch.Tensor,
    phase_mask: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signal fitting and differential-equation residual for the PINN."""

    valid = phase_mask.bool()
    if valid.shape[1] > 0:
        valid = valid.clone()
        valid[:, 0] = False
    point_error = F.smooth_l1_loss(
        outputs["fitted_curve"].float(),
        observed_curve.detach().float(),
        reduction="none",
        beta=0.1,
    )
    per_case = (point_error * valid.float()).sum(dim=1) / valid.float().sum(
        dim=1
    ).clamp_min(1.0)
    confidence = confidence.float().reshape(-1).clamp(0.0, 1.0)
    curve = (per_case * confidence).sum() / confidence.sum().clamp_min(1e-6)
    ode_mask = outputs["ode_mask"].bool()
    ode_error = outputs["ode_residual"].float().square()
    ode_per_case = (ode_error * ode_mask.float()).sum(dim=1) / ode_mask.float().sum(
        dim=1
    ).clamp_min(1.0)
    ode = (ode_per_case * confidence).sum() / confidence.sum().clamp_min(1e-6)
    return curve, ode


def phase_order_loss(logits: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
    valid = phase_mask.bool()
    if not bool(valid.any()):
        return logits.float().sum() * 0.0
    order = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    order = order.expand(logits.shape[0], -1)
    return F.cross_entropy(logits[valid].float(), order[valid].long())


__all__ = [
    "masked_voxel_reconstruction_loss",
    "pharmacokinetic_fitting_loss",
    "phase_order_loss",
    "vicreg_loss",
]
