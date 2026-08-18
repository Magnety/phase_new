from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PharmacokineticPINN(nn.Module):
    """Fit an extended-Tofts curve from DCE features and acquisition times.

    All constrained parameters are produced through bounded transforms.  The
    analytic forward model is differentiable, so curve fitting during
    self-supervision trains both the parameter estimator and its DCE encoder.
    """

    output_feature_dim = 10

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        ktrans_range: tuple[float, float] = (0.001, 2.5),
        ve_range: tuple[float, float] = (0.02, 1.0),
        vp_range: tuple[float, float] = (0.0, 0.5),
        maximum_bat_minutes: float = 3.0,
    ) -> None:
        super().__init__()
        self.ktrans_range = tuple(float(value) for value in ktrans_range)
        self.ve_range = tuple(float(value) for value in ve_range)
        self.vp_range = tuple(float(value) for value in vp_range)
        self.maximum_bat_minutes = float(maximum_bat_minutes)
        self.parameter_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), 6),
        )
        final = self.parameter_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, std=0.01)
            with torch.no_grad():
                final.bias.copy_(
                    torch.tensor([-2.4, -0.9, -2.2, -2.2, 0.54, 0.54])
                )
        # A positive population AIF is learned globally.  It cannot absorb
        # centre IDs because no acquisition metadata enters this branch.
        self.aif_raw = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    @staticmethod
    def _bounded(
        raw: torch.Tensor, bounds: tuple[float, float]
    ) -> torch.Tensor:
        lower, upper = bounds
        return lower + (upper - lower) * torch.sigmoid(raw)

    def _aif(self, times: torch.Tensor, bat: torch.Tensor) -> torch.Tensor:
        amplitude = F.softplus(self.aif_raw[0]) + 1e-3
        alpha = F.softplus(self.aif_raw[1]) + 0.5
        beta = F.softplus(self.aif_raw[2]) + 0.25
        shifted = (times - bat).clamp_min(0.0)
        positive = times > bat
        values = amplitude * shifted.clamp_min(1e-6).pow(alpha) * torch.exp(
            -beta * shifted
        )
        return torch.where(positive, values, torch.zeros_like(values))

    def forward(
        self,
        feature: torch.Tensor,
        phase_times: torch.Tensor,
        phase_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.parameter_head(feature.float())
        ktrans = self._bounded(raw[:, 0], self.ktrans_range)
        ve = self._bounded(raw[:, 1], self.ve_range)
        vp = self._bounded(raw[:, 2], self.vp_range)
        maximum_time = phase_times.float().amax(dim=1).clamp_min(1e-3)
        bat_limit = torch.minimum(
            maximum_time,
            maximum_time.new_full(maximum_time.shape, self.maximum_bat_minutes),
        )
        bat = torch.sigmoid(raw[:, 3]) * bat_limit
        curve_scale = F.softplus(raw[:, 4]).clamp(0.05, 20.0)
        # The sixth output softly modulates the population AIF amplitude per
        # case without changing its physically positive gamma-variate shape.
        aif_scale = F.softplus(raw[:, 5]).clamp(0.25, 4.0)
        kep = ktrans / ve.clamp_min(1e-4)

        times = phase_times.float()
        valid = phase_mask.bool()
        cp = self._aif(times, bat.unsqueeze(1)) * aif_scale.unsqueeze(1)
        cp = cp * valid.float()
        dt = torch.zeros_like(times)
        dt[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(0.0)
        dt = dt * valid.float()
        time_delta = times.unsqueeze(2) - times.unsqueeze(1)
        causal = time_delta >= 0
        kernel = torch.exp(
            -kep[:, None, None] * time_delta.clamp_min(0.0)
        ) * causal.float()
        integral = (
            kernel * cp.unsqueeze(1) * dt.unsqueeze(1)
        ).sum(dim=2)
        fitted = curve_scale.unsqueeze(1) * (
            vp.unsqueeze(1) * cp + ktrans.unsqueeze(1) * integral
        )
        fitted = fitted * valid.float()

        safe_dt = dt[:, 1:].clamp_min(1e-3)
        dct = (fitted[:, 1:] - fitted[:, :-1]) / safe_dt
        dcp = (cp[:, 1:] - cp[:, :-1]) / safe_dt
        ce = fitted[:, 1:] / curve_scale.unsqueeze(1).clamp_min(1e-4) - vp.unsqueeze(
            1
        ) * cp[:, 1:]
        rhs = curve_scale.unsqueeze(1) * (
            ktrans.unsqueeze(1) * cp[:, 1:]
            - kep.unsqueeze(1) * ce
            + vp.unsqueeze(1) * dcp
        )
        ode_mask = valid[:, 1:] & valid[:, :-1] & (dt[:, 1:] > 0)
        ode_residual = (dct - rhs) * ode_mask.float()

        count = valid.float().sum(dim=1).clamp_min(1.0)
        last_index = (valid.long().sum(dim=1) - 1).clamp_min(0)
        rows = torch.arange(feature.shape[0], device=feature.device)
        early_index = torch.minimum(last_index, torch.ones_like(last_index))
        early = fitted[rows, early_index]
        peak = fitted.masked_fill(~valid, float("-inf")).amax(dim=1)
        peak = torch.nan_to_num(peak, neginf=0.0)
        auc = (fitted * dt).sum(dim=1) / maximum_time.clamp_min(1e-3)
        washout = fitted[rows, last_index] - peak
        parameter_feature = torch.stack(
            [
                ktrans,
                ve,
                vp,
                kep,
                bat,
                curve_scale,
                early,
                peak,
                auc,
                washout,
            ],
            dim=-1,
        )
        return {
            "fitted_curve": fitted,
            "aif": cp,
            "ode_residual": ode_residual,
            "ode_mask": ode_mask,
            "parameters": torch.stack(
                [ktrans, ve, vp, kep, bat, curve_scale], dim=-1
            ),
            "parameter_feature": parameter_feature,
        }


__all__ = ["PharmacokineticPINN"]
