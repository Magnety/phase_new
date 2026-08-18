"""CPU-side pretraining diagnostics for PHASE.

These exports are intentionally detached from the training graph.  They give
pretraining the same auditability as downstream evaluation without delaying
GPU work on every batch.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def export_pretraining_history(
    *,
    root: Path,
    history: list[Mapping[str, Any]],
    figure_formats: tuple[str, ...] = ("png",),
    dpi: int = 200,
) -> dict[str, Any]:
    """Export loss tables and train/validation trajectories for every term."""
    root.mkdir(parents=True, exist_ok=True)
    metric_root = root / "metrics"
    figure_root = root / "figures" / "pretraining"
    metric_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    keys = sorted(
        {
            key
            for row in history
            for split in ("train", "val")
            for key, value in dict(row.get(split, {}) or {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    with (metric_root / "pretraining_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "lr", *[f"train/{key}" for key in keys], *[f"val/{key}" for key in keys]]
        )
        writer.writeheader()
        for row in history:
            record: dict[str, Any] = {"epoch": row.get("epoch"), "lr": row.get("lr")}
            for split in ("train", "val"):
                for key in keys:
                    record[f"{split}/{key}"] = dict(row.get(split, {}) or {}).get(key)
            writer.writerow(record)
    summary = {
        "epochs": len(history),
        "available_metrics": keys,
        "best_validation_total": min(
            (float(row.get("val", {}).get("total")) for row in history if row.get("val", {}).get("total") is not None),
            default=None,
        ),
    }
    (metric_root / "pretraining_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not history or not keys:
        return summary
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {
        "objective": ["total", "pretrain/vicreg_invariance", "pretrain/vicreg_variance", "pretrain/vicreg_covariance"],
        "reconstruction_and_kinetics": ["pretrain/dce_voxel_mae", "pretrain/modality_voxel_mae", "pretrain/pinn_curve", "pretrain/pinn_ode", "pretrain/order"],
        "invariance_and_routing": ["pretrain/domain_adversarial", "pretrain/style_domain", "pretrain/orthogonality", "pretrain/modality_moe_balance", "pretrain/modality_moe_entropy"],
    }
    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    for name, candidates in groups.items():
        selected = [key for key in candidates if key in keys]
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(10.5, 4.8))
        for split, linestyle in (("train", "-"), ("val", "--")):
            for key in selected:
                values = [dict(row.get(split, {}) or {}).get(key) for row in history]
                if any(value is not None for value in values):
                    axis.plot(epochs, values, linestyle=linestyle, linewidth=1.5, label=f"{split}: {key}")
        axis.set(xlabel="Epoch", ylabel="Loss", title=f"PHASE pretraining: {name.replace('_', ' ')}")
        axis.legend(frameon=False, fontsize=7, ncol=2)
        figure.tight_layout()
        _save(figure, figure_root / name, figure_formats, dpi)
    return summary


def export_pretraining_validation_artifacts(
    *,
    root: Path,
    artifacts: Mapping[str, Any],
    figure_formats: tuple[str, ...] = ("png",),
    dpi: int = 200,
) -> dict[str, Any]:
    """Plot PINN curves, phenotype embedding and multimodal-MoE utilization."""
    root.mkdir(parents=True, exist_ok=True)
    metric_root = root / "metrics"
    figure_root = root / "figures" / "pretraining"
    metric_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    curves = np.asarray(artifacts.get("observed_curve", np.empty((0, 0))), dtype=np.float32)
    fitted = np.asarray(artifacts.get("fitted_curve", np.empty((0, 0))), dtype=np.float32)
    aif = np.asarray(artifacts.get("aif_curve", np.empty((0, 0))), dtype=np.float32)
    pinn_confidence = np.asarray(
        artifacts.get("pinn_confidence", np.empty((0,))), dtype=np.float32
    ).reshape(-1)
    pinn_dynamic_valid = np.asarray(
        artifacts.get("pinn_dynamic_valid", np.empty((0,))), dtype=bool
    ).reshape(-1)
    times = np.asarray(artifacts.get("phase_times", np.empty((0, 0))), dtype=np.float32)
    phase_positions = np.asarray(
        artifacts.get("phase_positions", np.empty((0, 0))), dtype=np.float32
    )
    mask = np.asarray(artifacts.get("phase_mask", np.empty((0, 0))), dtype=bool)
    phenotype = np.asarray(artifacts.get("phenotype", np.empty((0, 0))), dtype=np.float32)
    routing = np.asarray(artifacts.get("modality_routing", np.empty((0, 0))), dtype=np.float32)
    centers = np.asarray(artifacts.get("center", []), dtype=object)
    patient_ids = np.asarray(artifacts.get("patient_id", []), dtype=object)
    sample_ids = np.asarray(artifacts.get("sample_id", []), dtype=object)
    visits = np.asarray(artifacts.get("visit", []), dtype=object)
    parameters = np.asarray(artifacts.get("pinn_parameters", np.empty((0, 6))), dtype=np.float32)
    mae_target = np.asarray(artifacts.get("mae_target_slice", np.empty((0, 0, 0, 0))), dtype=np.float32)
    mae_reconstruction = np.asarray(artifacts.get("mae_reconstruction_slice", np.empty((0, 0, 0, 0))), dtype=np.float32)
    mae_mask = np.asarray(artifacts.get("mae_mask_slice", np.empty((0, 0, 0, 0))), dtype=bool)
    auxiliary_mae_target = np.asarray(
        artifacts.get("auxiliary_mae_target_slice", np.empty((0, 0, 0, 0))),
        dtype=np.float32,
    )
    auxiliary_mae_reconstruction = np.asarray(
        artifacts.get("auxiliary_mae_reconstruction_slice", np.empty((0, 0, 0, 0))),
        dtype=np.float32,
    )
    auxiliary_mae_mask = np.asarray(
        artifacts.get("auxiliary_mae_mask_slice", np.empty((0, 0, 0, 0))),
        dtype=bool,
    )
    auxiliary_modality_mask = np.asarray(
        artifacts.get("auxiliary_modality_mask", np.empty((0, 0))), dtype=bool
    )
    auxiliary_modality_names = np.asarray(
        artifacts.get("auxiliary_modality_names", ("T1", "T2", "DWI", "ADC")),
        dtype=object,
    )
    phase_logits = np.asarray(artifacts.get("phase_order_logits", np.empty((0, 0, 0))), dtype=np.float32)
    np.savez_compressed(
        metric_root / "pretraining_validation_artifacts.npz",
        observed_curve=curves, fitted_curve=fitted, aif_curve=aif,
        pinn_confidence=pinn_confidence, pinn_dynamic_valid=pinn_dynamic_valid,
        phase_times=times,
        phase_positions=phase_positions, phase_mask=mask, phenotype=phenotype,
        modality_routing=routing, centers=centers, patient_ids=patient_ids,
        sample_ids=sample_ids, visits=visits, pinn_parameters=parameters,
        mae_target_slice=mae_target,
        mae_reconstruction_slice=mae_reconstruction, mae_mask_slice=mae_mask,
        auxiliary_mae_target_slice=auxiliary_mae_target,
        auxiliary_mae_reconstruction_slice=auxiliary_mae_reconstruction,
        auxiliary_mae_mask_slice=auxiliary_mae_mask,
        auxiliary_modality_mask=auxiliary_modality_mask,
        auxiliary_modality_names=auxiliary_modality_names,
        phase_order_logits=phase_logits,
    )
    report = {
        "n_cases": int(curves.shape[0]),
        "feature_dim": int(phenotype.shape[1]) if phenotype.ndim == 2 else 0,
        "visualization_samples": {},
    }
    if not curves.size:
        (metric_root / "pretraining_validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dynamic_cases = _dynamic_curve_cases(
        curves, mask, pinn_confidence, dynamic_valid=pinn_dynamic_valid
    )
    report["pinn_dynamic_cases"] = int(len(dynamic_cases))
    report["pinn_excluded_flat_cases"] = int(curves.shape[0] - len(dynamic_cases))
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    for order, index in enumerate(dynamic_cases):
        valid = mask[index] if mask.size else np.ones(curves.shape[1], dtype=bool)
        axis.plot(times[index, valid], curves[index, valid], "o", alpha=0.62, label="observed" if order == 0 else None)
        axis.plot(times[index, valid], fitted[index, valid], "-", alpha=0.70, label="PINN fit" if order == 0 else None)
    if dynamic_cases:
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No dynamically enhancing validation curve", ha="center", va="center", transform=axis.transAxes)
    axis.set(
        xlabel="Time (minutes)", ylabel="Normalized enhancement",
        title=(
            "Validation pharmacokinetic curves "
            f"(dynamic={len(dynamic_cases)}, excluded flat={report['pinn_excluded_flat_cases']})"
        ),
    )
    figure.tight_layout()
    _save(figure, figure_root / "pinn_validation_curves", figure_formats, dpi)
    if parameters.ndim == 2 and parameters.shape[0]:
        pinn_case = _plot_pinn_case_diagnostics(
            times=times,
            observed=curves,
            fitted=fitted,
            aif=aif,
            confidence=pinn_confidence,
            dynamic_valid=pinn_dynamic_valid,
            phase_mask=mask,
            parameters=parameters,
            centers=centers,
            patient_ids=patient_ids,
            sample_ids=sample_ids,
            visits=visits,
            dce_slices=mae_target,
            stem=figure_root / "pinn_case_diagnostics",
            formats=figure_formats,
            dpi=dpi,
        )
        if pinn_case is not None:
            report["visualization_samples"]["pinn"] = _case_metadata(
                pinn_case, centers, patient_ids, sample_ids, visits
            )
    if mae_target.ndim == 4 and mae_target.shape[0]:
        mae_case = _plot_mae_panel(
            target=mae_target,
            reconstruction=mae_reconstruction,
            mask=mae_mask,
            phase_mask=mask,
            phase_times=times,
            auxiliary_target=auxiliary_mae_target,
            auxiliary_reconstruction=auxiliary_mae_reconstruction,
            auxiliary_mask=auxiliary_mae_mask,
            auxiliary_available=auxiliary_modality_mask,
            auxiliary_names=auxiliary_modality_names,
            centers=centers,
            patient_ids=patient_ids,
            sample_ids=sample_ids,
            visits=visits,
            stem=figure_root / "mae_reconstruction_preview",
            formats=figure_formats,
            dpi=dpi,
        )
        if mae_case is not None:
            report["visualization_samples"]["mae"] = _case_metadata(
                mae_case, centers, patient_ids, sample_ids, visits
            )
        phase_order_case = _plot_phase_order_panel(
            volumes=mae_target,
            logits=phase_logits,
            phase_mask=mask,
            phase_times=times,
            centers=centers,
            patient_ids=patient_ids,
            sample_ids=sample_ids,
            visits=visits,
            stem=figure_root / "phase_order_recovery",
            formats=figure_formats,
            dpi=dpi,
        )
        if phase_order_case is not None:
            report["visualization_samples"]["phase_order"] = _case_metadata(
                phase_order_case, centers, patient_ids, sample_ids, visits
            )
    if routing.ndim == 2 and routing.size:
        unique_centers = sorted(set(str(value) for value in centers))
        matrix = np.asarray([routing[np.asarray([str(value) == center for value in centers])].mean(axis=0) for center in unique_centers])
        figure, axis = plt.subplots(figsize=(7.2, max(3.0, 0.55 * len(unique_centers))))
        image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set(yticks=np.arange(len(unique_centers)), yticklabels=unique_centers, xlabel="Input modality", title="Pretraining modality-MoE routing by centre")
        axis.set_xticks(np.arange(matrix.shape[1]), labels=["DCE", "T1", "T2", "DWI", "ADC"][: matrix.shape[1]])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        _save(figure, figure_root / "modality_routing_by_center", figure_formats, dpi)
    if phenotype.ndim == 2 and phenotype.shape[0] >= 3 and phenotype.shape[1] >= 2:
        values = phenotype - phenotype.mean(axis=0, keepdims=True)
        coordinates = np.linalg.svd(values, full_matrices=False)[0][:, :2] * np.linalg.svd(values, full_matrices=False)[1][:2]
        figure, axis = plt.subplots(figsize=(6.4, 5.0))
        for center in sorted(set(str(value) for value in centers)):
            selected = np.asarray([str(value) == center for value in centers])
            axis.scatter(coordinates[selected, 0], coordinates[selected, 1], s=28, alpha=0.75, label=center)
        axis.set(xlabel="PC 1", ylabel="PC 2", title="Pretraining phenotype by centre")
        axis.legend(frameon=False, fontsize=7)
        figure.tight_layout()
        _save(figure, figure_root / "phenotype_by_center", figure_formats, dpi)
    (metric_root / "pretraining_validation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _save(figure: Any, stem: Path, formats: tuple[str, ...], dpi: int) -> None:
    import matplotlib.pyplot as plt

    for extension in formats:
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.quantile(finite, (0.01, 0.99))
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)


def _plot_mae_panel(
    *, target: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray,
    phase_mask: np.ndarray, phase_times: np.ndarray,
    auxiliary_target: np.ndarray, auxiliary_reconstruction: np.ndarray,
    auxiliary_mask: np.ndarray, auxiliary_available: np.ndarray,
    auxiliary_names: np.ndarray, centers: np.ndarray, patient_ids: np.ndarray,
    sample_ids: np.ndarray, visits: np.ndarray, stem: Path,
    formats: tuple[str, ...], dpi: int,
) -> int | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A single randomly chosen case keeps every DCE and auxiliary
    # reconstruction legible while ensuring repeated exports are not anchored
    # to a fixed validation subject.
    candidates = np.flatnonzero(
        phase_mask[: target.shape[0]].astype(bool).sum(axis=1) > 0
    )
    if not candidates.size:
        return None
    case_indices = [int(np.random.default_rng().choice(candidates))]
    rows: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for case in case_indices:
        valid_phases = np.flatnonzero(phase_mask[case].astype(bool))
        for phase in valid_phases[:4]:
            time = phase_times[case, phase]
            rows.append(
                (
                    case,
                    f"DCE p{phase} | t={time:.2f} min",
                    target[case, phase],
                    reconstruction[case, phase],
                    mask[case, phase],
                )
            )
        if (
            auxiliary_target.ndim == 4
            and case < auxiliary_target.shape[0]
            and auxiliary_available.ndim == 2
            and case < auxiliary_available.shape[0]
        ):
            for modality_index, modality in enumerate(auxiliary_names):
                if (
                    modality_index >= auxiliary_target.shape[1]
                    or modality_index >= auxiliary_available.shape[1]
                    or not auxiliary_available[case, modality_index]
                ):
                    continue
                rows.append(
                    (
                        case,
                        str(modality),
                        auxiliary_target[case, modality_index],
                        auxiliary_reconstruction[case, modality_index],
                        auxiliary_mask[case, modality_index],
                    )
                )
    if not rows:
        return None
    figure, axes = plt.subplots(
        len(rows), 6,
        figsize=(15.6, max(7.2, 2.15 * len(rows))),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.25, 1.0, 1.0, 1.0, 1.0, 1.0)},
    )
    axes[0, 0].set_title("Input")
    for column, title in enumerate(("Target", "Masked input", "Reconstruction", "|error|", "Voxel mask")):
        axes[0, column + 1].set_title(title)
    for row, (case, label, original, reconstruction_slice, voxel_mask) in enumerate(rows):
        label_axis = axes[row, 0]
        label_axis.text(
            0.02, 0.5, f"case {case}\n{label}", ha="left", va="center", fontsize=8
        )
        label_axis.axis("off")
        panels = (
            _normalize(original),
            _normalize(np.where(voxel_mask, 0.0, original)),
            _normalize(reconstruction_slice),
            np.abs(reconstruction_slice - original),
            voxel_mask.astype(np.float32),
        )
        for column, panel in enumerate(panels):
            axes[row, column + 1].imshow(panel, cmap="magma" if column == 3 else "gray")
            axes[row, column + 1].axis("off")
    figure.suptitle(
        "Voxel-MAE preview: "
        f"{_case_title(case_indices[0], centers, patient_ids, sample_ids, visits)} "
        "| DCE phases and available auxiliary modalities",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    _save(figure, stem, formats, dpi)
    return case_indices[0]


def _plot_phase_order_panel(
    *, volumes: np.ndarray, logits: np.ndarray, phase_mask: np.ndarray,
    phase_times: np.ndarray, centers: np.ndarray, patient_ids: np.ndarray,
    sample_ids: np.ndarray, visits: np.ndarray, stem: Path,
    formats: tuple[str, ...], dpi: int,
) -> int | None:
    if logits.ndim != 3 or not logits.shape[0]:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = np.flatnonzero(phase_mask[: logits.shape[0]].astype(bool).sum(axis=1) > 0)
    if not candidates.size:
        return None
    case = int(np.random.default_rng().choice(candidates))
    valid = phase_mask[case].astype(bool)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return None
    generator = np.random.default_rng()
    shuffled = generator.permutation(valid_indices)
    predicted = logits[case].argmax(axis=-1)
    predicted_order = shuffled[np.argsort(predicted[shuffled], kind="stable")]
    correct_order = valid_indices[np.argsort(phase_times[case, valid_indices], kind="stable")]
    columns = len(valid_indices)
    figure, axes = plt.subplots(
        3, columns, figsize=(max(10.0, columns * 2.1), 8.5), squeeze=False
    )
    for column, phase in enumerate(shuffled):
        axes[0, column].imshow(_normalize(volumes[case, phase]), cmap="gray")
        axes[0, column].set_title(f"input p{phase}\npred={predicted[phase]}", fontsize=8)
        axes[0, column].axis("off")
    for column, phase in enumerate(predicted_order):
        axes[1, column].imshow(_normalize(volumes[case, phase]), cmap="gray")
        axes[1, column].set_title(
            f"predicted p{phase}\nt={phase_times[case, phase]:.2f} min", fontsize=8
        )
        axes[1, column].axis("off")
    for column, phase in enumerate(correct_order):
        axes[2, column].imshow(_normalize(volumes[case, phase]), cmap="gray")
        axes[2, column].set_title(
            f"correct p{phase}\nt={phase_times[case, phase]:.2f} min", fontsize=8
        )
        axes[2, column].axis("off")
    figure.text(0.01, 0.73, "Randomized input", rotation=90, va="center")
    figure.text(0.01, 0.48, "Predicted order", rotation=90, va="center")
    figure.text(0.01, 0.22, "Correct chronological order", rotation=90, va="center")
    accuracy = float(np.mean(predicted[valid_indices] == valid_indices))
    figure.suptitle(
        "DCE phase-order recovery: "
        f"{_case_title(case, centers, patient_ids, sample_ids, visits)} "
        f"(per-phase accuracy={accuracy:.3f})",
        y=0.995,
    )
    figure.tight_layout(rect=(0.035, 0, 1, 0.96))
    _save(figure, stem, formats, dpi)
    return case


def _plot_pinn_case_diagnostics(
    *, times: np.ndarray, observed: np.ndarray, fitted: np.ndarray,
    aif: np.ndarray, confidence: np.ndarray, dynamic_valid: np.ndarray,
    phase_mask: np.ndarray,
    parameters: np.ndarray, centers: np.ndarray, patient_ids: np.ndarray,
    sample_ids: np.ndarray, visits: np.ndarray,
    dce_slices: np.ndarray,
    stem: Path, formats: tuple[str, ...], dpi: int,
) -> int | None:
    """Export a refine-style case-level PINN hemodynamic audit.

    The all-case overlay is useful for a cohort-level sanity check, while this
    panel shows the actual population AIF and makes a fit traceable to its
    phase samples, residuals and predicted kinetic parameters.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = _select_random_pinn_case(
        observed,
        fitted,
        phase_mask,
        confidence,
        dynamic_valid=dynamic_valid,
        times=times,
    )
    if case is None:
        return None
    valid = phase_mask[case].astype(bool) if phase_mask.size else np.ones(observed.shape[1], dtype=bool)
    valid &= np.isfinite(times[case]) & np.isfinite(observed[case]) & np.isfinite(fitted[case])
    x = times[case, valid]
    y = observed[case, valid]
    prediction = fitted[case, valid]
    phase_ids = np.flatnonzero(valid)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    prediction = prediction[order]
    phase_ids = phase_ids[order]
    residual = prediction - y
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    figure = plt.figure(figsize=(12.0, 11.6))
    grid = figure.add_gridspec(
        4, 2, height_ratios=(1.0, 1.0, 0.13, 0.78), hspace=0.36
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        ]
    )

    axes[0, 0].plot(x, y, marker="o", linewidth=2.2, label="Observed")
    axes[0, 0].plot(x, prediction, marker="s", linewidth=2.2, label="PINN fit")
    for x_value, y_value, phase_id in zip(x, y, phase_ids):
        axes[0, 0].annotate(
            f"p{phase_id}", (x_value, y_value), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8,
        )
    axes[0, 0].set(
        xlabel="Time (minutes)", ylabel="Normalized enhancement",
        title="DCE enhancement curve",
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[0, 1].plot(x, residual, marker="d", linewidth=2.0, color="#b04a3a")
    axes[0, 1].fill_between(x, 0.0, residual, color="#e8b8ab", alpha=0.45)
    axes[0, 1].set(
        xlabel="Time (minutes)", ylabel="PINN fit - observed",
        title=f"Pointwise residual (RMSE={rmse:.4f})",
    )
    axes[0, 1].grid(alpha=0.25)

    if aif.ndim == 2 and case < aif.shape[0] and aif.shape[1] == observed.shape[1]:
        aif_values = aif[case, valid][order]
        axes[1, 0].plot(x, aif_values, marker="^", linewidth=2.0, color="#2f6c8f")
        axes[1, 0].fill_between(x, 0.0, aif_values, color="#b8d4e5", alpha=0.35)
        axes[1, 0].set(
            xlabel="Time (minutes)", ylabel="AIF",
            title="Population AIF used by PINN",
        )
        axes[1, 0].grid(alpha=0.25)
    else:
        axes[1, 0].text(0.5, 0.5, "No AIF curve available", ha="center", va="center")
        axes[1, 0].axis("off")

    parameter_names = (
        "Ktrans (min^-1)", "ve (fraction)", "vp (fraction)",
        "kep (min^-1)", "BAT (min)", "curve scale",
    )[: parameters.shape[1]]
    parameter_values = parameters[case, : len(parameter_names)]
    positions = np.arange(len(parameter_names))
    bars = axes[1, 1].barh(positions, parameter_values, color="#4d8f6d")
    axes[1, 1].set(
        yticks=positions, yticklabels=parameter_names, xlabel="Predicted value",
        title="Hemodynamic parameters",
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="x", alpha=0.25)
    axes[1, 1].bar_label(bars, fmt="%.3g", padding=3, fontsize=8)

    strip_label = figure.add_subplot(grid[2, :])
    strip_label.text(
        0.5, 0.45, "DCE phases aligned with the curve time points",
        ha="center", va="center", fontsize=10,
    )
    strip_label.axis("off")
    image_grid = grid[3, :].subgridspec(1, len(phase_ids), wspace=0.04)
    for column, (phase_id, time) in enumerate(zip(phase_ids, x)):
        axis = figure.add_subplot(image_grid[0, column])
        if (
            dce_slices.ndim == 4
            and case < dce_slices.shape[0]
            and phase_id < dce_slices.shape[1]
        ):
            axis.imshow(_normalize(dce_slices[case, phase_id]), cmap="gray")
            axis.set_title(f"p{phase_id}\nt={time:.2f} min", fontsize=8)
        else:
            axis.text(0.5, 0.5, f"p{phase_id}", ha="center", va="center")
        axis.axis("off")

    center = str(centers[case]) if case < centers.size else "unknown"
    sample = _case_label(case, sample_ids, patient_ids)
    patient = str(patient_ids[case]) if case < patient_ids.size else "unknown"
    visit = str(visits[case]) if case < visits.size and str(visits[case]) else "unknown"
    confidence_value = float(confidence[case]) if case < confidence.size else float("nan")
    confidence_text = f"{confidence_value:.3f}" if np.isfinite(confidence_value) else "n/a"
    figure.suptitle("PINN hemodynamic fitting preview", fontsize=13, y=0.985)
    figure.text(
        0.5, 0.952,
        f"sample={sample} | patient={patient} | center={center} | visit={visit} | "
        f"valid phases={int(valid.sum())} | RMSE={rmse:.4f} | confidence={confidence_text}",
        ha="center", va="top", fontsize=9,
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.05, top=0.89)
    _save(figure, stem, formats, dpi)
    return case


def _select_representative_pinn_case(
    observed: np.ndarray,
    fitted: np.ndarray,
    phase_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    times: np.ndarray | None = None,
) -> int | None:
    """Choose the median-RMSE dynamically enhancing case for compatibility."""
    candidates: list[tuple[int, float, float]] = []
    for case in range(observed.shape[0]):
        valid = (
            phase_mask[case].astype(bool)
            if phase_mask.size
            else np.ones(observed.shape[1], dtype=bool)
        )
        valid &= np.isfinite(observed[case]) & np.isfinite(fitted[case])
        if times is not None:
            valid &= np.isfinite(times[case])
        if int(valid.sum()) < 3:
            continue
        if float(np.ptp(observed[case, valid])) <= 1e-4:
            continue
        rmse = float(np.sqrt(np.mean(np.square(fitted[case, valid] - observed[case, valid]))))
        case_confidence = float(confidence[case]) if case < confidence.size else float("nan")
        candidates.append((case, rmse, case_confidence))
    if not candidates:
        return None
    trusted = [item for item in candidates if np.isfinite(item[2]) and item[2] > 0.0]
    ranked = sorted(trusted or candidates, key=lambda item: item[1])
    return int(ranked[(len(ranked) - 1) // 2][0])


def _dynamic_curve_cases(
    observed: np.ndarray,
    phase_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    dynamic_valid: np.ndarray | None = None,
) -> list[int]:
    """Return curves that are both usable and non-flat after masking."""
    cases: list[int] = []
    for case in range(observed.shape[0]):
        valid = (
            phase_mask[case].astype(bool)
            if phase_mask.size
            else np.ones(observed.shape[1], dtype=bool)
        )
        if int(valid.sum()) < 3 or not np.isfinite(observed[case, valid]).all():
            continue
        declared_valid = (
            bool(dynamic_valid[case]) if dynamic_valid is not None and case < dynamic_valid.size else True
        )
        case_confidence = float(confidence[case]) if case < confidence.size else 1.0
        if (
            declared_valid
            and np.isfinite(case_confidence)
            and case_confidence > 0.0
            and float(np.ptp(observed[case, valid])) > 1e-4
        ):
            cases.append(case)
    return cases


def _select_random_pinn_case(
    observed: np.ndarray,
    fitted: np.ndarray,
    phase_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    dynamic_valid: np.ndarray | None = None,
    times: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> int | None:
    """Draw one traceable, non-flat PINN curve for a case-level preview."""
    candidates = _dynamic_curve_cases(
        observed, phase_mask, confidence, dynamic_valid=dynamic_valid
    )
    eligible: list[int] = []
    for case in candidates:
        valid = (
            phase_mask[case].astype(bool)
            if phase_mask.size
            else np.ones(observed.shape[1], dtype=bool)
        )
        valid &= np.isfinite(fitted[case])
        if times is not None:
            valid &= np.isfinite(times[case])
        if int(valid.sum()) >= 3:
            eligible.append(case)
    if not eligible:
        return None
    return int((rng or np.random.default_rng()).choice(eligible))


def _case_metadata(
    case: int,
    centers: np.ndarray,
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
    visits: np.ndarray,
) -> dict[str, Any]:
    return {
        "case_index": int(case),
        "center": str(centers[case]) if case < centers.size else "unknown",
        "patient_id": str(patient_ids[case]) if case < patient_ids.size else "unknown",
        "sample_id": _case_label(case, sample_ids, patient_ids),
        "visit": str(visits[case]) if case < visits.size else "unknown",
    }


def _case_title(
    case: int,
    centers: np.ndarray,
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
    visits: np.ndarray,
) -> str:
    metadata = _case_metadata(case, centers, patient_ids, sample_ids, visits)
    return (
        f"sample={metadata['sample_id']} | patient={metadata['patient_id']} | "
        f"center={metadata['center']} | visit={metadata['visit']}"
    )


def _case_label(case: int, sample_ids: np.ndarray, patient_ids: np.ndarray) -> str:
    for values in (sample_ids, patient_ids):
        if case < values.size and str(values[case]).strip():
            return str(values[case])
    return f"case-{case}"
