"""Publication-oriented statistical exports for PHASE predictions."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def export_prediction_statistics(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    tasks: Sequence[str],
    task_kinds: Mapping[str, str],
    stage: str,
    analysis: Mapping[str, Any],
    figure_formats: tuple[str, ...],
    dpi: int,
) -> dict[str, Any]:
    """Export bootstrap CIs, calibration, decision curves and subgroups.

    The function only reads frozen prediction rows.  It therefore cannot alter
    threshold selection, probability calibration or any model parameters.
    """
    metric_root = root / "metrics"
    figure_root = root / "figures" / "statistics"
    metric_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    requested_stages = {str(value) for value in analysis.get("bootstrap_stages", ("test", "infer_test", "infer_all"))}
    bootstrap_count = int(analysis.get("bootstrap_samples", 160)) if stage in requested_stages else 0
    report: dict[str, Any] = {"stage": stage, "bootstrap_samples": bootstrap_count, "tasks": {}}
    subgroup_columns = tuple(str(value) for value in analysis.get("subgroup_columns", ("dataset_id", "visit_timepoint")))
    subgroup_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        kind = task_kinds[task]
        metric = dict(metrics.get(task, {}) or {})
        task_rows.append({"task": task, "kind": kind, **{key: value for key, value in metric.items() if not isinstance(value, Mapping)}})
        entry: dict[str, Any] = {"kind": kind, "metrics": metric}
        if kind == "binary":
            labels, probabilities = _binary_arrays(rows, task)
            if labels.size:
                entry["calibration"] = _calibration_and_decision_curves(
                    figure_root, task, labels, probabilities, figure_formats, dpi
                )
                if bootstrap_count > 0 and np.unique(labels).size == 2:
                    entry["bootstrap"] = _bootstrap_binary(labels, probabilities, bootstrap_count)
            for column in subgroup_columns:
                for group, selected in _subgroups(rows, column).items():
                    y, p = _binary_arrays(selected, task)
                    if y.size:
                        subgroup_rows.append(
                            {
                                "task": task, "kind": kind, "column": column,
                                "group": group, "n": int(y.size),
                                **_binary_summary(y, p),
                            }
                        )
        else:
            for column in subgroup_columns:
                for group, selected in _subgroups(rows, column).items():
                    labelled = [row for row in selected if row.get(f"{task}_label") not in (None, "")]
                    if labelled:
                        subgroup_rows.append({"task": task, "kind": kind, "column": column, "group": group, "n": len(labelled)})
        report["tasks"][task] = entry
    _write_csv(metric_root / "task_metric_summary.csv", task_rows)
    _write_csv(metric_root / "subgroup_metrics.csv", subgroup_rows)
    (metric_root / "statistical_analysis.json").write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _binary_arrays(rows: Sequence[Mapping[str, Any]], task: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(row[f"{task}_label"]) > 0.5, float(row[f"{task}_probability"]))
        for row in rows
        if row.get(f"{task}_label") not in (None, "") and row.get(f"{task}_probability") not in (None, "")
    ]
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.asarray([pair[0] for pair in pairs], dtype=np.int64), np.asarray([pair[1] for pair in pairs], dtype=np.float64)


def _binary_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    predicted = probabilities >= 0.5
    accuracy = float(np.mean(predicted == labels))
    result: dict[str, float | None] = {"accuracy": accuracy, "prevalence": float(labels.mean())}
    if np.unique(labels).size == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["pr_auc"] = float(average_precision_score(labels, probabilities))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def _bootstrap_binary(labels: np.ndarray, probabilities: np.ndarray, samples: int) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    generator = np.random.default_rng(2026)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(max(1, samples)):
        indices = generator.integers(0, labels.size, labels.size)
        y, p = labels[indices], probabilities[indices]
        if np.unique(y).size != 2:
            continue
        values["roc_auc"].append(float(roc_auc_score(y, p)))
        values["pr_auc"].append(float(average_precision_score(y, p)))
        values["accuracy"].append(float(np.mean((p >= 0.5) == y)))
    return {
        name: {
            "mean": float(np.mean(items)),
            "ci95": [float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975))],
            "valid_resamples": len(items),
        }
        for name, items in values.items()
        if items
    }


def _calibration_and_decision_curves(
    root: Path, task: str, labels: np.ndarray, probabilities: np.ndarray,
    formats: tuple[str, ...], dpi: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.clip(np.digitize(probabilities, bins) - 1, 0, len(bins) - 2)
    observed, predicted, counts = [], [], []
    for index in range(len(bins) - 1):
        selected = bin_index == index
        if selected.any():
            observed.append(float(labels[selected].mean()))
            predicted.append(float(probabilities[selected].mean()))
            counts.append(int(selected.sum()))
    ece = float(sum(abs(x - y) * n for x, y, n in zip(observed, predicted, counts)) / max(labels.size, 1))
    brier = float(np.mean((probabilities - labels) ** 2))
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    axis.plot([0, 1], [0, 1], "--", color="0.55", label="Ideal")
    axis.plot(predicted, observed, "o-", color="#2563eb", label="PHASE")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"{task} calibration (ECE={ece:.3f}, Brier={brier:.3f})")
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, root / f"{task}_calibration", formats, dpi)
    thresholds = np.linspace(0.01, 0.99, 99)
    prevalence = float(labels.mean())
    net_benefit = []
    for threshold in thresholds:
        positive = probabilities >= threshold
        true_positive = float(np.sum(positive & (labels == 1))) / labels.size
        false_positive = float(np.sum(positive & (labels == 0))) / labels.size
        net_benefit.append(true_positive - false_positive * threshold / (1.0 - threshold))
    treat_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)
    figure, axis = plt.subplots(figsize=(6.0, 4.8))
    axis.plot(thresholds, net_benefit, label="PHASE", color="#dc2626")
    axis.plot(thresholds, treat_all, "--", label="Treat all", color="0.5")
    axis.axhline(0.0, linestyle=":", color="0.5", label="Treat none")
    axis.set(xlabel="Threshold probability", ylabel="Net benefit", title=f"{task} decision curve")
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, root / f"{task}_decision_curve", formats, dpi)
    return {"brier_score": brier, "expected_calibration_error": ece, "calibration_bins": {"mean_predicted": predicted, "observed": observed, "count": counts}}


def _subgroups(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(column, "missing"))].append(row)
    return grouped


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["task", "kind", "column", "group", "n"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save(figure: Any, stem: Path, formats: tuple[str, ...], dpi: int) -> None:
    import matplotlib.pyplot as plt

    for extension in formats:
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
