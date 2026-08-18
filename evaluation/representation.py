from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_phase_feature_analysis(
    *,
    root: Path,
    prediction_rows: list[dict[str, Any]],
    feature_sets: dict[str, np.ndarray],
    tasks: tuple[str, ...] = ("pCR", "HER2"),
    task_kinds: dict[str, str] | None = None,
    figure_formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
) -> dict[str, Any]:
    feature_root = root / "features"
    figure_root = root / "figures" / "phase_representations"
    metric_root = root / "metrics"
    feature_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for task in tasks:
        matrix = feature_sets.get(f"task_feature_{task}")
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            continue
        if matrix.shape[0] != len(prediction_rows):
            report[task] = {
                "available": False,
                "reason": f"row mismatch: features={matrix.shape[0]} rows={len(prediction_rows)}",
            }
            continue
        np.savez_compressed(
            feature_root / f"task_feature_{task}.npz",
            features=matrix.astype(np.float32, copy=False),
            feature_names=np.asarray(
                [f"{task}_feature_{index:04d}" for index in range(matrix.shape[1])],
                dtype=object,
            ),
        )
        kind = (task_kinds or {}).get(task, "binary")
        task_report, coordinates = _representation_probe(
            matrix, prediction_rows, task, kind
        )
        report[task] = task_report
        if coordinates is not None:
            _plot_embedding(
                coordinates,
                prediction_rows,
                task=task,
                kind=kind,
                output_stem=figure_root / f"{task}_feature_by_label",
                color_by="label",
                formats=figure_formats,
                dpi=dpi,
            )
            _plot_embedding(
                coordinates,
                prediction_rows,
                task=task,
                kind=kind,
                output_stem=figure_root / f"{task}_feature_by_center",
                color_by="center",
                formats=figure_formats,
                dpi=dpi,
            )
    (metric_root / "phase_representation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _representation_probe(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    task: str,
    kind: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover
        return {"available": False, "reason": str(exc)}, None
    finite_rows: list[int] = []
    target: list[int] = []
    for index, row in enumerate(rows):
        value = row.get(f"{task}_label")
        try:
            label = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(label) and np.isfinite(matrix[index]).all():
            if kind == "survival":
                event = row.get(f"{task}_event")
                if event in (None, ""):
                    continue
                target.append(int(event))
            else:
                target.append(int(label) if kind == "multiclass" else int(label > 0.5))
            finite_rows.append(index)
    if len(finite_rows) < 10 or len(set(target)) < 2:
        return {"available": False, "reason": "insufficient labelled classes"}, None
    selected = matrix[np.asarray(finite_rows)]
    target_array = np.asarray(target)
    selected_rows = [rows[index] for index in finite_rows]
    center_names = [str(row.get("dataset_id", "unknown")) for row in selected_rows]
    centers = LabelEncoder().fit_transform(center_names)
    components = min(20, selected.shape[0] - 1, selected.shape[1])
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=2026),
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"),
    )
    _, class_counts = np.unique(target_array, return_counts=True)
    center_counts = np.bincount(centers)
    task_folds = min(5, int(class_counts.min()))
    center_folds = min(5, int(center_counts.min()))
    if task_folds < 2 or center_folds < 2:
        return {"available": False, "reason": "insufficient samples per probe class"}, None
    task_scoring = "roc_auc" if len(set(target)) == 2 else "balanced_accuracy"
    task_scores = cross_val_score(
        model,
        selected,
        target_array,
        cv=StratifiedKFold(task_folds, shuffle=True, random_state=2026),
        scoring=task_scoring,
        n_jobs=1,
    )
    center_scores = cross_val_score(
        model,
        selected,
        centers,
        cv=StratifiedKFold(center_folds, shuffle=True, random_state=2026),
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    standardized = StandardScaler().fit_transform(matrix)
    coordinates = PCA(n_components=2, random_state=2026).fit_transform(standardized)
    return (
        {
            "available": True,
            "n": int(selected.shape[0]),
            "feature_dim": int(selected.shape[1]),
            "task_probe_metric": (
                "event_roc_auc" if kind == "survival" else task_scoring
            ),
            "task_probe_score_mean": float(task_scores.mean()),
            "task_probe_score_std": float(task_scores.std()),
            **(
                {
                    "task_probe_roc_auc_mean": float(task_scores.mean()),
                    "task_probe_roc_auc_std": float(task_scores.std()),
                }
                if task_scoring == "roc_auc"
                else {}
            ),
            "center_probe_balanced_accuracy_mean": float(center_scores.mean()),
            "center_probe_balanced_accuracy_std": float(center_scores.std()),
            "center_balanced_chance": float(1.0 / len(center_counts)),
            "center_counts": center_counts.tolist(),
        },
        coordinates,
    )


def _plot_embedding(
    coordinates: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    task: str,
    kind: str,
    output_stem: Path,
    color_by: str,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if color_by == "center":
        groups = np.asarray([str(row.get("dataset_id", "unknown")) for row in rows])
        title = f"{task} PHASE representation by centre"
    else:
        groups = []
        for row in rows:
            value = row.get(f"{task}_label")
            if value in (None, ""):
                groups.append("missing")
            elif kind == "multiclass":
                groups.append(str(row.get(f"{task}_label_name", int(float(value)))))
            elif kind == "survival":
                groups.append("event" if int(row[f"{task}_event"]) else "censored")
            else:
                groups.append("positive" if float(value) > 0.5 else "negative")
        groups = np.asarray(groups)
        title = f"{task} PHASE representation by class"
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for group in sorted(set(groups)):
        selected = groups == group
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=20,
            alpha=0.72,
            label=f"{group} (n={int(selected.sum())})",
        )
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
