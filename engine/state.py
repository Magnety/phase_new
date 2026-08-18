from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..evaluation.analysis import (
    export_phase_feature_analysis,
    export_prediction_figures,
    export_training_history,
)
from ..data import (
    TASKS,
    CenterBalancedSampler,
    MultiTaskGroupSampler,
    PHASEDataset,
    collate_phase,
    read_manifest_samples,
    stratified_patient_split,
)
from ..objectives import (
    class_conditional_alignment_loss,
    class_conditional_domain_loss,
    cross_center_supervised_contrastive_loss,
    domain_classification_loss,
    group_dro_loss,
    masked_multitask_bce,
    orthogonality_loss,
    pairwise_ranking_loss,
    phase_order_loss,
    phenotype_compactness_loss,
    vicreg_loss,
)
from ..models import PHASEModel
from ..evaluation.visualization import export_model_explanations
from ..evaluation.metrics import best_threshold as _best_threshold
from ..evaluation.metrics import binary_metrics as _binary_metrics
from .checkpoint import (
    atomic_torch_save,
    atomic_write_json,
    json_ready as _json_ready,
    strip_module_prefix as _strip_module_prefix,
)


class CheckpointMixin:
    CHECKPOINT_FORMAT = "PHASE-independent-v5-spatiotemporal-transformer"

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        *,
        kind: str,
        metrics: Mapping[str, Any],
    ) -> None:
        if not self.is_primary:
            return
        atomic_torch_save(
            path,
            {
                "format": self.CHECKPOINT_FORMAT,
                "kind": kind,
                "epoch": int(epoch),
                "model": self._unwrapped_model().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "center_to_id": self.center_to_id,
                "task_thresholds": self.thresholds,
                "probability_calibration": self.probability_calibration,
                "metrics": _json_ready(metrics),
                "config": self.config,
            },
        )

    def _load_model_weights(
        self,
        path: Path,
        *,
        compatible: bool,
        require_kind: str | None,
    ) -> Mapping[str, Any]:
        path = path.expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if require_kind and payload.get("kind") != require_kind:
            raise ValueError(f"Expected a PHASE {require_kind} checkpoint, got {payload.get('kind')}: {path}")
        if require_kind and payload.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError(
                "This model requires a PHASE v5 spatiotemporal Transformer "
                f"checkpoint, got {payload.get('format')!r}: {path}. Re-run pretraining."
            )
        state = _strip_module_prefix(payload.get("model", payload))
        model = self._unwrapped_model()
        if compatible:
            current = model.state_dict()
            state = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
            critical_prefixes = (
                "spatial_encoder.",
                "patch_temporal_encoder.",
                "phase_embedding.",
                "temporal_encoder.",
                "phenotype_fusion.",
                "auxiliary_encoders.",
                "auxiliary_projections.",
                "modality_moe.",
                "pinn.",
            )
            missing_critical = [
                prefix
                for prefix in critical_prefixes
                if any(key.startswith(prefix) for key in current)
                and not any(key.startswith(prefix) for key in state)
            ]
            if missing_critical:
                raise ValueError(
                    "Pretraining checkpoint lacks required PHASE v5 branches: "
                    + ", ".join(missing_critical)
                )
            missing, unexpected = model.load_state_dict(state, strict=False)
            if self.is_primary:
                atomic_write_json(self.output_root / "compatible_load_report.json", {"checkpoint": str(path), "loaded": len(state), "missing": missing, "unexpected": unexpected})
        else:
            model.load_state_dict(state, strict=True)
        return payload

    def _resume_if_requested(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        *,
        expected_kind: str,
    ) -> int:
        resume = self._stage_cfg().get("resume_path")
        if not resume:
            return 1
        payload = self._load_model_weights(Path(resume), compatible=False, require_kind=expected_kind)
        self._resume_payload = payload
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        return int(payload.get("epoch", 0)) + 1

    def _existing_history(self, *, before_epoch: int) -> list[dict[str, Any]]:
        path = self.output_root / "history.json"
        if not path.exists():
            return []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict) and int(row.get("epoch", 0)) < before_epoch]

    def _log_epoch(self, record: Mapping[str, Any]) -> None:
        if not self.is_primary:
            return
        epoch = record.get("epoch")
        train_loss = record.get("train", {}).get("total")
        performance = record.get("train", {}).get("performance", {}) or {}
        validation = record.get("val", {})
        val_value = validation.get("total", validation.get("checkpoint_selection", {}).get("robust_score"))
        timing = ""
        if performance:
            timing = (
                f" cases/s={float(performance.get('cases_per_second', 0.0)):.2f}"
                f" data_wait={float(performance.get('data_wait_ms', 0.0)):.1f}ms"
                f" collate={float(performance.get('collate_ms', 0.0)):.1f}ms"
                f" h2d={float(performance.get('h2d_ms', 0.0)):.1f}ms"
                f" forward={float(performance.get('forward_ms', 0.0)):.1f}ms"
                f" backward={float(performance.get('backward_ms', 0.0)):.1f}ms"
                f" optimizer={float(performance.get('optimizer_ms', 0.0)):.1f}ms"
            )
        print(
            f"[{self.mode}] epoch={epoch} train={train_loss!s} "
            f"val={val_value!s}{timing}",
            flush=True,
        )
