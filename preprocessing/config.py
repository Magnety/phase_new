from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


PREPROCESSING_VERSION = "PHASE-preprocess-v3-multimodal-voxel-mae"


@dataclass(frozen=True)
class PreprocessSettings:
    source_manifest: str
    output_root: str
    output_manifest: str = "phase_manifest.csv"
    target_shape: tuple[int, int, int] = (48, 96, 96)
    max_phases: int = 8
    modalities: tuple[str, ...] = ("DCE", "T1", "T2", "DWI", "ADC")
    baseline_only: bool = True
    foreground_crop: bool = True
    crop_margin: int = 8
    motion_correction: bool = True
    maximum_translation_voxels: float = 12.0
    normalization_lower_percentile: float = 0.5
    normalization_upper_percentile: float = 99.5
    output_clip: float = 8.0
    workers: int = 4
    overwrite: bool = False
    save_previews: bool = True
    preview_dpi: int = 140
    fail_on_error: bool = False
    dataset_root: str | None = None
    ftv_segmentation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "PreprocessSettings":
        if not value.get("source_manifest") or not value.get("output_root"):
            raise ValueError("preprocessing.source_manifest and output_root are required")
        payload = dict(value)
        payload["target_shape"] = tuple(
            int(item) for item in payload.get("target_shape", (48, 96, 96))
        )
        payload["modalities"] = tuple(
            dict.fromkeys(
                str(item).upper()
                for item in payload.get(
                    "modalities", ("DCE", "T1", "T2", "DWI", "ADC")
                )
            )
        )
        supported = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in payload.items() if key in supported})


@dataclass
class LoadedVolume:
    array: np.ndarray
    spacing_zyx: tuple[float, float, float]
    source_kind: str


__all__ = ["LoadedVolume", "PREPROCESSING_VERSION", "PreprocessSettings"]
