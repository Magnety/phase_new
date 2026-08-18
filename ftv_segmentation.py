"""Backward-compatible imports for the reorganized segmentation package."""

from .segmentation import (
    ANATOMICAL_FTV_VERSION,
    AnatomicalFTVResult,
    AnatomicalFTVSettings,
    segment_anatomical_ftv,
)

__all__ = [
    "ANATOMICAL_FTV_VERSION",
    "AnatomicalFTVResult",
    "AnatomicalFTVSettings",
    "segment_anatomical_ftv",
]
