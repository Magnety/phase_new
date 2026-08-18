"""Anatomical safeguards and conservative FTV-proxy segmentation."""

from .ftv import segment_anatomical_ftv
from .types import ANATOMICAL_FTV_VERSION, AnatomicalFTVResult, AnatomicalFTVSettings

__all__ = [
    "ANATOMICAL_FTV_VERSION",
    "AnatomicalFTVResult",
    "AnatomicalFTVSettings",
    "segment_anatomical_ftv",
]
