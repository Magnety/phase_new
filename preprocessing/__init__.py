"""Standalone medical-image ingestion and preprocessing pipeline."""

from .cli import main
from .config import LoadedVolume, PREPROCESSING_VERSION, PreprocessSettings
from .discovery import discover_dicom_manifest
from .medical_io import load_medical_volume, read_source_groups
from .pipeline import preprocess_group, run_preprocessing

__all__ = [
    "LoadedVolume",
    "PREPROCESSING_VERSION",
    "PreprocessSettings",
    "discover_dicom_manifest",
    "load_medical_volume",
    "main",
    "preprocess_group",
    "read_source_groups",
    "run_preprocessing",
]
