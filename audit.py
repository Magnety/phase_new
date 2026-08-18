"""Backward-compatible CLI and exports for PHASE audit utilities."""

from .evaluation.audit import audit_feature_probe, audit_manifest, audit_predictions, main

__all__ = ["audit_feature_probe", "audit_manifest", "audit_predictions", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
