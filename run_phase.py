"""IDE-friendly PHASE entry point; edit the constants and run this file."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_MODE = "pipeline"  # preprocess | visualize-data | pretrain | finetune | infer | both | pipeline
GPUS = "1,2,3"
CONFIG: str | None = None
CHECKPOINT: str | None = None
OUTPUT_DIR: str | None = None


def main() -> int:
    from src.breast_mri_ai.breast_dce_moe_pinn_phase.main import main as phase_main

    arguments = ["run_phase.py", "--mode", RUN_MODE, "--gpus", GPUS]
    if CONFIG:
        arguments.extend(["--config", CONFIG])
    if CHECKPOINT:
        arguments.extend(["--checkpoint", CHECKPOINT])
    if OUTPUT_DIR:
        arguments.extend(["--output-dir", OUTPUT_DIR])
    previous = sys.argv
    try:
        sys.argv = arguments
        return int(phase_main())
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
