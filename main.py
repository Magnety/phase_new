"""PHASE command-line entry point.

This module deliberately owns argument parsing (rather than delegating to a
``cli`` package) so its invocation contract mirrors the refine framework while
remaining a completely independent implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
# ``python path/to/main.py`` does not establish a package context.  Put the
# repository root on sys.path so the same absolute PHASE imports below work for
# both that invocation and ``python -m src.breast_mri_ai...``.
REPOSITORY_ROOT = PACKAGE_DIR.parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CONFIG_DIR = PACKAGE_DIR / "configs"
DEFAULT_PRETRAIN_CONFIG = CONFIG_DIR / "pretrain_phase.yaml"
DEFAULT_FINETUNE_CONFIG = CONFIG_DIR / 'finetune_phase_primary_endpoints.yaml'#"finetune_phase.yaml"
DEFAULT_INFER_CONFIG = CONFIG_DIR / "infer_phase.yaml"
DEFAULT_SEED = 2026


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(config)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False


def configure_performance(config: dict[str, Any]) -> None:
    import torch

    performance = dict(config.get("performance", {}) or {})
    parallel_mode = str(performance.get("parallel_mode", "single")).lower()
    # DP still performs host-side shard assembly for deferred multimodal
    # batches. One CPU thread leaves that work serialized and exposes it as a
    # regular GPU idle interval between steps. Eight threads are a good
    # default for the large tensor stack/copy operations without allowing
    # unrelated BLAS work to consume the whole host.
    threads = int(performance.get("cpu_num_threads", 8))
    if threads > 0:
        torch.set_num_threads(threads)
    # nn.DataParallel executes replicas concurrently in Python threads. On
    # CUDA 12.8/Blackwell, concurrent cuDNN Conv3d autotuning can poison an
    # execution plan and surface as CUDA error 700 on the following batch.
    # DDP remains free to use benchmark mode because each process owns one GPU.
    torch.backends.cudnn.benchmark = bool(
        performance.get("cudnn_benchmark", True)
        and (
            parallel_mode != "dp"
            or performance.get("dp_cudnn_benchmark", False)
        )
    )
    allow_tf32 = bool(performance.get("allow_tf32", True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse refine-style stage, runtime and split overrides for PHASE."""

    parser = argparse.ArgumentParser(
        description="PHASE phenotype-aligned multimodal foundation model entry point."
    )
    parser.add_argument(
        "--mode",
        default="finetune",
        choices=("preprocess", "visualize-data", "pretrain", "finetune", "infer", "both", "pipeline"),
        help="pretrain | finetune | infer | both, plus standalone preprocessing/QC modes.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Shared YAML override for the selected stage(s).")
    parser.add_argument("--pretrain-config", type=Path, default=DEFAULT_PRETRAIN_CONFIG)
    parser.add_argument("--finetune-config", type=Path, default=DEFAULT_FINETUNE_CONFIG)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_INFER_CONFIG)
    parser.add_argument(
        "--gpus", default="1,2,3,4",
        help=(
            "CUDA GPU ids. The config's performance.parallel_mode selects "
            "DDP, DP, or one-GPU execution."
        ),
    )
    parser.add_argument(
        "--parallel-mode",
        choices=("ddp", "dp", "single"),
        default=None,
        help="Override performance.parallel_mode from YAML.",
    )
    parser.add_argument(
        "--distributed", dest="distributed", action="store_true", default=None,
        help="Explicitly opt into one DDP process per selected GPU.",
    )
    parser.add_argument(
        "--no-distributed", dest="distributed", action="store_false",
        help="Disable DDP for single-process debugging even when multiple GPUs are visible.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global and split seed (default: 2026).")
    parser.add_argument("--resume", type=Path, default=None, help="Resume checkpoint for the selected training stage.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Pretrain checkpoint for finetune or finetune checkpoint for inference.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output root for a single stage, or parent root for --mode both/pipeline.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration and exit.")

    # Refine-compatible practical DataLoader and runtime overrides.
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--drop-last", dest="drop_last", action="store_true", default=None)
    parser.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", default=None)
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=None)
    parser.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false")
    parser.add_argument(
        "--parallel-debug",
        action="store_true",
        help=(
            "Emit one DataParallel-replica input record per forward and disable "
            "cuDNN algorithm autotuning. Stage-boundary synchronization is used "
            "without serializing every CUDA kernel."
        ),
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help=(
            "Set CUDA_LAUNCH_BLOCKING=1 before CUDA initialization so an illegal "
            "access is reported at the launching PHASE operation."
        ),
    )

    # The same split controls exposed by refine. PHASE accepts aliases used by
    # its older configs, but all overrides are written to split_strategy.
    parser.add_argument("--split-mode", choices=("manifest", "by_dataset", "by_ratio"), default=None)
    parser.add_argument("--train-datasets", default=None)
    parser.add_argument("--val-datasets", default=None)
    parser.add_argument("--test-datasets", default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--include-datasets", default=None)
    parser.add_argument("--exclude-datasets", default=None)
    return parser.parse_args(argv)


# Compatibility for older PHASE scripts. New code should use parse_arguments.
parse_args = parse_arguments


def _csv_items(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _stage_config_path(args: argparse.Namespace, stage: str) -> Path:
    if args.config is not None:
        return args.config
    if stage == "pretrain":
        return args.pretrain_config
    if stage == "finetune":
        return args.finetune_config
    if stage == "infer":
        return args.infer_config
    raise ValueError(f"Unknown PHASE stage: {stage}")


def _apply_split_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    data = config.setdefault("data", {})
    if args.include_datasets is not None:
        data["include_datasets"] = _csv_items(args.include_datasets)
    if args.exclude_datasets is not None:
        data["exclude_datasets"] = _csv_items(args.exclude_datasets)
    if not any(
        value is not None
        for value in (
            args.split_mode, args.train_datasets, args.val_datasets,
            args.test_datasets, args.train_ratio, args.val_ratio,
        )
    ):
        return
    split = dict(data.get("split_strategy", {}) or {})
    if args.split_mode is not None:
        split["mode"] = args.split_mode
    if args.train_datasets is not None:
        split["train_datasets"] = _csv_items(args.train_datasets)
    if args.val_datasets is not None:
        split["val_datasets"] = _csv_items(args.val_datasets)
    if args.test_datasets is not None:
        split["test_datasets"] = _csv_items(args.test_datasets)
    if args.train_ratio is not None:
        split["train_ratio"] = float(args.train_ratio)
    if args.val_ratio is not None:
        split["val_ratio"] = float(args.val_ratio)
    if "mode" not in split:
        split["mode"] = "by_ratio" if "train_ratio" in split else "by_dataset"
    data["split_strategy"] = split


def _apply_runtime_overrides(
    config: dict[str, Any], *, args: argparse.Namespace, stage: str
) -> dict[str, Any]:
    config["seed"] = int(args.seed)
    distributed = getattr(args, "_distributed", None)
    if distributed is not None:
        config["_distributed"] = dict(distributed)
    data = config.setdefault("data", {})
    split = data.setdefault("split_strategy", {})
    split["seed"] = int(args.seed)
    _apply_split_overrides(config, args)

    loader = config.setdefault("pretraining" if stage == "pretrain" else "training", {})
    if stage == "infer":
        loader = config.setdefault("inference", {})
    for field in ("batch_size", "num_workers", "prefetch_factor", "persistent_workers", "pin_memory", "drop_last"):
        value = getattr(args, field)
        if value is not None:
            loader[field] = value
    performance = config.setdefault("performance", {})
    legacy_parallel_override = (
        "ddp" if args.distributed else "single"
        if args.distributed is not None
        else None
    )
    parallel_mode = str(
        getattr(args, "_parallel_mode", None)
        or args.parallel_mode
        or legacy_parallel_override
        or performance.get("parallel_mode", "single" if stage == "infer" else "dp")
    ).lower()
    if parallel_mode not in {"ddp", "dp", "single"}:
        raise ValueError(
            "performance.parallel_mode must be one of: ddp, dp, single"
        )
    performance["parallel_mode"] = parallel_mode
    # Keep the former field resolved for old tooling, while the engine treats
    # performance.parallel_mode as the source of truth.
    loader["data_parallel"] = parallel_mode == "dp"
    if args.cpu_threads is not None:
        performance["cpu_num_threads"] = int(args.cpu_threads)
    if args.cudnn_benchmark is not None:
        performance["cudnn_benchmark"] = bool(args.cudnn_benchmark)
    if args.allow_tf32 is not None:
        performance["allow_tf32"] = bool(args.allow_tf32)
    if args.parallel_debug:
        performance["parallel_debug"] = True
        # cuDNN benchmark is useful for steady-state performance, but its
        # asynchronous candidate search hides the exact failing Conv3d shape.
        performance["cudnn_benchmark"] = False

    if args.resume is not None:
        config.setdefault("pretraining" if stage == "pretrain" else "training", {})["resume_path"] = str(args.resume)
    if args.checkpoint is not None:
        if stage == "pretrain":
            config.setdefault("pretraining", {})["resume_path"] = str(args.checkpoint)
        elif stage == "finetune":
            config.setdefault("training", {})["pretrained_checkpoint"] = str(args.checkpoint)
        else:
            config.setdefault("inference", {})["checkpoint"] = str(args.checkpoint)
    if args.output_dir is not None:
        config.setdefault("output", {})["root"] = str(args.output_dir)
    return config


def _load_stage_config(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    path = _stage_config_path(args, stage)
    return _apply_runtime_overrides(load_config(path), args=args, stage=stage)


def _run_preprocess_or_visualize(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "preprocess":
        from src.breast_mri_ai.breast_dce_moe_pinn_phase.preprocessing import run_preprocessing

        return run_preprocessing(dict(config.get("preprocessing", {}) or {}))
    from src.breast_mri_ai.breast_dce_moe_pinn_phase.evaluation import export_dataset_overview

    visual = dict(config.get("data_visualization", {}) or {})
    data = dict(config.get("data", {}) or {})
    for key in ("manifest_path", "dataset_root", "target_shape", "max_phases", "modalities", "relative_time_unit", "default_phase_interval_seconds", "fallback_time_confidence", "preprocessed", "ftv_segmentation"):
        visual.setdefault(key, data.get(key))
    visual.setdefault("cache_dir", dict(data.get("cache", {}) or {}).get("dir"))
    tasks = dict(config.get("tasks", {}) or {})
    visual.setdefault("active_tasks", tasks.get("active"))
    visual.setdefault("molecular_subtype_classes", tasks.get("molecular_subtype_classes"))
    return export_dataset_overview(visual)


def _run_both(args: argparse.Namespace, *, pipeline: bool) -> dict[str, Any]:
    pretrain = _load_stage_config(args, "pretrain")
    finetune = _load_stage_config(args, "finetune")
    preprocessing = None
    if pipeline:
        preprocessing = _run_preprocess_or_visualize(finetune, "preprocess")
        for config in (pretrain, finetune):
            config.setdefault("data", {})["manifest_path"] = preprocessing["output_manifest"]
            config["data"]["dataset_root"] = None
            config["data"]["preprocessed"] = True
            config["data"]["cache_dir"] = None
        _run_preprocess_or_visualize(finetune, "visualize-data")
    if args.output_dir is not None:
        base = args.output_dir.expanduser()
        pretrain.setdefault("output", {})["root"] = str(base / "pretrain")
        finetune.setdefault("output", {})["root"] = str(base / "finetune")
    from src.breast_mri_ai.breast_dce_moe_pinn_phase.engine import PHASESolver

    set_seed(args.seed)
    configure_performance(pretrain)
    pretrain_result = PHASESolver(pretrain, mode="pretrain").run()
    finetune.setdefault("training", {})["pretrained_checkpoint"] = pretrain_result["best_checkpoint"]
    set_seed(args.seed)
    configure_performance(finetune)
    finetune_result = PHASESolver(finetune, mode="finetune").run()
    return {"mode": "pipeline" if pipeline else "both", "preprocessing": preprocessing, "pretrain": pretrain_result, "finetune": finetune_result}


def _run_from_args(args: argparse.Namespace) -> int:
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    is_primary = int(getattr(args, "_distributed", {}).get("rank", 0)) == 0

    if args.mode in {"preprocess", "visualize-data"}:
        config = _load_stage_config(args, "finetune")
        if args.print_config:
            if is_primary:
                print(json.dumps(config, indent=2, ensure_ascii=False))
            return 0
        result = _run_preprocess_or_visualize(config, args.mode)
        if is_primary:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.mode in {"both", "pipeline"}:
        if args.print_config:
            if is_primary:
                print(json.dumps({"pretrain": _load_stage_config(args, "pretrain"), "finetune": _load_stage_config(args, "finetune")}, indent=2, ensure_ascii=False))
            return 0
        result = _run_both(args, pipeline=args.mode == "pipeline")
        if is_primary:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0

    config = _load_stage_config(args, args.mode)
    if args.print_config:
        if is_primary:
            print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    from src.breast_mri_ai.breast_dce_moe_pinn_phase.engine import PHASESolver

    set_seed(args.seed)
    configure_performance(config)
    result = PHASESolver(config, mode=args.mode).run()
    if is_primary:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def _selected_gpu_count(value: str) -> int:
    return len({item.strip() for item in str(value).split(",") if item.strip()})


def _configured_parallel_mode(args: argparse.Namespace) -> str:
    """Resolve one process topology before any CUDA context is created."""

    explicit = getattr(args, "parallel_mode", None)
    if explicit is None and args.distributed is not None:
        explicit = "ddp" if args.distributed else "single"
    if explicit is not None:
        return str(explicit).lower()

    stages = {
        "pretrain": ("pretrain",),
        "finetune": ("finetune",),
        "infer": ("infer",),
        "both": ("pretrain", "finetune"),
        "pipeline": ("pretrain", "finetune"),
    }.get(args.mode, ("finetune",))
    configured: dict[str, str] = {}
    for stage in stages:
        config = load_config(_stage_config_path(args, stage))
        default = "single" if stage == "infer" else "dp"
        mode = str(
            dict(config.get("performance", {}) or {}).get(
                "parallel_mode", default
            )
        ).lower()
        if mode not in {"ddp", "dp", "single"}:
            raise ValueError(
                f"{stage} performance.parallel_mode must be one of: "
                "ddp, dp, single"
            )
        configured[stage] = mode
    unique = set(configured.values())
    if len(unique) != 1:
        choices = ", ".join(
            f"{stage}={mode}" for stage, mode in configured.items()
        )
        raise ValueError(
            "--mode both/pipeline requires the same parallel mode for both "
            f"stages because process topology cannot change in-place ({choices})"
        )
    return next(iter(unique))


def _should_launch_ddp(args: argparse.Namespace) -> bool:
    """Launch one process per selected GPU only for configured DDP."""
    return bool(
        args.mode in {"pretrain", "finetune", "both", "pipeline"}
        and not args.print_config
        and _selected_gpu_count(args.gpus) > 1
        and _configured_parallel_mode(args) == "ddp"
    )


def _distributed_worker(
    local_rank: int, world_size: int, args: argparse.Namespace
) -> None:
    """DDP worker launched by the normal ``main.py`` entry point."""
    import torch
    import torch.distributed as distributed

    torch.cuda.set_device(int(local_rank))
    # Fail collectives close to their source instead of allowing one failed
    # rank to leave the remaining workers waiting indefinitely.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=int(local_rank),
        world_size=int(world_size),
        device_id=torch.device("cuda", int(local_rank)),
    )
    args._distributed = {
        "enabled": True,
        "rank": int(local_rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
    }
    try:
        _run_from_args(args)
    finally:
        distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    args._parallel_mode = _configured_parallel_mode(args)
    # Expandable virtual-memory segments are valuable for one process per GPU,
    # but PyTorch 2.7's single-process multi-GPU path on RTX 5090 systems has
    # exhibited stale-address CUDA 700 failures. Respect an explicit user
    # allocator setting; otherwise keep DP on the native allocator.
    if args._parallel_mode != "dp":
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    if args.cuda_launch_blocking:
        # This must happen before the training code creates its first CUDA
        # context.  Repeating it in _run_from_args also covers spawned DDP
        # workers and keeps direct unit invocation equivalent.
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if not _should_launch_ddp(args):
        return _run_from_args(args)

    selected = _selected_gpu_count(args.gpus)

    import torch.multiprocessing as multiprocessing

    # MASTER_PORT must be set before spawn. Choosing an explicit, overridable
    # local default keeps direct ``python main.py`` launches self contained.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29526")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    multiprocessing.spawn(
        _distributed_worker,
        args=(selected, args),
        nprocs=selected,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
