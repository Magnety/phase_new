"""Deprecated compatibility import; PHASE's real entry point is now main.py."""

from ..main import (
    CONFIG_DIR,
    configure_performance,
    load_config,
    main,
    parse_arguments,
    parse_args,
    set_seed,
)

__all__ = [
    "CONFIG_DIR",
    "configure_performance",
    "load_config",
    "main",
    "parse_arguments",
    "parse_args",
    "set_seed",
]
