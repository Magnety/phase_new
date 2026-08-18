from __future__ import annotations

from collections.abc import Mapping

import torch

from ..tasks import TASK_INDEX, TASKS


def task_value(
    container: Mapping[str, torch.Tensor] | torch.Tensor,
    task: str,
    device: torch.device,
) -> torch.Tensor | None:
    if isinstance(container, Mapping):
        value = container.get(task)
        return None if value is None else value.to(device=device, non_blocking=True)
    index = TASK_INDEX.get(task)
    if index is None or container.ndim < 2 or container.shape[-1] <= index:
        return None
    return container[..., index].to(device=device, non_blocking=True)


__all__ = ["TASKS", "TASK_INDEX", "task_value"]
