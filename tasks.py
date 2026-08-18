"""Canonical PHASE endpoint definitions.

Labels are kept in one fixed tensor order even when only a subset of tasks is
active.  This makes checkpoints and cached batches unambiguous while allowing
``tasks.active`` to select any non-empty subset at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


ALL_TASKS = (
    "pCR",
    "HER2",
    "ER",
    "PR",
    "HR",
    "molecular_subtype",
    "survival",
)
TASKS = ALL_TASKS
TASK_INDEX = {task: index for index, task in enumerate(ALL_TASKS)}
BINARY_TASKS = ("pCR", "HER2", "ER", "PR", "HR")
MULTICLASS_TASKS = ("molecular_subtype",)
SURVIVAL_TASKS = ("survival",)
DEFAULT_SUBTYPE_CLASSES = ("0", "1", "2", "3")

_ALIASES = {
    "pcr": "pCR",
    "her2": "HER2",
    "er": "ER",
    "pr": "PR",
    "hr": "HR",
    "molecular_subtype": "molecular_subtype",
    "molecular-subtype": "molecular_subtype",
    "subtype": "molecular_subtype",
    "分子分型": "molecular_subtype",
    "survival": "survival",
    "生存预测": "survival",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    num_classes: int = 1
    class_names: tuple[str, ...] = ()


def normalize_task_name(value: object) -> str:
    text = str(value).strip()
    canonical = _ALIASES.get(text.lower(), _ALIASES.get(text, text))
    if canonical not in ALL_TASKS:
        raise ValueError(
            f"Unknown PHASE task {value!r}; choose from {', '.join(ALL_TASKS)}"
        )
    return canonical


def normalize_active_tasks(values: Iterable[object] | None) -> tuple[str, ...]:
    requested = (
        ALL_TASKS
        if values is None
        else (values,)
        if isinstance(values, (str, bytes))
        else tuple(values)
    )
    normalized = tuple(dict.fromkeys(normalize_task_name(value) for value in requested))
    if not normalized:
        raise ValueError("tasks.active must contain at least one task")
    return normalized


def task_specs(
    active_tasks: Iterable[object] | None = None,
    *,
    molecular_subtype_classes: Sequence[object] = DEFAULT_SUBTYPE_CLASSES,
) -> dict[str, TaskSpec]:
    active = normalize_active_tasks(active_tasks)
    subtype_classes = tuple(str(value).strip() for value in molecular_subtype_classes)
    if len(subtype_classes) < 2 or len(set(subtype_classes)) != len(subtype_classes):
        raise ValueError("tasks.molecular_subtype_classes must contain unique class labels")
    result: dict[str, TaskSpec] = {}
    for task in active:
        if task in BINARY_TASKS:
            result[task] = TaskSpec(task, "binary", 2, ("0", "1"))
        elif task == "molecular_subtype":
            result[task] = TaskSpec(
                task, "multiclass", len(subtype_classes), subtype_classes
            )
        else:
            result[task] = TaskSpec(task, "survival", 1, ())
    return result


def categorical_tasks(specs: Mapping[str, TaskSpec]) -> tuple[str, ...]:
    return tuple(task for task, spec in specs.items() if spec.kind != "survival")


__all__ = [
    "ALL_TASKS",
    "BINARY_TASKS",
    "DEFAULT_SUBTYPE_CLASSES",
    "MULTICLASS_TASKS",
    "SURVIVAL_TASKS",
    "TASK_INDEX",
    "TASKS",
    "TaskSpec",
    "categorical_tasks",
    "normalize_active_tasks",
    "normalize_task_name",
    "task_specs",
]
