"""Pretraining, classification and centre-robust optimization objectives."""

from .classification import (
    cox_partial_likelihood_loss,
    masked_multitask_bce,
    masked_multitask_loss,
    pairwise_ranking_loss,
    phenotype_compactness_loss,
    prototype_margin_loss,
)
from .common import TASK_INDEX, TASKS, task_value
from .domain_robustness import (
    class_conditional_alignment_loss,
    class_conditional_domain_loss,
    cross_center_supervised_contrastive_loss,
    domain_classification_loss,
    group_dro_loss,
    orthogonality_loss,
)
from .pretraining import (
    masked_voxel_reconstruction_loss,
    pharmacokinetic_fitting_loss,
    phase_order_loss,
    vicreg_loss,
)

__all__ = [
    "TASK_INDEX",
    "TASKS",
    "class_conditional_alignment_loss",
    "class_conditional_domain_loss",
    "cox_partial_likelihood_loss",
    "cross_center_supervised_contrastive_loss",
    "domain_classification_loss",
    "group_dro_loss",
    "masked_multitask_bce",
    "masked_multitask_loss",
    "masked_voxel_reconstruction_loss",
    "orthogonality_loss",
    "pairwise_ranking_loss",
    "pharmacokinetic_fitting_loss",
    "phase_order_loss",
    "phenotype_compactness_loss",
    "prototype_margin_loss",
    "task_value",
    "vicreg_loss",
]
