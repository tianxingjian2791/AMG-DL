"""
Utility modules for AMG learning.

The package initializer keeps imports lazy so lightweight scripts can use
``utils.data`` without requiring training dependencies such as torch.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    # Multigrid utilities
    "compute_coarse_grid_operator": "utils.multigrid_utils",
    "jacobi_relaxation_matrix": "utils.multigrid_utils",
    "gauss_seidel_relaxation_matrix": "utils.multigrid_utils",
    "two_grid_error_matrix": "utils.multigrid_utils",
    "convergence_factor_frobenius": "utils.multigrid_utils",
    "convergence_factor_spectral": "utils.multigrid_utils",
    "TwoGridLoss": "utils.multigrid_utils",
    "compute_two_grid_convergence": "utils.multigrid_utils",
    "sparse_two_grid_error_matrix": "utils.multigrid_utils",
    # AMG utilities
    "classical_cf_splitting": "utils.amg_utils",
    "compute_strength_matrix": "utils.amg_utils",
    "compute_baseline_prolongation": "utils.amg_utils",
    "extract_coarse_nodes": "utils.amg_utils",
    "extract_fine_nodes": "utils.amg_utils",
    "compute_interpolation_sparsity_pattern": "utils.amg_utils",
    "visualize_cf_splitting": "utils.amg_utils",
    # Training utilities
    "Checkpointer": "utils.training_utils",
    "MetricsLogger": "utils.training_utils",
    "compute_accuracy": "utils.training_utils",
    "compute_relative_error": "utils.training_utils",
    "EarlyStopping": "utils.training_utils",
    "set_random_seed": "utils.training_utils",
    "count_parameters": "utils.training_utils",
    "get_device": "utils.training_utils",
    # Data utilities
    "SampleRecord": "utils.data",
    "SampleRecordRepository": "utils.data",
    "record_join_key": "utils.data",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
