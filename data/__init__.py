"""
Data processing modules for AMG learning

This package contains dataset and dataloader classes for:
    - CNN data (pooled matrix images)
    - GNN data (sparse graph representations)
    - Unified data (both theta and P-value prediction)
    - NPY/NPZ data (binary format)
"""

from .cnn_data_processing import (
    CSVDataset,
    create_dataloaders
)

from .gnn_data_processing import (
    GNNThetaDataset,
    GNNPValueDataset,
    create_theta_data_loaders,
    create_p_value_data_loaders
)

from .unified_data_processing import (
    UnifiedAMGDataset,
    UnifiedAMGDataLoader,
    create_unified_data_loaders,
    split_unified_csv
)

# Import NPY loaders from root directory (already implemented)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader_npy import (
    GNNThetaDatasetNPY,
    CNNThetaDatasetNPY,
    PValueDatasetNPY,
    create_theta_data_loaders_npy,
    create_theta_cnn_data_loaders_npy,
    create_pvalue_data_loaders_npy
)

__all__ = [
    # CNN data
    'CSVDataset',
    'create_dataloaders',

    # GNN data (CSV format)
    'GNNThetaDataset',
    'GNNPValueDataset',
    'create_theta_data_loaders',
    'create_p_value_data_loaders',

    # GNN data (NPY format - high performance)
    'GNNThetaDatasetNPY',
    'CNNThetaDatasetNPY',
    'PValueDatasetNPY',
    'create_theta_data_loaders_npy',
    'create_theta_cnn_data_loaders_npy',
    'create_pvalue_data_loaders_npy',

    # Unified data
    'UnifiedAMGDataset',
    'UnifiedAMGDataLoader',
    'create_unified_data_loaders',
    'split_unified_csv'
]
