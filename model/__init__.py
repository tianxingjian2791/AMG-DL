"""
AMG Learning Models

This package contains neural network models for accelerating Algebraic Multigrid (AMG) methods:
    - CNNModel: CNN for theta/rho prediction (existing)
    - GNNModel: Simple Graph Neural Network for theta/rho prediction (new)
    - EncodeProcessDecode: Graph network for P-value prediction (new)
    - UnifiedAMGModel: Two-stage model combining theta and P-value prediction (new)
"""

from .cnn_model import CNNModel, train as cnn_train, test as cnn_test
from .gnn_model import GNNModel, train as gnn_train, test as gnn_test
from .graph_net_model import (
    EncodeProcessDecode,
    MLPBlock,
    EdgeModel,
    NodeModel,
    GlobalModel,
    GraphNetwork
)

__all__ = [
    # Stage 1 models (theta prediction)
    'CNNModel',
    'GNNModel',

    # Stage 2 model (P-value prediction)
    'EncodeProcessDecode',

    # Training/testing functions
    'cnn_train',
    'cnn_test',
    'gat_train',
    'gat_test',
    'gnn_train',
    'gnn_test',

    # Graph network components
    'MLPBlock',
    'EdgeModel',
    'NodeModel',
    'GlobalModel',
    'GraphNetwork'
]
