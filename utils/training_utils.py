"""
Training utilities for unified AMG model

Includes:
    - Checkpointing
    - Logging
    - Evaluation metrics
    - Training helpers
"""

import torch
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Checkpointer:
    """
    Checkpointer for saving and loading model states
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpointer

        Parameters:
            checkpoint_dir: directory to save checkpoints
            max_checkpoints: maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save(self, model, optimizer, epoch, loss, is_best=False, metadata=None):
        """
        Save checkpoint

        Parameters:
            model: model to save
            optimizer: optimizer state
            epoch: current epoch
            loss: current loss
            is_best: whether this is the best model so far
            metadata: additional metadata dict
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
        }

        if metadata:
            checkpoint['metadata'] = metadata

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def load(self, model, optimizer=None, checkpoint_path=None):
        """
        Load checkpoint

        Parameters:
            model: model to load into
            optimizer: optimizer to load into (optional)
            checkpoint_path: specific checkpoint path (default: best_model.pt)

        Returns:
            Dict with epoch, loss, and metadata
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'metadata': checkpoint.get('metadata', {})
        }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove older checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint.unlink()


class MetricsLogger:
    """
    Logger for training metrics
    """

    def __init__(self, log_dir: str, experiment_name: str = 'experiment'):
        """
        Initialize logger

        Parameters:
            log_dir: directory to save logs
            experiment_name: name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f'{experiment_name}.json'

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'timestamps': [],
            'metadata': {}
        }

        # Load existing log if it exists
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.metrics = json.load(f)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """
        Log metrics for an epoch

        Parameters:
            epoch: current epoch
            train_loss: training loss
            val_loss: validation loss (optional)
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['timestamps'].append(datetime.now().isoformat())

        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)

        self._save()

    def log_metadata(self, metadata: Dict[str, Any]):
        """Log experiment metadata"""
        self.metrics['metadata'].update(metadata)
        self._save()

    def get_best_epoch(self, metric='val_loss'):
        """Get epoch with best metric"""
        if metric not in self.metrics or not self.metrics[metric]:
            return None

        values = self.metrics[metric]
        best_idx = np.argmin(values)
        return self.metrics['epochs'][best_idx]

    def _save(self):
        """Save metrics to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def compute_accuracy(predictions, targets, threshold=0.1):
    """
    Compute accuracy of predictions

    Parameters:
        predictions: predicted values
        targets: target values
        threshold: relative error threshold for "correct" prediction

    Returns:
        Accuracy as fraction of predictions within threshold
    """
    relative_errors = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
    correct = (relative_errors < threshold).float()
    return correct.mean().item()


def compute_relative_error(predictions, targets):
    """
    Compute mean relative error

    Parameters:
        predictions: predicted values
        targets: target values

    Returns:
        Mean relative error
    """
    relative_errors = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
    return relative_errors.mean().item()


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """

    def __init__(self, patience=10, min_delta=0.0):
        """
        Initialize early stopping

        Parameters:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Check if should stop

        Parameters:
            val_loss: current validation loss

        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility

    Parameters:
        seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """
    Count trainable parameters in model

    Parameters:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda=True):
    """
    Get computing device

    Parameters:
        prefer_cuda: whether to prefer CUDA if available. On Apple Silicon,
            this will fall back to the MPS backend when CUDA is unavailable.

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif prefer_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple Silicon GPU device: MPS')
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device
