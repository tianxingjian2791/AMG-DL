"""
Stage 1 Training Script: Theta Prediction

Trains CNN or GNN models to predict optimal theta for C/F splitting.

Usage:
    python train_stage1.py --model CNN --dataset datasets/train --epochs 50
    python train_stage1.py --model GNN --dataset datasets/train --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from pathlib import Path

# Import models
from model import CNNModel, GNNModel, cnn_train, cnn_test, gnn_train, gnn_test

# Import data processing
from data import create_dataloaders, create_theta_data_loaders

# Import utilities
from utils import (
    Checkpointer, MetricsLogger, EarlyStopping,
    set_random_seed, count_parameters, get_device
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Stage 1: Theta Prediction')

    # Model configuration
    parser.add_argument('--model', type=str, required=True, choices=['CNN', 'GNN'],
                      help='Model type: CNN or GNN')

    # Data configuration
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset directory (e.g., datasets/unified)')
    parser.add_argument('--problem-type', type=str, default='auto',
                      choices=['auto', 'D', 'E', 'S', 'GL', 'SC'],
                      help='Problem type for dataset resolution')
    parser.add_argument('--diffusion-scale', type=str, default='small',
                      choices=['small', 'large', 'paper'],
                      help='Diffusion scale folder to use when problem-type is D')
    parser.add_argument('--train-file', type=str, default='train_D.csv',
                      help='Training CSV filename (or problem type for NPY)')
    parser.add_argument('--test-file', type=str, default='test_D.csv',
                      help='Test CSV filename (or problem type for NPY)')
    parser.add_argument('--use-npy', action='store_true',
                      help='Use NPY/NPZ format instead of CSV (5× faster)')
    parser.add_argument('--split-policy', type=str, default='test_case',
                      choices=['test_case', 'sample', 'h', 'epsilon', 'h_theta'],
                      help='Split policy for unsplit NPZ datasets (CSV CNN loader only supports sample/h_theta)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Training ratio for unsplit NPZ/CSV datasets')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay (L2 regularization)')

    # Model hyperparameters
    parser.add_argument('--hidden-channels', type=int, default=64,
                      help='Number of hidden channels')
    parser.add_argument('--dropout', type=float, default=0.25,
                      help='Dropout rate')

    # Training options
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='weights/stage1',
                      help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/stage1',
                      help='Directory to save training logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Experiment name (default: {model}_{timestamp})')

    return parser.parse_args()


def resolve_dataset_root(base_dataset, problem_type, diffusion_scale):
    """Resolve the on-disk dataset root, including scale-specific diffusion folders."""
    if problem_type != 'D':
        return base_dataset

    diffusion_root = os.path.join(base_dataset, 'diffusion', diffusion_scale)
    return diffusion_root if os.path.exists(diffusion_root) else base_dataset


def infer_problem_type(train_file, test_file, requested_problem_type):
    """Infer diffusion datasets from filenames when the problem type is left on auto."""
    if requested_problem_type != 'auto':
        return requested_problem_type

    file_blob = f"{train_file}_{test_file}".lower()
    if '_d' in file_blob:
        return 'D'
    return 'auto'


def create_model(args, device):
    """Create model based on configuration"""
    if args.model == 'CNN':
        model = CNNModel(
            in_channels=1,
            matrix_size=50,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout
        )
    else:  # GNN
        model = GNNModel(
            hidden_channels=args.hidden_channels,
            dropout=args.dropout
        )

    model = model.to(device)

    print(f"\nModel: {args.model}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Dropout: {args.dropout}")

    return model


def create_dataloaders(args):
    """Create train and test dataloaders"""
    print("\nLoading data...")
    print(f"Format: {'NPY/NPZ' if args.use_npy else 'CSV'}")

    if args.model == 'CNN':
        if args.use_npy:
            # CNN with NPY format
            from data_loader_npy import create_theta_cnn_data_loaders_npy

            # Extract problem type from filename (e.g., 'train_D.csv' -> 'train_D')
            train_problem = args.train_file.replace('.csv', '')
            test_problem = args.test_file.replace('.csv', '')

            train_loader, test_loader = create_theta_cnn_data_loaders_npy(
                dataset_root=args.dataset,
                train_problem=train_problem,
                test_problem=test_problem,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split_policy=args.split_policy,
                split_seed=args.seed,
                train_ratio=args.train_ratio
            )
        else:
            # CNN with CSV format
            train_path = os.path.join(args.dataset, 'train', 'raw', 'theta_cnn', args.train_file)
            test_path = os.path.join(args.dataset, 'test', 'raw', 'theta_cnn', args.test_file)

            if os.path.exists(train_path) and os.path.exists(test_path):
                # Pre-split legacy layout: separate train/ and test/ CSVs already on disk
                from data.cnn_data_processing import CSVDataset
                from torch.utils.data import DataLoader

                train_dataset = CSVDataset(train_path)
                test_dataset = CSVDataset(test_path)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers
                )
            else:
                # Unsplit single CSV (e.g. datasets/diffusion/large/quad/diffusion_reports):
                # split rows into train/test inside the dataloader.
                unsplit_path = os.path.join(args.dataset, 'theta_cnn', args.train_file)
                if not os.path.exists(unsplit_path):
                    unsplit_path = os.path.join(args.dataset, args.train_file)
                if not os.path.exists(unsplit_path):
                    raise FileNotFoundError(
                        "Could not find a pre-split train/test CSV pair or an unsplit "
                        f"theta_cnn CSV. Checked:\n  {train_path}\n  {test_path}\n  {unsplit_path}"
                    )

                split_policy = args.split_policy if args.split_policy in ('sample', 'h_theta') else 'sample'
                if args.split_policy not in ('sample', 'h_theta'):
                    print(
                        f"Warning: split-policy '{args.split_policy}' is not supported for the "
                        f"unsplit CSV CNN loader, falling back to 'sample'."
                    )

                from data.cnn_data_processing import create_theta_cnn_dataloaders_from_csv

                train_loader, test_loader = create_theta_cnn_dataloaders_from_csv(
                    unsplit_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train_ratio=args.train_ratio,
                    seed=args.seed,
                    split_policy=split_policy
                )

    else:  # GNN
        # GNN supports both CSV and NPY formats
        if args.use_npy:
            from data import create_theta_data_loaders_npy
            # Extract problem type from filename (e.g., 'train_D.csv' -> 'train_D')
            train_problem = args.train_file.replace('.csv', '')
            test_problem = args.test_file.replace('.csv', '')
            train_loader, test_loader = create_theta_data_loaders_npy(
                dataset_root=args.dataset,
                train_problem=train_problem,
                test_problem=test_problem,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split_policy=args.split_policy,
                split_seed=args.seed,
                train_ratio=args.train_ratio
            )
        else:
            from data import create_theta_data_loaders
            train_loader, test_loader = create_theta_data_loaders(
                data_dir=args.dataset,
                train_file=args.train_file,
                test_file=args.test_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device, model_type):
    """Train for one epoch"""
    if model_type == 'CNN':
        return cnn_train(model, loader, optimizer, device)
    else:  # GNN
        return gnn_train(model, loader, optimizer, device)


def test_epoch(model, loader, device, model_type):
    """Evaluate on test set"""
    if model_type == 'CNN':
        return cnn_test(model, loader, device)
    else:  # GNN
        return gnn_test(model, loader, device)


def main():
    """Main training loop"""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Set experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_{timestamp}"

    print("="*60)
    print("Stage 1 Training: Theta Prediction")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Problem type: {args.problem_type}")
    if args.problem_type == 'D':
        print(f"Diffusion scale: {args.diffusion_scale}")
    print(f"Experiment: {args.experiment_name}")
    print("\n\n")

    # Get device
    device = get_device()

    resolved_problem_type = infer_problem_type(args.train_file, args.test_file, args.problem_type)
    dataset_root = resolve_dataset_root(args.dataset, resolved_problem_type, args.diffusion_scale)
    if dataset_root != args.dataset:
        print(f"Resolved dataset root: {dataset_root}")
        args.dataset = dataset_root
    if resolved_problem_type != args.problem_type:
        print(f"Resolved problem type: {resolved_problem_type}")
        args.problem_type = resolved_problem_type

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(args)

    # Create model
    model = create_model(args, device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Create training utilities
    checkpointer = Checkpointer(
        checkpoint_dir=os.path.join(args.save_dir, args.experiment_name),
        max_checkpoints=5
    )

    logger = MetricsLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )

    # Log configuration
    logger.log_metadata({
        'model': args.model,
        'hidden_channels': args.hidden_channels,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'parameters': count_parameters(model)
    })

    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    print("\nStarting training...")
    best_test_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.model)
        print(f"Train Loss: {train_loss:.6f}")

        # Evaluate
        test_loss = test_epoch(model, test_loader, device, args.model)
        print(f"Test Loss:  {test_loss:.6f}")

        # Log metrics
        logger.log_epoch(epoch, train_loss, test_loss)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Save checkpoint
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            print(f"New best model! Test loss: {test_loss:.6f}")

        checkpointer.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=test_loss,
            is_best=is_best,
            metadata={'train_loss': train_loss, 'test_loss': test_loss}
        )

        # Early stopping
        if early_stopping(test_loss):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Best model saved to: {checkpointer.checkpoint_dir / 'best_model.pt'}")
    print(f"Logs saved to: {logger.log_file}")


if __name__ == '__main__':
    main()
