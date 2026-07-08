"""
Stage 2 Training Script: P-Value Prediction

Trains GNN model to predict prolongation matrix P values with frozen Stage 1.

Usage:
    # With NPY format (5× faster, recommended)
    python train_stage2.py --stage1-weights weights/stage1/best_model.pt \
                           --dataset datasets/unified \
                           --train-file train_GL \
                           --test-file test_GL \
                           --use-npy \
                           --epochs 100

    # With CSV format (legacy)
    python train_stage2.py --stage1-weights weights/stage1/best_model.pt \
                           --dataset datasets/unified \
                           --train-file train_D.csv \
                           --test-file test_D.csv \
                           --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Import models
from model import EncodeProcessDecode

# Import data processing
from data import create_p_value_data_loaders

# Import utilities
from utils import (
    Checkpointer, MetricsLogger, EarlyStopping,
    set_random_seed, count_parameters, get_device,
    TwoGridLoss
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Stage 2: P-Value Prediction')

    # Stage 1 model
    parser.add_argument('--stage1-weights', type=str, default=None,
                      help='Path to pretrained Stage 1 weights (optional)')
    parser.add_argument('--stage1-type', type=str, default='GNN', choices=['CNN', 'GNN'],
                      help='Stage 1 model type')

    # Data configuration
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset directory')
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
                      choices=['test_case', 'sample', 'h', 'epsilon'],
                      help='Split policy for unsplit NPZ datasets')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Training ratio for unsplit NPZ datasets')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size (typically smaller for Stage 2)')
    parser.add_argument('--lr', type=float, default=3e-3,
                      help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')

    # Model hyperparameters
    parser.add_argument('--latent-size', type=int, default=64,
                      help='Latent dimension size')
    parser.add_argument('--num-layers', type=int, default=4,
                      help='Number of MLP layers')
    parser.add_argument('--num-message-passing', type=int, default=3,
                      help='Number of message passing rounds')

    # Training options
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='weights/stage2',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/stage2',
                      help='Directory to save logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Experiment name')

    return parser.parse_args()


def resolve_dataset_root(base_dataset, problem_type, diffusion_scale):
    """Resolve the on-disk dataset root, including scale-specific diffusion folders."""
    if problem_type != 'D':
        return base_dataset

    diffusion_root = os.path.join(base_dataset, 'diffusion', diffusion_scale)
    return diffusion_root if os.path.exists(diffusion_root) else base_dataset


def infer_problem_type(train_file, test_file, requested_problem_type):
    if requested_problem_type != 'auto':
        return requested_problem_type

    file_blob = f"{train_file}_{test_file}".lower()
    if '_d' in file_blob:
        return 'D'
    return 'auto'


def create_model(args, device):
    """Create Stage 2 model"""
    print("\nCreating Stage 2 model (P-value predictor)...")

    # Stage 2 configuration
    stage2_config = {
        'edge_input_size': 3,
        'node_input_size': 2,
        'global_input_size': 128,
        'edge_output_size': 1,
        'node_output_size': 1,
        'latent_size': args.latent_size,
        'num_layers': args.num_layers,
        'num_message_passing': args.num_message_passing,
        'global_block': False,
        'concat_encoder': True
    }

    model = EncodeProcessDecode(**stage2_config)
    model = model.to(device)

    print(f"Model: EncodeProcessDecode (GNN)")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Latent size: {args.latent_size}")
    print(f"Message passing rounds: {args.num_message_passing}")

    return model


def reconstruct_prolongation_matrices(batch, edge_predictions):
    """
    Reconstruct prolongation matrices from edge predictions

    Returns:
        List of (A, P_pred, S) tuples for TwoGridLoss
        - A, S: numpy arrays (fixed, no gradients needed)
        - P_pred: PyTorch tensor (trainable, needs gradients!)
    """
    import numpy as np

    # Handle batching
    if hasattr(batch, 'ptr'):
        batch_size = batch.num_graphs
    else:
        # Single graph (no batching)
        batch_size = 1

    matrices = []

    for graph_idx in range(batch_size):
        # Get graph-specific data
        if batch_size > 1:
            # Extract from batch using to_data_list()
            data_list = batch.to_data_list()
            graph_data = data_list[graph_idx]

            # Get edges for this specific graph (with LOCAL indices)
            graph_edges = graph_data.edge_index
            graph_edge_attr = graph_data.edge_attr

            # Find which edges belong to this graph in the batch
            edge_mask = (batch.batch[batch.edge_index[0]] == graph_idx)
            graph_edge_preds = edge_predictions[edge_mask]  # Keep as tensor!

            # Get stored sparse matrices
            A = graph_data.A_sparse
            baseline_P = graph_data.baseline_P_sparse
            S = graph_data.S_sparse
            coarse_nodes = graph_data.coarse_nodes
            num_nodes = graph_data.num_nodes
        else:
            # Single graph - use directly
            graph_edges = batch.edge_index
            graph_edge_attr = batch.edge_attr
            graph_edge_preds = edge_predictions  # Keep as tensor!

            A = batch.A_sparse
            baseline_P = batch.baseline_P_sparse
            S = batch.S_sparse
            coarse_nodes = batch.coarse_nodes
            num_nodes = batch.num_nodes

        # Convert to numpy/tensors
        graph_edges_np = graph_edges.cpu().numpy()
        graph_edge_attr_np = graph_edge_attr.cpu().numpy()
        coarse_nodes_np = coarse_nodes.cpu().numpy()
        num_coarse = len(coarse_nodes_np)

        # Build P as PyTorch tensor (to keep gradients!)
        device = graph_edge_preds.device
        P_pred = torch.zeros((num_nodes, num_coarse), dtype=torch.float32, device=device)

        # Fill in P values using predicted edge values
        for edge_idx in range(graph_edges_np.shape[1]):
            row = graph_edges_np[0, edge_idx]  # Local index
            col = graph_edges_np[1, edge_idx]  # Local index

            # Check if this edge is in the prolongation pattern
            in_baseline = graph_edge_attr_np[edge_idx, 1] > 0.5

            if in_baseline:
                # Find which coarse node index this corresponds to
                coarse_idx = np.where(coarse_nodes_np == col)[0]
                if len(coarse_idx) > 0:
                    coarse_idx = coarse_idx[0]
                    # Use predicted value (clamped for stability)
                    pred_val = graph_edge_preds[edge_idx, 0]
                    # Clamp to reasonable range to prevent extreme values
                    pred_val = torch.clamp(pred_val, min=-10.0, max=10.0)
                    P_pred[row, coarse_idx] = pred_val

        # Normalize rows (only for rows with non-zero sum)
        row_sums = P_pred.sum(dim=1, keepdim=True)

        # Only normalize rows that have entries
        non_zero_mask = (row_sums.squeeze() > 1e-10)
        if non_zero_mask.any():
            P_pred[non_zero_mask] = P_pred[non_zero_mask] / row_sums[non_zero_mask]

        # For rows with no entries (shouldn't happen for valid P), set identity-like structure
        zero_mask = ~non_zero_mask
        if zero_mask.any():
            # For nodes with no prolongation entries, use baseline
            # This shouldn't happen often, but prevents singular matrices
            baseline_P_dense = torch.from_numpy(baseline_P.toarray()).float().to(device)
            P_pred[zero_mask] = baseline_P_dense[zero_mask]

        # Convert A and S to numpy (these are fixed, no gradients needed)
        A_dense = A.toarray()
        S_dense = S.toarray()

        # P_pred stays as PyTorch tensor! (maintains gradient chain)
        matrices.append((A_dense, P_pred, S_dense))

    return matrices


def train_epoch(epoch, model, loader, optimizer, criterion_mse, criterion_twogrid, device, use_twogrid=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_twogrid_loss = 0
    num_batches = 0

    progress_bar = tqdm(loader, desc="Training")

    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(batch)

        # Compute MSE loss (for monitoring)
        if hasattr(batch, 'y_edge'):
            mse_loss = criterion_mse(output.edge_attr, batch.y_edge)
        else:
            # Shouldn't happen with proper data loader
            target = batch.edge_attr[:, 0:1]
            mse_loss = criterion_mse(output.edge_attr, target)

        # Compute TwoGridLoss (primary training objective - now differentiable!)
        # Reconstruct P matrices from predictions
        P_matrices = reconstruct_prolongation_matrices(batch, output.edge_attr)

        # Extract A, P, S lists
        A_list = [A for A, P, S in P_matrices]
        P_list = [P for A, P, S in P_matrices]
        S_list = [S for A, P, S in P_matrices]

        # Compute two-grid convergence loss (returns differentiable tensor!)
        twogrid_loss = criterion_twogrid(A_list, P_list, S_list)

        '''
        # Use Mixed loss for backpropagation (it's now differentiable!)
        if epoch <= 10:
            # Warm-up with MSE loss for first 10 epochs
            loss = mse_loss
        else:
            loss = 0.01 * twogrid_loss + mse_loss  # Combine with MSE loss
        '''

        if use_twogrid:
            loss = twogrid_loss
        else:
            loss = mse_loss          

        # Backward pass
        loss.backward()

        # DEBUG: gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += (p.grad.data.norm(2).item())**2
        total_grad_norm = total_grad_norm ** 0.5 

        optimizer.step()

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_twogrid_loss += (twogrid_loss.item() if isinstance(twogrid_loss, torch.Tensor) else twogrid_loss)
        num_batches += 1

        # Display progress
        if isinstance(twogrid_loss, torch.Tensor):
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mse': f'{mse_loss.item():.6f}',
                'twogrid': f'{twogrid_loss.item():.6f}',
                'grad': f'{total_grad_norm:.3e}'
            })
        else:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mse': f'{mse_loss.item():.6f}'
            })

    return total_loss / num_batches, total_mse_loss / num_batches, total_twogrid_loss / num_batches


def test_epoch(epoch, model, loader, criterion_mse, criterion_twogrid, device, use_twogrid=True):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    total_mse_loss = 0
    total_twogrid_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Compute MSE loss
            if hasattr(batch, 'y_edge'):
                mse_loss = criterion_mse(output.edge_attr, batch.y_edge)
            else:
                target = batch.edge_attr[:, 0:1]
                mse_loss = criterion_mse(output.edge_attr, target)

            # Compute TwoGridLoss (now returns differentiable tensor)
            P_matrices = reconstruct_prolongation_matrices(batch, output.edge_attr)
            A_list = [A for A, P, S in P_matrices]
            P_list = [P for A, P, S in P_matrices]
            S_list = [S for A, P, S in P_matrices]

            twogrid_loss = criterion_twogrid(A_list, P_list, S_list)

            '''
            if epoch <= 10:
                # During warm-up, report MSE loss only 
                loss = mse_loss
            else:            
                loss = mse_loss + 0.01 * twogrid_loss
            '''

            if use_twogrid:
                loss = twogrid_loss
            else:
                loss = mse_loss   

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_twogrid_loss += twogrid_loss.item()
            num_batches += 1

    return total_loss / num_batches, total_mse_loss / num_batches, total_twogrid_loss / num_batches


def main():
    """Main training loop"""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Set experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"stage2_{timestamp}"

    print("="*60)
    print("Stage 2 Training: P-Value Prediction")
    print(f"Dataset: {args.dataset}")
    print(f"Problem type: {args.problem_type}")
    if args.problem_type == 'D':
        print(f"Diffusion scale: {args.diffusion_scale}")
    print(f"Experiment: {args.experiment_name}")
    if args.stage1_weights:
        print(f"Stage 1 weights: {args.stage1_weights}")
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
    print("\nLoading data...")
    print(f"Format: {'NPY/NPZ' if args.use_npy else 'CSV'}")

    if args.use_npy:
        from data import create_pvalue_data_loaders_npy
        # Extract problem type from filename (e.g., 'train_D.csv' -> 'train_D')
        train_problem = args.train_file.replace('.csv', '')
        test_problem = args.test_file.replace('.csv', '')
        train_loader, test_loader = create_pvalue_data_loaders_npy(
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
        train_loader, test_loader = create_p_value_data_loaders(
            data_dir=args.dataset,
            train_file=args.train_file,
            test_file=args.test_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            node_indicators=True,
            edge_indicators=True
        )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

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

    # Loss functions
    criterion_mse = nn.MSELoss()  # For monitoring
    criterion_twogrid = TwoGridLoss(loss_type='frobenius')  # Primary training objective

    print(f"\nLoss functions:")
    print("  TwoGridLoss (Frobenius norm)")
    # print(f"  Warm up:  MSE loss")
    # print(f"  Following: MSE + 0.01 * TwoGridLoss")

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
        'stage': 2,
        'latent_size': args.latent_size,
        'num_layers': args.num_layers,
        'num_message_passing': args.num_message_passing,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
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
        train_loss, train_mse, train_twogrid = train_epoch(epoch,
            model, train_loader, optimizer, criterion_mse, criterion_twogrid, device, use_twogrid=True
        )
        print(f"Train Loss: {train_loss:.6f} | MSE: {train_mse:.6f} | TwoGrid: {train_twogrid:.6f}")

        # Evaluate
        test_loss, test_mse, test_twogrid = test_epoch(epoch,
            model, test_loader, criterion_mse, criterion_twogrid, device, use_twogrid=True
        )
        print(f"Test Loss:  {test_loss:.6f} | MSE: {test_mse:.6f} | TwoGrid: {test_twogrid:.6f}")

        # Log metrics
        logger.log_epoch(epoch, train_twogrid, test_twogrid)

        # Learning rate scheduling
        scheduler.step(test_twogrid)  # Use combined metric for scheduling

        # Save checkpoint
        is_best = test_twogrid < best_test_loss
        if is_best:
            best_test_loss = test_twogrid
            print(f"New best model! Test loss: {best_test_loss:.6f}")

        checkpointer.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=test_twogrid,
            is_best=is_best,
            metadata={'train_loss': train_loss, 'test_loss': test_loss, 'test_mse': test_mse, 'test_twogrid': test_twogrid}
        )

        # Early stopping
        if early_stopping(test_twogrid):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\n\n")
    print("Training Complete!")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Best model saved to: {checkpointer.checkpoint_dir / 'best_model.pt'}")

if __name__ == '__main__':
    main()
