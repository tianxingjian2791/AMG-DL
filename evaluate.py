"""
Evaluation Script: Comprehensive Model Evaluation

Evaluates and compares different AMG approaches:
  1. Classical AMG (baseline)
  2. Learned Theta (Stage 1 only)
  3. Full Pipeline (Stage 1 + Stage 2)

Usage:
    # Stage 1 evaluation with NPY format
    python evaluate.py --model weights/stage1/best_model.pt \
                       --model-type stage1_gnn \
                       --dataset datasets/unified \
                       --test-file test_D \
                       --use-npy

    # Stage 2 evaluation with NPY format
    python evaluate.py --model weights/stage2/best_model.pt \
                       --model-type stage2 \
                       --dataset datasets/unified \
                       --test-file test_GL \
                       --use-npy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import models
from model import create_unified_model, GNNModel, CNNModel, EncodeProcessDecode

# Import data
from data import create_theta_data_loaders

# Import utilities
from utils import get_device, set_random_seed, TwoGridLoss


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate AMG Models')

    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='unified',
                      choices=['unified', 'stage1_cnn', 'stage1_gnn', 'stage2'],
                      help='Model type')

    # Data configuration
    parser.add_argument('--dataset', type=str, required=True,
                      help='Test dataset directory')
    parser.add_argument('--problem-type', type=str, default='auto',
                      choices=['auto', 'D', 'E', 'S', 'GL', 'SC'],
                      help='Problem type for dataset resolution')
    parser.add_argument('--diffusion-scale', type=str, default='small',
                      choices=['small', 'large', 'paper'],
                      help='Diffusion scale folder to use when problem-type is D')
    parser.add_argument('--test-file', type=str, default='test_D.csv',
                      help='Test CSV filename (or problem type for NPY)')
    parser.add_argument('--use-npy', action='store_true',
                      help='Use NPY/NPZ format instead of CSV (5× faster)')
    parser.add_argument('--split-policy', type=str, default='test_case',
                      choices=['test_case', 'sample', 'h', 'epsilon'],
                      help='Split policy for unsplit NPZ datasets')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Training ratio for unsplit NPZ datasets')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')

    # Model hyperparameters (Stage 1)
    parser.add_argument('--hidden-channels', type=int, default=64,
                      help='Hidden channels (for stage1 models)')
    parser.add_argument('--dropout', type=float, default=0.25,
                      help='Dropout rate')

    # Model hyperparameters (Stage 2)
    parser.add_argument('--latent-size', type=int, default=64,
                      help='Latent dimension size (for stage2)')
    parser.add_argument('--num-layers', type=int, default=4,
                      help='Number of MLP layers (for stage2)')
    parser.add_argument('--num-message-passing', type=int, default=3,
                      help='Number of message passing rounds (for stage2)')

    # Evaluation options
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--save-predictions', action='store_true',
                      help='Save individual predictions to file')

    return parser.parse_args()


def resolve_dataset_root(base_dataset, problem_type, diffusion_scale):
    """Resolve the on-disk dataset root, including scale-specific diffusion folders."""
    if problem_type != 'D':
        return base_dataset

    diffusion_root = os.path.join(base_dataset, 'diffusion', diffusion_scale)
    return diffusion_root if os.path.exists(diffusion_root) else base_dataset


def infer_problem_type(file_name, requested_problem_type):
    if requested_problem_type != 'auto':
        return requested_problem_type

    if '_d' in file_name.lower():
        return 'D'
    return 'auto'


def load_model(args, device):
    """Load trained model"""
    print(f"\nLoading model from: {args.model}")

    if args.model_type == 'unified':
        # Load unified model
        checkpoint = torch.load(args.model, map_location=device)
        model = create_unified_model(
            stage1_type='GNN',  # Adjust based on checkpoint
            device=device
        )
        model.load_state_dict(checkpoint)

    elif args.model_type == 'stage1_gnn':
        # Load GNN theta predictor
        model = GNNModel(hidden_channels=args.hidden_channels, dropout=args.dropout)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model = model.to(device)

    elif args.model_type == 'stage1_cnn':
        # Load CNN theta predictor
        model = CNNModel(in_channels=1, hidden_channels=args.hidden_channels, dropout=args.dropout)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model = model.to(device)

    elif args.model_type == 'stage2':
        # Load Stage 2 P-value predictor
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
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model = model.to(device)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.eval()
    print(f"Model loaded successfully")

    return model


def evaluate_stage1(model, loader, device, model_type='stage1_gnn'):
    """Evaluate Stage 1 (theta prediction)"""
    print("\nEvaluating Stage 1 (Theta Prediction)...")

    predictions = []
    targets = []
    metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'relative_error': 0.0
    }

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            if model_type == 'stage1_cnn':
                # CNN model: data is tuple (input_tensor, target)
                input_data, target = data
                input_data = input_data.to(device)
                target = target.to(device)

                # Forward pass
                output = model(input_data)

                # Store predictions and targets
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
            else:
                # GNN model: data is PyG Batch object
                batch = data.to(device)

                # Forward pass
                output = model(batch)

                # Store predictions and targets
                predictions.extend(output.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # Relative error with better handling of small values
    threshold = 0.01  # Adjust based on typical scale
    mask = np.abs(targets) > threshold
    if np.any(mask):
        relative_error = np.mean(np.abs(predictions[mask] - targets[mask]) / np.abs(targets[mask]))
    else:
        # All targets are below threshold, use normalized MAE
        relative_error = mae / (np.mean(np.abs(targets)) + 1e-8)

    metrics['mse'] = float(mse)
    metrics['mae'] = float(mae)
    metrics['relative_error'] = float(relative_error)

    print(f"\nStage 1 Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Relative Error: {relative_error:.4%}")

    return metrics, predictions, targets


def reconstruct_prolongation_matrices(batch, edge_predictions):
    """
    Reconstruct prolongation matrices from edge predictions

    Returns:
        List of (A, P_pred, S) tuples for TwoGridLoss
    """
    # Handle batching
    if hasattr(batch, 'ptr'):
        batch_size = batch.num_graphs
    else:
        batch_size = 1

    matrices = []

    for graph_idx in range(batch_size):
        # Get graph-specific data
        if batch_size > 1:
            data_list = batch.to_data_list()
            graph_data = data_list[graph_idx]
            graph_edges = graph_data.edge_index
            graph_edge_attr = graph_data.edge_attr
            edge_mask = (batch.batch[batch.edge_index[0]] == graph_idx)
            graph_edge_preds = edge_predictions[edge_mask]
            A = graph_data.A_sparse
            baseline_P = graph_data.baseline_P_sparse
            S = graph_data.S_sparse
            coarse_nodes = graph_data.coarse_nodes
            num_nodes = graph_data.num_nodes
        else:
            graph_edges = batch.edge_index
            graph_edge_attr = batch.edge_attr
            graph_edge_preds = edge_predictions
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

        # Build P as PyTorch tensor
        device = graph_edge_preds.device
        P_pred = torch.zeros((num_nodes, num_coarse), dtype=torch.float32, device=device)

        # Fill in P values using predicted edge values
        for edge_idx in range(graph_edges_np.shape[1]):
            row = graph_edges_np[0, edge_idx]
            col = graph_edges_np[1, edge_idx]
            in_baseline = graph_edge_attr_np[edge_idx, 1] > 0.5

            if in_baseline:
                coarse_idx = np.where(coarse_nodes_np == col)[0]
                if len(coarse_idx) > 0:
                    coarse_idx = coarse_idx[0]
                    pred_val = graph_edge_preds[edge_idx, 0]
                    pred_val = torch.clamp(pred_val, min=-10.0, max=10.0)
                    P_pred[row, coarse_idx] = pred_val

        # Normalize rows
        row_sums = P_pred.sum(dim=1, keepdim=True)
        non_zero_mask = (row_sums.squeeze() > 1e-10)
        if non_zero_mask.any():
            P_pred[non_zero_mask] = P_pred[non_zero_mask] / row_sums[non_zero_mask]

        # For rows with no entries, use baseline
        zero_mask = ~non_zero_mask
        if zero_mask.any():
            baseline_P_dense = torch.from_numpy(baseline_P.toarray()).float().to(device)
            P_pred[zero_mask] = baseline_P_dense[zero_mask]

        # Convert A and S to numpy
        A_dense = A.toarray()
        S_dense = S.toarray()

        matrices.append((A_dense, P_pred, S_dense))

    return matrices


def evaluate_stage2(model, loader, device):
    """Evaluate Stage 2 (P-value prediction)"""
    print("\nEvaluating Stage 2 (P-Value Prediction)...")

    criterion_mse = torch.nn.MSELoss()
    criterion_twogrid = TwoGridLoss(loss_type='frobenius')

    total_mse = 0.0
    total_twogrid = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []
    all_twogrid_losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Compute MSE loss
            if hasattr(batch, 'y_edge'):
                target = batch.y_edge
            else:
                target = batch.edge_attr[:, 0:1]

            mse_loss = criterion_mse(output.edge_attr, target)

            # Compute TwoGridLoss
            P_matrices = reconstruct_prolongation_matrices(batch, output.edge_attr)
            A_list = [A for A, P, S in P_matrices]
            P_list = [P for A, P, S in P_matrices]
            S_list = [S for A, P, S in P_matrices]

            twogrid_loss = criterion_twogrid(A_list, P_list, S_list)

            total_mse += mse_loss.item()
            total_twogrid += twogrid_loss.item()
            num_batches += 1

            # Store predictions and targets (flatten to 1D)
            all_predictions.extend(output.edge_attr.squeeze().cpu().numpy().tolist())
            all_targets.extend(target.squeeze().cpu().numpy().tolist())
            all_twogrid_losses.append(twogrid_loss.item())

    avg_mse = total_mse / num_batches
    avg_twogrid = total_twogrid / num_batches

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - targets))

    # Relative error with better handling of small values
    # Use a threshold to avoid division by very small numbers
    # Only compute relative error for targets with absolute value > threshold
    threshold = 0.01  # Adjust based on typical scale of P-values
    mask = np.abs(targets) > threshold
    if np.any(mask):
        relative_error = np.mean(np.abs(predictions[mask] - targets[mask]) / np.abs(targets[mask]))
    else:
        # All targets are below threshold, use normalized MAE instead
        relative_error = mae / (np.mean(np.abs(targets)) + 1e-8)

    metrics = {
        'mse': float(avg_mse),
        'mae': float(mae),
        'relative_error': float(relative_error),
        'twogrid_frobenius': float(avg_twogrid),
        'mean_twogrid': float(np.mean(all_twogrid_losses)),
        'std_twogrid': float(np.std(all_twogrid_losses)),
        'max_twogrid': float(np.max(all_twogrid_losses)),
        'min_twogrid': float(np.min(all_twogrid_losses))
    }

    print(f"\nStage 2 Results:")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Relative Error: {relative_error:.4%}")
    print(f"  TwoGrid (Frobenius): {avg_twogrid:.6f}")
    print(f"  TwoGrid Range: [{metrics['min_twogrid']:.6f}, {metrics['max_twogrid']:.6f}]")

    return metrics, predictions, targets


def compute_statistics(predictions, targets):
    """Compute additional statistics"""
    stats = {}

    # Flatten arrays to 1D if needed (for edge-level predictions)
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Basic statistics
    stats['mean_prediction'] = float(np.mean(predictions_flat))
    stats['std_prediction'] = float(np.std(predictions_flat))
    stats['mean_target'] = float(np.mean(targets_flat))
    stats['std_target'] = float(np.std(targets_flat))

    # Error statistics
    errors = predictions_flat - targets_flat
    stats['mean_error'] = float(np.mean(errors))
    stats['std_error'] = float(np.std(errors))
    stats['max_error'] = float(np.max(np.abs(errors)))
    stats['min_error'] = float(np.min(np.abs(errors)))

    # Correlation (compute on flattened 1D arrays with proper handling of edge cases)
    if len(predictions_flat) > 1:
        try:
            # Check if arrays have zero variance (constant values)
            pred_std = np.std(predictions_flat)
            target_std = np.std(targets_flat)

            if pred_std < 1e-10 or target_std < 1e-10:
                # One or both arrays are constant, correlation is undefined
                stats['correlation'] = 0.0
            else:
                # Compute correlation
                with np.errstate(invalid='ignore', divide='ignore'):
                    correlation = np.corrcoef(predictions_flat, targets_flat)[0, 1]
                stats['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
            print(f"Warning: Could not compute correlation: {e}")
            stats['correlation'] = 0.0
    else:
        stats['correlation'] = 0.0

    # R² score
    ss_res = np.sum((targets_flat - predictions_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    stats['r2_score'] = float(r2)

    return stats


def save_results(args, metrics, stats, predictions, targets):
    """Save evaluation results"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"evaluation_{timestamp}.json"

    # Prepare results
    results = {
        'model': args.model,
        'model_type': args.model_type,
        'dataset': args.dataset,
        'timestamp': timestamp,
        'metrics': metrics,
        'statistics': stats,
        'num_samples': len(predictions)
    }

    # Save JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_file = output_dir / f"predictions_{timestamp}.npz"
        np.savez(predictions_file,
                predictions=predictions,
                targets=targets)
        print(f"Predictions saved to: {predictions_file}")

    return results_file

def main():
    """Main evaluation function"""
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    print("="*60)
    print(" "*18 + "MODEL EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Problem type: {args.problem_type}")
    if args.problem_type == 'D':
        print(f"Diffusion scale: {args.diffusion_scale}")
    print(f"Format: {'NPY/NPZ' if args.use_npy else 'CSV'}")
    print("="*60)

    resolved_problem_type = infer_problem_type(args.test_file, args.problem_type)
    dataset_root = resolve_dataset_root(args.dataset, resolved_problem_type, args.diffusion_scale)
    if dataset_root != args.dataset:
        print(f"Resolved dataset root: {dataset_root}")
        args.dataset = dataset_root
    if resolved_problem_type != args.problem_type:
        print(f"Resolved problem type: {resolved_problem_type}")
        args.problem_type = resolved_problem_type

    # Get device
    device = get_device()

    # Load model
    model = load_model(args, device)

    # Load test data
    print(f"\nLoading test data...")

    if args.model_type == 'stage2':
        # Stage 2 evaluation - P-value prediction
        if args.use_npy:
            from data import create_pvalue_data_loaders_npy
            # Extract problem type from filename
            test_problem = args.test_file.replace('.csv', '')
            _, test_loader = create_pvalue_data_loaders_npy(
                dataset_root=args.dataset,
                train_problem='train_D',  # Not used, but required
                test_problem=test_problem,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split_policy=args.split_policy,
                split_seed=args.seed,
                train_ratio=args.train_ratio
            )
        else:
            from data import create_p_value_data_loaders
            _, test_loader = create_p_value_data_loaders(
                data_dir=args.dataset,
                train_file='train_D.csv',  # Not used
                test_file=args.test_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                node_indicators=True,
                edge_indicators=True
            )

        print(f"Loaded {len(test_loader.dataset)} test samples")

        # Evaluate Stage 2
        metrics, predictions, targets = evaluate_stage2(model, test_loader, device)

    elif args.model_type.startswith('stage1'):
        # Stage 1 evaluation - theta prediction
        if args.use_npy:
            if args.model_type == 'stage1_cnn':
                # CNN model uses different NPY data loader
                from data_loader_npy import create_theta_cnn_data_loaders_npy
                # Extract problem type from filename
                test_problem = args.test_file.replace('.csv', '')
                _, test_loader = create_theta_cnn_data_loaders_npy(
                    dataset_root=args.dataset,
                    train_problem='train_D',  # Not used, but required
                    test_problem=test_problem,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    split_policy=args.split_policy,
                    split_seed=args.seed,
                    train_ratio=args.train_ratio
                )
            else:
                # GNN model uses standard NPY data loader
                from data import create_theta_data_loaders_npy
                # Extract problem type from filename
                test_problem = args.test_file.replace('.csv', '')
                _, test_loader = create_theta_data_loaders_npy(
                    dataset_root=args.dataset,
                    train_problem='train_D',  # Not used, but required
                    test_problem=test_problem,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    split_policy=args.split_policy,
                    split_seed=args.seed,
                    train_ratio=args.train_ratio
                )
        else:
            _, test_loader = create_theta_data_loaders(
                data_dir=args.dataset,
                train_file='train_D.csv',  # Not used
                test_file=args.test_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

        print(f"Loaded {len(test_loader.dataset)} test samples")

        # Evaluate Stage 1
        metrics, predictions, targets = evaluate_stage1(model, test_loader, device, args.model_type)

    else:
        # Unified model evaluation
        print("Note: Unified model evaluation requires both Stage 1 and Stage 2 data")
        if args.use_npy:
            from data import create_theta_data_loaders_npy
            test_problem = args.test_file.replace('.csv', '')
            _, test_loader = create_theta_data_loaders_npy(
                dataset_root=args.dataset,
                train_problem='train_D',
                test_problem=test_problem,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split_policy=args.split_policy,
                split_seed=args.seed,
                train_ratio=args.train_ratio
            )
        else:
            _, test_loader = create_theta_data_loaders(
                data_dir=args.dataset,
                train_file='train_D.csv',
                test_file=args.test_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

        print(f"Loaded {len(test_loader.dataset)} test samples")

        # Evaluate (currently only Stage 1)
        metrics, predictions, targets = evaluate_stage1(model, test_loader, device, args.model_type)

    # Compute statistics
    stats = compute_statistics(predictions, targets)

    # Save results
    results_file = save_results(args, metrics, stats, predictions, targets)

    print("\n" + "="*60)
    print(" "*18 + "EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_file}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
