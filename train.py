"""
Training Example for AdaptiveMixGNN

Demonstrates complete usage including:
- Model initialization with graph structure
- Alpha evolution logging (for hypothesis validation)
- Parameter counting (Simplicity criterion)
- Training loop with evaluation

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WebKB
import argparse
from pathlib import Path

import csv
from model import AdaptiveMixGNN, AlphaLogger, print_model_parameters, get_optimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AdaptiveMixGNN for node classification')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer dimension (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers (default: 2)')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization) (default: 5e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (default: 0.5)')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Number of warmup epochs for α (default: 20)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed', 'Texas', 'Wisconsin', 'Cornell'],
                        help='Dataset to use (default: Cora). Texas/Wisconsin/Cornell are heterophilic.')

    # Logging
    parser.add_argument('--log_alpha', action='store_true',
                        help='Enable alpha evolution logging to CSV')
    parser.add_argument('--alpha_log_path', type=str, default='alpha_evolution.csv',
                        help='Path to save alpha evolution CSV (default: alpha_evolution.csv)')
    parser.add_argument('--save_alpha_distribution', action='store_true',
                        help='Save per-node alpha distribution to CSV for paper figures')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed training progress')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')

    return parser.parse_args()


def load_dataset(dataset_name, device):
    """
    Load and prepare graph dataset.

    Args:
        dataset_name: Name of the dataset (Cora, CiteSeer, PubMed, Texas, Wisconsin, Cornell)
        device: Device to place data on

    Returns:
        data: PyTorch Geometric data object
        dataset: Dataset object
    """
    # Load dataset
    path = Path('./data')

    # Heterophilic datasets (WebKB)
    if dataset_name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=str(path), name=dataset_name)
        data = dataset[0].to(device)

        # WebKB has multiple splits (10 splits), convert to single split format
        # Use first split (index 0) for train/val/test masks
        if data.train_mask.dim() == 2:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]

        print(f"\n⚠️  Dataset: {dataset_name} (HETEROPHILIC)")
        print(f"  - Expected: α → 0 (high-pass filtering)")
    # Homophilic datasets (Planetoid)
    else:
        dataset = Planetoid(root=str(path), name=dataset_name)
        data = dataset[0].to(device)

        print(f"\nDataset: {dataset_name} (Homophilic)")
        print(f"  - Expected: α → 1 (low-pass filtering)")

    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Edges: {data.num_edges}")
    print(f"  - Features: {dataset.num_features}")
    print(f"  - Classes: {dataset.num_classes}")
    print(f"  - Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")

    return data, dataset


def train_epoch(model, data, optimizer, warmup_epoch, current_epoch, base_lr):
    """
    Single training epoch with warmup support.

    Args:
        model: AdaptiveMixGNN model
        data: Graph data
        optimizer: Optimizer
        warmup_epoch: Number of warmup epochs
        current_epoch: Current epoch number
        base_lr: Base learning rate

    Returns:
        loss: Training loss
        acc: Training accuracy
    """
    # Increase α learning rate after warmup
    if current_epoch == warmup_epoch + 1:
        for pg in optimizer.param_groups:
            if pg['weight_decay'] == 0.0:  # α params have no weight decay
                pg['lr'] = base_lr

    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # Compute training accuracy
    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask]).float().mean()

    return loss.item(), acc.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """
    Evaluate model on given mask.

    Args:
        model: AdaptiveMixGNN model
        data: Graph data
        mask: Boolean mask for evaluation nodes

    Returns:
        loss: Evaluation loss
        acc: Evaluation accuracy
    """
    model.eval()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss
    loss = F.cross_entropy(out[mask], data.y[mask])

    # Compute accuracy
    pred = out[mask].argmax(dim=1)
    acc = (pred == data.y[mask]).float().mean()

    return loss.item(), acc.item()


@torch.no_grad()
def save_alpha_distribution(model, data, dataset_name, output_path='alpha_distribution_results.csv', append=False):
    """
    Extract and save per-node alpha values from the last layer for paper figures.

    Args:
        model: Trained AdaptiveMixGNN model
        data: Graph data
        dataset_name: Name of the dataset
        output_path: Path to save CSV
        append: If True, append to existing file

    Returns:
        alpha_last_layer: Tensor of alpha values for last layer
    """
    model.eval()

    # Forward pass to get all alpha values
    out, alphas = model(data.x, return_alpha=True)

    # Get predictions
    pred = out.argmax(dim=1)
    correct = (pred == data.y)

    # Get alpha from last layer (most informative for classification)
    alpha_last_layer = alphas[-1].squeeze().cpu().numpy()
    true_labels = data.y.cpu().numpy()
    predictions_correct = correct.cpu().numpy()

    # Write to CSV
    mode = 'a' if append else 'w'
    write_header = not append

    with open(output_path, mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['dataset_name', 'node_id', 'true_label', 'alpha_value', 'prediction_correct'])

        for node_id in range(len(alpha_last_layer)):
            writer.writerow([
                dataset_name,
                node_id,
                int(true_labels[node_id]),
                float(alpha_last_layer[node_id]),
                int(predictions_correct[node_id])
            ])

    print(f"\nAlpha distribution saved to: {output_path}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Nodes: {len(alpha_last_layer)}")
    print(f"  - Mean α: {alpha_last_layer.mean():.4f}")
    print(f"  - Std α: {alpha_last_layer.std():.4f}")

    return alpha_last_layer


def main():
    """Main training loop."""
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Load dataset
    data, dataset = load_dataset(args.dataset, device)

    # Initialize model
    model = AdaptiveMixGNN(
        num_features=dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=args.num_layers,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        device=device,
        dropout=args.dropout
    ).to(device)

    # Print model information (Simplicity criterion)
    total_params = model.print_model_info()
    if args.verbose:
        print_model_parameters(model)

    # Initialize optimizer with differential learning rates
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)


    # Initialize alpha logger if requested
    alpha_logger = None
    if args.log_alpha:
        alpha_logger = AlphaLogger(
            save_path=args.alpha_log_path,
            num_layers=args.num_layers
        )
        print(f"Alpha evolution will be logged to: {args.alpha_log_path}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs (warmup: {args.warmup_epochs}, patience: {args.patience})...")
    print("="*70)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, args.warmup_epochs, epoch, args.lr)

        # Evaluate
        val_loss, val_acc = evaluate(model, data, data.val_mask)
        test_loss, test_acc = evaluate(model, data, data.test_mask)

        # Get alpha values
        alpha_values = model.get_alpha_values(data.x)

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log alpha values
        if alpha_logger is not None:
            alpha_logger.log(epoch, alpha_values)

        # Print progress
        if args.verbose or epoch % 10 == 0 or epoch == 1:
            alpha_str = ""
            if epoch % 50 == 0 or epoch == 1:  # Print alpha every 50 epochs
                alpha_str = " | α=" + ",".join([f"{a:.3f}" for a in alpha_values])

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}"
                  f"{alpha_str}")

        # Early stopping
        if patience_counter >= args.patience and epoch > args.warmup_epochs:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print("="*70)
    print(f"\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy (at best val): {best_test_acc:.4f}")

    # Print final alpha values (for hypothesis validation)
    final_alphas = model.get_alpha_values(data.x)
    avg_alpha = sum(final_alphas) / len(final_alphas)

    print(f"\nFinal α values (per layer):")
    for i, alpha in enumerate(final_alphas):
        interpretation = "LP-focused" if alpha > 0.6 else "HP-focused" if alpha < 0.4 else "balanced"
        print(f"  Layer {i}: α = {alpha:.4f} ({interpretation})")

    print(f"\nHypothesis check:")
    if args.dataset in ['Texas', 'Wisconsin', 'Cornell']:
        expected = "α < 0.5 (heterophilic)"
        confirmed = avg_alpha < 0.5
    else:
        expected = "α > 0.5 (homophilic)"
        confirmed = avg_alpha > 0.5

    print(f"  - Expected: {expected}")
    print(f"  - Average α: {avg_alpha:.4f}")
    print(f"  - Status: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    if alpha_logger is not None:
        print(f"\nAlpha evolution saved to: {alpha_logger.save_path}")
        print("Use this CSV for plotting in your paper.")

    # Save per-node alpha distribution for figures
    if args.save_alpha_distribution:
        save_alpha_distribution(
            model, data, args.dataset,
            output_path='alpha_distribution_results.csv',
            append=False
        )

    return best_val_acc, best_test_acc, total_params


if __name__ == '__main__':
    main()
