"""
Training Example for AdaptiveMixGNN

Demonstrates complete usage including:
- Model initialization with graph structure
- Alpha evolution logging (for hypothesis validation)
- Parameter counting (Simplicity criterion)
- Ablation mode support (--ablation_mode gcn/hp)
- Training loop with evaluation

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import argparse
from pathlib import Path

from model import AdaptiveMixGNN, AlphaLogger, print_model_parameters


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

    # Dataset
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use (default: Cora)')

    # Ablation study
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=['gcn', 'hp'],
                        help='Ablation mode: "gcn" forces α=1 (pure low-pass), '
                             '"hp" forces α=0 (pure high-pass), None learns α (default: None)')

    # Logging
    parser.add_argument('--log_alpha', action='store_true',
                        help='Enable alpha evolution logging to CSV')
    parser.add_argument('--alpha_log_path', type=str, default='alpha_evolution.csv',
                        help='Path to save alpha evolution CSV (default: alpha_evolution.csv)')
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
        dataset_name: Name of the dataset (Cora, CiteSeer, PubMed)
        device: Device to place data on

    Returns:
        data: PyTorch Geometric data object
        dataset: Dataset object
    """
    # Load dataset
    path = Path('./data')
    dataset = Planetoid(root=str(path), name=dataset_name)
    data = dataset[0].to(device)

    print(f"\nDataset: {dataset_name}")
    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Edges: {data.num_edges}")
    print(f"  - Features: {dataset.num_features}")
    print(f"  - Classes: {dataset.num_classes}")
    print(f"  - Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")

    return data, dataset


def train_epoch(model, data, optimizer):
    """
    Single training epoch.

    Args:
        model: AdaptiveMixGNN model
        data: Graph data
        optimizer: Optimizer

    Returns:
        loss: Training loss
        acc: Training accuracy
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
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
        ablation_mode=args.ablation_mode
    ).to(device)

    # Print model information (Simplicity criterion)
    total_params = model.print_model_info()
    if args.verbose:
        print_model_parameters(model)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Initialize alpha logger if requested
    alpha_logger = None
    if args.log_alpha:
        alpha_logger = AlphaLogger(
            save_path=args.alpha_log_path,
            num_layers=args.num_layers
        )
        print(f"Alpha evolution will be logged to: {args.alpha_log_path}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer)

        # Evaluate
        val_loss, val_acc = evaluate(model, data, data.val_mask)
        test_loss, test_acc = evaluate(model, data, data.test_mask)

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Log alpha values
        if alpha_logger is not None:
            alpha_values = model.get_alpha_values()
            alpha_logger.log(epoch, alpha_values)

        # Print progress
        if args.verbose or epoch % 10 == 0 or epoch == 1:
            alpha_str = ""
            if epoch % 50 == 0 or epoch == 1:  # Print alpha every 50 epochs
                alphas = model.get_alpha_values()
                alpha_str = " | α=" + ",".join([f"{a:.3f}" for a in alphas])

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}"
                  f"{alpha_str}")

    print("="*70)
    print(f"\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy (at best val): {best_test_acc:.4f}")

    # Print final alpha values (for hypothesis validation)
    final_alphas = model.get_alpha_values()
    print(f"\nFinal α values (per layer):")
    for i, alpha in enumerate(final_alphas):
        interpretation = "homophily-focused" if alpha > 0.6 else "heterophily-focused" if alpha < 0.4 else "balanced"
        print(f"  Layer {i}: α = {alpha:.4f} ({interpretation})")

    print(f"\nHypothesis check:")
    print(f"  - For homophilic graphs (Cora): expect α → 1")
    print(f"  - For heterophilic graphs: expect α → 0")
    print(f"  - Current average α: {sum(final_alphas)/len(final_alphas):.4f}")

    if alpha_logger is not None:
        print(f"\nAlpha evolution saved to: {alpha_logger.save_path}")
        print("Use this CSV for plotting in your paper.")

    return best_val_acc, best_test_acc, total_params


if __name__ == '__main__':
    main()
