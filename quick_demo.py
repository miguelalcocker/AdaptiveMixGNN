"""
Quick Demo: AdaptiveMixGNN in 5 epochs

Demonstrates the model working end-to-end in ~30 seconds.
Perfect for quick testing before running full experiments.

Usage: python quick_demo.py
"""

import torch
from torch_geometric.datasets import Planetoid
from model import AdaptiveMixGNN, AlphaLogger, count_parameters
import torch.nn.functional as F


def main():
    print("="*70)
    print("AdaptiveMixGNN - Quick Demo (5 epochs)")
    print("="*70)

    # Load Cora dataset
    print("\n[1/5] Loading Cora dataset...")
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    print(f"  ✓ Nodes: {data.num_nodes}")
    print(f"  ✓ Edges: {data.num_edges}")
    print(f"  ✓ Features: {dataset.num_features}")
    print(f"  ✓ Classes: {dataset.num_classes}")

    # Initialize model
    print("\n[2/5] Initializing AdaptiveMixGNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = AdaptiveMixGNN(
        num_features=dataset.num_features,
        hidden_dim=64,
        num_classes=dataset.num_classes,
        num_layers=2,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        device=device
    ).to(device)

    total_params = count_parameters(model)
    print(f"  ✓ Model initialized")
    print(f"  ✓ Total parameters: {total_params:,}")

    # Print initial alpha
    initial_alphas = model.get_alpha_values()
    print(f"  ✓ Initial α: {[f'{a:.3f}' for a in initial_alphas]}")

    # Setup training
    print("\n[3/5] Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    alpha_logger = AlphaLogger(save_path='demo_alpha.csv', num_layers=2)
    print("  ✓ Optimizer: Adam (lr=0.01)")
    print("  ✓ Alpha logging enabled")

    # Quick training
    print("\n[4/5] Training for 5 epochs...")
    print("-"*70)

    for epoch in range(1, 6):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        # Log alpha
        alphas = model.get_alpha_values()
        alpha_logger.log(epoch, alphas)

        # Print progress
        alpha_str = ",".join([f"{a:.3f}" for a in alphas])
        print(f"Epoch {epoch}/5 | Loss: {loss:.4f} | "
              f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f} | "
              f"α: [{alpha_str}]")

    print("-"*70)

    # Results
    print("\n[5/5] Results Summary")
    print("="*70)

    final_alphas = model.get_alpha_values()
    avg_alpha = sum(final_alphas) / len(final_alphas)

    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nFinal α values:")
    for i, alpha in enumerate(final_alphas):
        print(f"  Layer {i}: α = {alpha:.4f}")
    print(f"\nAverage α: {avg_alpha:.4f}")

    # Interpretation
    if avg_alpha > 0.7:
        print("\n✓ Interpretation: HOMOPHILIC graph (α → 1)")
        print("  → Model favors low-pass filtering (smooth signals)")
        print("  → Expected for Cora dataset")
    elif avg_alpha < 0.3:
        print("\n✓ Interpretation: HETEROPHILIC graph (α → 0)")
        print("  → Model favors high-pass filtering (differences)")
    else:
        print("\n✓ Interpretation: MIXED homophily (α ≈ 0.5)")
        print("  → Model uses balanced filtering")

    print("\nAlpha evolution saved to: demo_alpha.csv")
    print("\nTo visualize:")
    print("  python visualize_alpha.py --csv demo_alpha.csv --output demo_alpha.pdf")

    print("\nTo run full experiments (200 epochs):")
    print("  python train_example.py --dataset Cora --epochs 200 --log_alpha --verbose")

    print("\nTo run all experiments for paper:")
    print("  bash run_all_experiments.sh")

    print("\n" + "="*70)
    print("Demo completed successfully! ✓")
    print("="*70)


if __name__ == '__main__':
    main()
