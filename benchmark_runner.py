"""
Benchmark Runner for ICLR Tiny Papers Track

Trains AdaptiveMixGNN on multiple datasets and generates results table.

Usage:
    python benchmark_runner.py
    python benchmark_runner.py --runs 5  # For statistical significance

Output:
    - results_table.md: Markdown table with results
    - Console output with detailed statistics

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WebKB
import argparse
import time
import numpy as np
from pathlib import Path

from model_ref import AdaptiveMixGNN, get_optimizer, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark AdaptiveMixGNN')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per dataset for mean/std (default: 3)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs per run (default: 200)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout (default: 0.5)')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Warmup epochs (default: 20)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_dataset(dataset_name, device):
    """Load dataset."""
    path = Path('./data')

    if dataset_name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=str(path), name=dataset_name)
        data = dataset[0].to(device)
        if data.train_mask.dim() == 2:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
        graph_type = "Heterophilic"
    else:
        dataset = Planetoid(root=str(path), name=dataset_name)
        data = dataset[0].to(device)
        graph_type = "Homophilic"

    return data, dataset, graph_type


def train_single_run(data, dataset, args, device, verbose=False):
    """
    Train model for a single run.

    Returns:
        test_acc: Best test accuracy
        total_params: Number of parameters
        avg_epoch_time_ms: Average training time per epoch in milliseconds
        final_alpha: Final average alpha value
    """
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

    total_params = count_parameters(model)
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_state = None
    patience_counter = 0

    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        # Warmup LR adjustment
        if epoch == args.warmup_epochs + 1:
            for pg in optimizer.param_groups:
                if pg['weight_decay'] == 0.0:
                    pg['lr'] = args.lr

        # Training
        start_time = time.time()

        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_time = (time.time() - start_time) * 1000  # ms
        epoch_times.append(epoch_time)

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
            test_pred = out[data.test_mask].argmax(dim=1)
            test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience and epoch > args.warmup_epochs:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

    # Restore best model and get final alpha
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_alpha = model.get_alpha_values(data.x)
    avg_alpha = sum(final_alpha) / len(final_alpha)
    avg_epoch_time = np.mean(epoch_times)

    return best_test_acc, total_params, avg_epoch_time, avg_alpha


def run_benchmark(datasets, args):
    """
    Run benchmark on multiple datasets.

    Returns:
        results: Dictionary with results for each dataset
    """
    device = torch.device(args.device)
    results = {}

    print("\n" + "="*70)
    print("BENCHMARK: AdaptiveMixGNN on Node Classification")
    print("="*70)
    print(f"Device: {device}")
    print(f"Runs per dataset: {args.runs}")
    print(f"Max epochs: {args.epochs}")
    print(f"Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    print("="*70 + "\n")

    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        data, dataset, graph_type = load_dataset(dataset_name, device)
        print(f"  Type: {graph_type}")
        print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        print(f"  Features: {dataset.num_features}, Classes: {dataset.num_classes}")

        test_accs = []
        epoch_times = []
        alphas = []
        total_params = None

        for run in range(args.runs):
            print(f"  Run {run + 1}/{args.runs}...", end=" ", flush=True)

            test_acc, params, epoch_time, avg_alpha = train_single_run(
                data, dataset, args, device, verbose=False
            )

            test_accs.append(test_acc)
            epoch_times.append(epoch_time)
            alphas.append(avg_alpha)
            total_params = params

            print(f"Test Acc: {test_acc:.4f}, α: {avg_alpha:.3f}")

        # Compute statistics
        results[dataset_name] = {
            'graph_type': graph_type,
            'test_acc_mean': np.mean(test_accs) * 100,
            'test_acc_std': np.std(test_accs) * 100,
            'params': total_params,
            'epoch_time_mean': np.mean(epoch_times),
            'epoch_time_std': np.std(epoch_times),
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
        }

        print(f"\n  Summary: {results[dataset_name]['test_acc_mean']:.2f} +/- {results[dataset_name]['test_acc_std']:.2f}%")
        print(f"  Avg α: {results[dataset_name]['alpha_mean']:.3f} +/- {results[dataset_name]['alpha_std']:.3f}")

    return results


def generate_markdown_table(results, output_path='results_table.md'):
    """Generate Markdown table from results."""

    lines = []
    lines.append("# AdaptiveMixGNN Benchmark Results\n")
    lines.append("Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)\n\n")

    # Main results table
    lines.append("## Table 1: Node Classification Accuracy\n")
    lines.append("| Dataset | Type | Test Accuracy | Parameters | Time/Epoch (ms) | Avg α |")
    lines.append("|---------|------|---------------|------------|-----------------|-------|")

    for dataset_name, r in results.items():
        lines.append(
            f"| {dataset_name} | {r['graph_type']} | "
            f"{r['test_acc_mean']:.2f} ± {r['test_acc_std']:.2f}% | "
            f"{r['params']:,} | "
            f"{r['epoch_time_mean']:.2f} ± {r['epoch_time_std']:.2f} | "
            f"{r['alpha_mean']:.3f} |"
        )

    lines.append("\n")

    # Hypothesis validation
    lines.append("## Hypothesis Validation\n")
    lines.append("| Dataset | Type | Expected α | Observed α | Status |")
    lines.append("|---------|------|------------|------------|--------|")

    for dataset_name, r in results.items():
        if r['graph_type'] == 'Homophilic':
            expected = "α > 0.5"
            confirmed = r['alpha_mean'] > 0.5
        else:
            expected = "α < 0.5"
            confirmed = r['alpha_mean'] < 0.5

        status = "CONFIRMED" if confirmed else "NOT CONFIRMED"
        lines.append(
            f"| {dataset_name} | {r['graph_type']} | {expected} | "
            f"{r['alpha_mean']:.3f} ± {r['alpha_std']:.3f} | {status} |"
        )

    lines.append("\n")

    # Notes
    lines.append("## Notes\n")
    lines.append("- Results averaged over multiple runs with different random seeds\n")
    lines.append("- α → 1 indicates low-pass (homophilic) filtering preference\n")
    lines.append("- α → 0 indicates high-pass (heterophilic) filtering preference\n")
    lines.append("- Early stopping with patience=50 after warmup\n")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n{'='*70}")
    print(f"Results table saved to: {output_path}")
    print(f"{'='*70}")

    # Print table to console
    print("\n" + '\n'.join(lines))

    return output_path


def main():
    args = parse_args()

    # Datasets to benchmark
    datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']

    # Run benchmark
    results = run_benchmark(datasets, args)

    # Generate Markdown table
    generate_markdown_table(results, 'results_table.md')

    # Also save alpha distributions for plotting
    print("\n" + "="*70)
    print("TO GENERATE FIGURE 2 (Alpha Distribution):")
    print("="*70)
    print("python train_ref.py --dataset Cora --save_alpha_distribution --epochs 200")
    print("mv alpha_distribution_results.csv alpha_cora.csv")
    print("python train_ref.py --dataset Texas --save_alpha_distribution --epochs 200")
    print("mv alpha_distribution_results.csv alpha_texas.csv")
    print("python plotting_paper.py")
    print("="*70)


if __name__ == '__main__':
    main()
