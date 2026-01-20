"""
Alpha Evolution Visualization for Workshop Paper

Creates publication-ready plots from alpha_evolution.csv to demonstrate:
1. Alpha convergence over training
2. Hypothesis validation (homophily vs heterophily)
3. Layer-wise behavior differences

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize alpha evolution')
    parser.add_argument('--csv', type=str, default='alpha_evolution.csv',
                        help='Path to alpha evolution CSV (default: alpha_evolution.csv)')
    parser.add_argument('--output', type=str, default='alpha_evolution.pdf',
                        help='Output plot filename (default: alpha_evolution.pdf)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title (default: auto-generated)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-paper',
                        choices=['seaborn-v0_8-paper', 'default', 'ggplot', 'bmh'],
                        help='Matplotlib style (default: seaborn-v0_8-paper)')
    return parser.parse_args()


def plot_alpha_evolution(csv_path, output_path, title=None, style='seaborn-v0_8-paper'):
    """
    Create publication-quality plot of alpha evolution.

    Args:
        csv_path: Path to alpha_evolution.csv
        output_path: Path to save plot
        title: Custom title (optional)
        style: Matplotlib style
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Set style
    try:
        plt.style.use(style)
    except:
        print(f"Warning: Style '{style}' not available, using default")
        plt.style.use('default')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get alpha columns
    alpha_cols = [col for col in df.columns if col.startswith('alpha_layer_')]

    # Plot each layer
    colors = plt.cm.tab10(range(len(alpha_cols)))
    for i, col in enumerate(alpha_cols):
        layer_num = col.split('_')[-1]
        ax.plot(df['epoch'], df[col],
                label=f'Layer {layer_num}',
                linewidth=2.5,
                color=colors[i],
                alpha=0.8)

    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='α=1 (Pure GCN)')
    ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='α=0 (Pure High-Pass)')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3, label='α=0.5 (Balanced)')

    # Formatting
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('α (Mixing Parameter)', fontsize=14, fontweight='bold')

    if title is None:
        title = 'Alpha Evolution During Training'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Add interpretation box
    final_alphas = df[alpha_cols].iloc[-1].values
    avg_alpha = final_alphas.mean()

    if avg_alpha > 0.7:
        interpretation = "Homophily-focused"
        color = 'green'
    elif avg_alpha < 0.3:
        interpretation = "Heterophily-focused"
        color = 'red'
    else:
        interpretation = "Balanced"
        color = 'orange'

    textstr = f'Final avg α: {avg_alpha:.3f}\n({interpretation})'
    props = dict(boxstyle='round', facecolor=color, alpha=0.15)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    # Show summary
    print("\n" + "="*60)
    print("Alpha Evolution Summary")
    print("="*60)
    print(f"Total epochs: {len(df)}")
    print(f"\nInitial α values:")
    for col in alpha_cols:
        layer_num = col.split('_')[-1]
        print(f"  Layer {layer_num}: {df[col].iloc[0]:.4f}")
    print(f"\nFinal α values:")
    for col in alpha_cols:
        layer_num = col.split('_')[-1]
        print(f"  Layer {layer_num}: {df[col].iloc[-1]:.4f}")
    print(f"\nAverage final α: {avg_alpha:.4f} ({interpretation})")
    print("="*60)

    return fig, avg_alpha


def create_comparison_plot(csv_files, labels, output_path='alpha_comparison.pdf'):
    """
    Compare alpha evolution across multiple experiments.

    Args:
        csv_files: List of CSV file paths
        labels: List of labels for each experiment
        output_path: Path to save comparison plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(range(len(csv_files)))

    for i, (csv_path, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_path)
        alpha_cols = [col for col in df.columns if col.startswith('alpha_layer_')]

        # Plot average alpha
        avg_alpha = df[alpha_cols].mean(axis=1)
        ax.plot(df['epoch'], avg_alpha,
                label=label,
                linewidth=2.5,
                color=colors[i],
                alpha=0.8)

    # Formatting
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average α', fontsize=14, fontweight='bold')
    ax.set_title('Alpha Evolution Comparison Across Datasets', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")

    return fig


def main():
    """Main visualization routine."""
    args = parse_args()

    # Check if file exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        print("\nRun training first:")
        print("  python train_example.py --dataset Cora --log_alpha --verbose")
        return

    # Create plot
    print(f"Loading data from: {args.csv}")
    fig, avg_alpha = plot_alpha_evolution(
        csv_path=args.csv,
        output_path=args.output,
        title=args.title,
        style=args.style
    )

    print(f"\nHypothesis Check:")
    if avg_alpha > 0.7:
        print("  ✓ α → 1: Graph is likely HOMOPHILIC (e.g., Cora, CiteSeer)")
        print("    → Model favors low-pass filtering (smooth signals)")
    elif avg_alpha < 0.3:
        print("  ✓ α → 0: Graph is likely HETEROPHILIC (e.g., Actor, Chameleon)")
        print("    → Model favors high-pass filtering (differences)")
    else:
        print("  ~ α ≈ 0.5: Graph has MIXED homophily")
        print("    → Model uses balanced filtering")

    print("\nFor your paper:")
    print("  - Include this plot in the results section")
    print("  - Highlight the adaptive behavior of α")
    print("  - Compare with ablation modes (--ablation_mode gcn/hp)")


if __name__ == '__main__':
    main()
