"""
Plotting Script for ICLR Tiny Papers Track

Generates Figure 2: Alpha Distribution Comparison between Homophilic and Heterophilic Graphs

Usage:
    # First, generate the data:
    python train_ref.py --dataset Cora --save_alpha_distribution --epochs 200
    mv alpha_distribution_results.csv alpha_cora.csv
    python train_ref.py --dataset Texas --save_alpha_distribution --epochs 200
    mv alpha_distribution_results.csv alpha_texas.csv

    # Then combine and plot:
    python plotting_paper.py

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_alpha_data(csv_path):
    """Load alpha distribution data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def create_figure2(cora_csv='alpha_cora.csv', texas_csv='alpha_texas.csv',
                   output_path='figure2_alpha_distribution.pdf'):
    """
    Create Figure 2: Side-by-side alpha distribution comparison.

    Args:
        cora_csv: Path to Cora alpha distribution CSV
        texas_csv: Path to Texas alpha distribution CSV
        output_path: Output path for the figure
    """
    # Set style for academic paper
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Color palette
    cora_color = '#2E86AB'  # Professional blue
    texas_color = '#A23B72'  # Professional red/magenta

    # Load data
    try:
        df_cora = load_alpha_data(cora_csv)
        df_texas = load_alpha_data(texas_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease generate the data first:")
        print("  python train_ref.py --dataset Cora --save_alpha_distribution")
        print("  mv alpha_distribution_results.csv alpha_cora.csv")
        print("  python train_ref.py --dataset Texas --save_alpha_distribution")
        print("  mv alpha_distribution_results.csv alpha_texas.csv")
        return None

    # ==================== LEFT PLOT: CORA (Homophilic) ====================
    ax1 = axes[0]
    alpha_cora = df_cora['alpha_value'].values

    # KDE + Histogram
    sns.histplot(alpha_cora, kde=True, ax=ax1, color=cora_color,
                 stat='density', alpha=0.6, linewidth=0, bins=30)

    # Add vertical line at mean
    mean_cora = np.mean(alpha_cora)
    ax1.axvline(mean_cora, color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_cora:.3f}')

    # Styling
    ax1.set_xlabel(r'Mixing Coefficient $\alpha$', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(r'(a) Cora (Homophilic): $\alpha \to 1$ (Low-Pass)', fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.annotate(f'$\\bar{{\\alpha}} = {mean_cora:.3f}$\n$\\sigma = {np.std(alpha_cora):.3f}$',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ==================== RIGHT PLOT: TEXAS (Heterophilic) ====================
    ax2 = axes[1]
    alpha_texas = df_texas['alpha_value'].values

    # KDE + Histogram
    sns.histplot(alpha_texas, kde=True, ax=ax2, color=texas_color,
                 stat='density', alpha=0.6, linewidth=0, bins=30)

    # Add vertical line at mean
    mean_texas = np.mean(alpha_texas)
    ax2.axvline(mean_texas, color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {mean_texas:.3f}')

    # Styling
    ax2.set_xlabel(r'Mixing Coefficient $\alpha$', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(r'(b) Texas (Heterophilic): $\alpha \to 0$ (High-Pass)', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate(f'$\\bar{{\\alpha}} = {mean_texas:.3f}$\n$\\sigma = {np.std(alpha_texas):.3f}$',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Global title
    fig.suptitle('Figure 2: Learned $\\alpha$ Distribution Validates Adaptive Behavior',
                 fontsize=13, fontweight='bold', y=1.02)

    # Tight layout
    plt.tight_layout()

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\nFigure saved to: {output_path}")

    # Also save as PNG for quick preview
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"PNG preview saved to: {png_path}")

    # Print statistics
    print("\n" + "="*60)
    print("Alpha Distribution Statistics")
    print("="*60)
    print(f"\nCora (Homophilic):")
    print(f"  - Mean α: {mean_cora:.4f}")
    print(f"  - Std α:  {np.std(alpha_cora):.4f}")
    print(f"  - Min α:  {np.min(alpha_cora):.4f}")
    print(f"  - Max α:  {np.max(alpha_cora):.4f}")
    print(f"  - Hypothesis (α > 0.5): {'CONFIRMED' if mean_cora > 0.5 else 'NOT CONFIRMED'}")

    print(f"\nTexas (Heterophilic):")
    print(f"  - Mean α: {mean_texas:.4f}")
    print(f"  - Std α:  {np.std(alpha_texas):.4f}")
    print(f"  - Min α:  {np.min(alpha_texas):.4f}")
    print(f"  - Max α:  {np.max(alpha_texas):.4f}")
    print(f"  - Hypothesis (α < 0.5): {'CONFIRMED' if mean_texas < 0.5 else 'NOT CONFIRMED'}")
    print("="*60)

    plt.show()

    return fig


def create_combined_csv_and_plot():
    """
    Alternative: Load from single combined CSV (alpha_distribution_results.csv).
    """
    csv_path = 'alpha_distribution_results.csv'

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        print("Run train_ref.py with --save_alpha_distribution first.")
        return None

    df = pd.read_csv(csv_path)

    # Check what datasets are in the file
    datasets = df['dataset_name'].unique()
    print(f"Datasets found: {datasets}")

    if len(datasets) < 2:
        print("Need at least 2 datasets (Cora and Texas) for comparison plot.")
        print("Run train_ref.py separately for each dataset.")
        return None

    # Split by dataset
    df_cora = df[df['dataset_name'] == 'Cora']
    df_texas = df[df['dataset_name'] == 'Texas']

    if len(df_cora) == 0 or len(df_texas) == 0:
        print("Need both Cora and Texas data.")
        return None

    # Save separate CSVs
    df_cora.to_csv('alpha_cora.csv', index=False)
    df_texas.to_csv('alpha_texas.csv', index=False)

    return create_figure2()


if __name__ == '__main__':
    # Try to create figure from separate CSVs first
    cora_exists = Path('alpha_cora.csv').exists()
    texas_exists = Path('alpha_texas.csv').exists()

    if cora_exists and texas_exists:
        create_figure2()
    else:
        print("Separate CSVs not found. Attempting to use combined CSV...")
        result = create_combined_csv_and_plot()

        if result is None:
            print("\n" + "="*60)
            print("INSTRUCTIONS TO GENERATE FIGURE 2")
            print("="*60)
            print("\n1. Train on Cora and save alpha distribution:")
            print("   python train_ref.py --dataset Cora --save_alpha_distribution --epochs 200")
            print("   mv alpha_distribution_results.csv alpha_cora.csv")
            print("\n2. Train on Texas and save alpha distribution:")
            print("   python train_ref.py --dataset Texas --save_alpha_distribution --epochs 200")
            print("   mv alpha_distribution_results.csv alpha_texas.csv")
            print("\n3. Generate the figure:")
            print("   python plotting_paper.py")
            print("="*60)
