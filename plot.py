"""
Plotting Suite for AdaptiveMixGNN Paper

Genera las figuras para el paper:
1. Comparativa de baselines (bar chart)
2. Distribucion de alpha (histogramas)

Usage:
    python plot.py --baselines          # Genera grafica de barras baselines
    python plot.py --alpha              # Genera histogramas de alpha
    python plot.py --all                # Genera todas las figuras

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def plot_baselines_comparison(output_dir='figuras'):
    """
    Genera grafico de barras comparando MLP, GCN, GAT, AdaptiveMixGNN.

    Los datos son del benchmark ejecutado previamente (benchmark.py --baselines).
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Resultados del benchmark
    datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']

    results = {
        'MLP': [56.54, 57.10, 77.84, 79.22],
        'GCN': [81.22, 68.42, 62.70, 53.73],
        'GAT': [80.44, 67.72, 61.08, 51.76],
        'AdaptiveMixGNN': [79.54, 68.14, 80.00, 80.78],
    }

    errors = {
        'MLP': [1.02, 1.07, 1.08, 1.57],
        'GCN': [0.76, 0.52, 2.02, 2.00],
        'GAT': [1.07, 0.95, 5.82, 5.05],
        'AdaptiveMixGNN': [0.33, 0.57, 1.32, 0.78],
    }

    colors = {
        'MLP': '#95a5a6',
        'GCN': '#e74c3c',
        'GAT': '#f39c12',
        'AdaptiveMixGNN': '#3498db',
    }

    # Crear figura
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(datasets))
    width = 0.18
    multiplier = 0

    for model, accuracy in results.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, accuracy, width,
                      label=model, color=colors[model],
                      yerr=errors[model], capsize=3,
                      error_kw={'linewidth': 1})
        multiplier += 1

    # Separador entre homofilicos y heterofilicos
    separator_x = 1.5 + width * 1.5
    ax.axvline(x=separator_x, color='#7f8c8d', linestyle='--', linewidth=1.5, alpha=0.6)

    # Anotaciones
    ax.text(0.75 + width*1.5, 94, 'Homofilicos', ha='center', fontsize=11,
            color='#27ae60', fontweight='bold')
    ax.text(2.75 + width*1.5, 94, 'Heterofilicos', ha='center', fontsize=11,
            color='#c0392b', fontweight='bold')

    # Configuracion
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_title('Comparativa de Arquitecturas en Clasificacion de Nodos',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='lower left', fontsize=10, ncol=2)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Guardar
    pdf_path = f'{output_dir}/figure_baselines_comparison.pdf'
    png_path = f'{output_dir}/figure_baselines_comparison.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Graficas guardadas: {pdf_path}, {png_path}")

    plt.close()
    return pdf_path


def plot_alpha_distribution(cora_csv='alpha_cora.csv', texas_csv='alpha_texas.csv',
                           output_dir='figuras'):
    """
    Genera Figure 2: Distribucion de alpha comparando Cora vs Texas.

    Requiere los CSVs generados con:
        python train.py --dataset Cora --save_alpha_distribution
        mv alpha_distribution_results.csv alpha_cora.csv
        python train.py --dataset Texas --save_alpha_distribution
        mv alpha_distribution_results.csv alpha_texas.csv
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Verificar archivos
    if not Path(cora_csv).exists() or not Path(texas_csv).exists():
        print(f"Error: No se encontraron los archivos CSV.")
        print("\nPrimero genera los datos con:")
        print("  python train.py --dataset Cora --save_alpha_distribution --epochs 200")
        print("  mv alpha_distribution_results.csv alpha_cora.csv")
        print("  python train.py --dataset Texas --save_alpha_distribution --epochs 200")
        print("  mv alpha_distribution_results.csv alpha_texas.csv")
        return None

    # Cargar datos
    df_cora = pd.read_csv(cora_csv)
    df_texas = pd.read_csv(texas_csv)

    # Estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cora_color = '#2E86AB'
    texas_color = '#A23B72'

    # CORA (Homofilico)
    ax1 = axes[0]
    alpha_cora = df_cora['alpha_value'].values
    sns.histplot(alpha_cora, kde=True, ax=ax1, color=cora_color,
                 stat='density', alpha=0.6, linewidth=0, bins=30)
    mean_cora = np.mean(alpha_cora)
    ax1.axvline(mean_cora, color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_cora:.3f}')
    ax1.set_xlabel(r'Mixing Coefficient $\alpha$', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(r'(a) Cora (Homofilico): $\alpha \to 1$ (Low-Pass)', fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f'$\\bar{{\\alpha}} = {mean_cora:.3f}$\n$\\sigma = {np.std(alpha_cora):.3f}$',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # TEXAS (Heterofilico)
    ax2 = axes[1]
    alpha_texas = df_texas['alpha_value'].values
    sns.histplot(alpha_texas, kde=True, ax=ax2, color=texas_color,
                 stat='density', alpha=0.6, linewidth=0, bins=30)
    mean_texas = np.mean(alpha_texas)
    ax2.axvline(mean_texas, color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {mean_texas:.3f}')
    ax2.set_xlabel(r'Mixing Coefficient $\alpha$', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(r'(b) Texas (Heterofilico): $\alpha \to 0$ (High-Pass)', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f'$\\bar{{\\alpha}} = {mean_texas:.3f}$\n$\\sigma = {np.std(alpha_texas):.3f}$',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Figure 2: Learned $\\alpha$ Distribution Validates Adaptive Behavior',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Guardar
    pdf_path = f'{output_dir}/figure_alpha_distribution.pdf'
    png_path = f'{output_dir}/figure_alpha_distribution.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Graficas guardadas: {pdf_path}, {png_path}")

    # Estadisticas
    print("\n" + "="*60)
    print("Estadisticas de Alpha")
    print("="*60)
    print(f"\nCora (Homofilico): Mean={mean_cora:.4f}, Std={np.std(alpha_cora):.4f}")
    print(f"  Hipotesis (alpha > 0.5): {'CONFIRMADA' if mean_cora > 0.5 else 'NO CONFIRMADA'}")
    print(f"\nTexas (Heterofilico): Mean={mean_texas:.4f}, Std={np.std(alpha_texas):.4f}")
    print(f"  Hipotesis (alpha < 0.5): {'CONFIRMADA' if mean_texas < 0.5 else 'NO CONFIRMADA'}")

    plt.close()
    return pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description='Genera figuras para el paper')
    parser.add_argument('--baselines', action='store_true',
                        help='Genera grafica de comparativa baselines')
    parser.add_argument('--alpha', action='store_true',
                        help='Genera grafica de distribucion de alpha')
    parser.add_argument('--all', action='store_true',
                        help='Genera todas las figuras')
    parser.add_argument('--output_dir', type=str, default='figuras',
                        help='Directorio de salida (default: figuras)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.all or (not args.baselines and not args.alpha):
        # Por defecto genera todas
        plot_baselines_comparison(args.output_dir)
        plot_alpha_distribution(output_dir=args.output_dir)
    else:
        if args.baselines:
            plot_baselines_comparison(args.output_dir)
        if args.alpha:
            plot_alpha_distribution(output_dir=args.output_dir)
