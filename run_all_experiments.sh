#!/bin/bash
# Automated Experiment Suite for AdaptiveMixGNN
# GRaM @ ICLR 2026 Workshop Submission
#
# This script runs all experiments needed for the paper:
# 1. Ablation studies on Cora, CiteSeer, PubMed
# 2. Alpha evolution logging
# 3. Visualization generation
#
# Usage: bash run_all_experiments.sh

set -e  # Exit on error

echo "=============================================================="
echo "AdaptiveMixGNN - Automated Experiment Suite"
echo "GRaM @ ICLR 2026 - Tiny Papers Track"
echo "=============================================================="
echo ""

# Configuration
EPOCHS=200
HIDDEN_DIM=64
NUM_LAYERS=2

# Create output directories
mkdir -p results
mkdir -p figures
mkdir -p logs

echo "[1/4] Running Ablation Study on Cora..."
echo "--------------------------------------------------------------"

# Cora - GCN baseline (α=1)
echo "  → Cora: GCN baseline (α=1)"
python train_example.py \
    --dataset Cora \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --ablation_mode gcn \
    2>&1 | tee logs/cora_gcn.log

# Cora - High-Pass baseline (α=0)
echo "  → Cora: High-Pass baseline (α=0)"
python train_example.py \
    --dataset Cora \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --ablation_mode hp \
    2>&1 | tee logs/cora_hp.log

# Cora - Adaptive (learned α)
echo "  → Cora: AdaptiveMixGNN (learned α)"
python train_example.py \
    --dataset Cora \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --log_alpha \
    --alpha_log_path results/alpha_evolution_cora.csv \
    --verbose \
    2>&1 | tee logs/cora_adaptive.log

# Generate Cora visualization
python visualize_alpha.py \
    --csv results/alpha_evolution_cora.csv \
    --output figures/alpha_evolution_cora.pdf \
    --title "Alpha Evolution on Cora (Homophilic)"

echo ""
echo "[2/4] Running Ablation Study on CiteSeer..."
echo "--------------------------------------------------------------"

# CiteSeer - GCN baseline
echo "  → CiteSeer: GCN baseline (α=1)"
python train_example.py \
    --dataset CiteSeer \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --ablation_mode gcn \
    2>&1 | tee logs/citeseer_gcn.log

# CiteSeer - High-Pass baseline
echo "  → CiteSeer: High-Pass baseline (α=0)"
python train_example.py \
    --dataset CiteSeer \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --ablation_mode hp \
    2>&1 | tee logs/citeseer_hp.log

# CiteSeer - Adaptive
echo "  → CiteSeer: AdaptiveMixGNN (learned α)"
python train_example.py \
    --dataset CiteSeer \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --log_alpha \
    --alpha_log_path results/alpha_evolution_citeseer.csv \
    --verbose \
    2>&1 | tee logs/citeseer_adaptive.log

# Generate CiteSeer visualization
python visualize_alpha.py \
    --csv results/alpha_evolution_citeseer.csv \
    --output figures/alpha_evolution_citeseer.pdf \
    --title "Alpha Evolution on CiteSeer (Homophilic)"

echo ""
echo "[3/4] Running Ablation Study on PubMed..."
echo "--------------------------------------------------------------"

# PubMed - GCN baseline
echo "  → PubMed: GCN baseline (α=1)"
python train_example.py \
    --dataset PubMed \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --ablation_mode gcn \
    2>&1 | tee logs/pubmed_gcn.log

# PubMed - High-Pass baseline
echo "  → PubMed: High-Pass baseline (α=0)"
python train_example.py \
    --dataset PubMed \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --ablation_mode hp \
    2>&1 | tee logs/pubmed_hp.log

# PubMed - Adaptive
echo "  → PubMed: AdaptiveMixGNN (learned α)"
python train_example.py \
    --dataset PubMed \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --log_alpha \
    --alpha_log_path results/alpha_evolution_pubmed.csv \
    --verbose \
    2>&1 | tee logs/pubmed_adaptive.log

# Generate PubMed visualization
python visualize_alpha.py \
    --csv results/alpha_evolution_pubmed.csv \
    --output figures/alpha_evolution_pubmed.pdf \
    --title "Alpha Evolution on PubMed (Homophilic)"

echo ""
echo "[4/4] Generating Summary Report..."
echo "--------------------------------------------------------------"

# Extract results from logs
python - <<'PYTHON_SCRIPT'
import re
from pathlib import Path

# Parse log files for test accuracy
datasets = ['cora', 'citeseer', 'pubmed']
modes = ['gcn', 'hp', 'adaptive']

results = {}

for dataset in datasets:
    results[dataset] = {}
    for mode in modes:
        log_file = f'logs/{dataset}_{mode}.log'
        if Path(log_file).exists():
            with open(log_file, 'r') as f:
                content = f.read()
                # Extract test accuracy
                match = re.search(r'Test Accuracy.*?(\d+\.\d+)', content)
                if match:
                    results[dataset][mode] = float(match.group(1))
                else:
                    results[dataset][mode] = None

# Extract final alpha values
for dataset in datasets:
    csv_file = f'results/alpha_evolution_{dataset}.csv'
    if Path(csv_file).exists():
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                alphas = [float(x) for x in last_line[1:]]
                results[dataset]['alpha'] = sum(alphas) / len(alphas)

# Generate LaTeX table
print("\n" + "="*70)
print("RESULTS TABLE (LaTeX Format)")
print("="*70)
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("Model & Cora & CiteSeer & PubMed \\\\")
print("\\midrule")

# GCN row
gcn_row = "GCN ($\\alpha=1$) & "
gcn_row += " & ".join([f"{results[d]['gcn']:.2f}" if results[d].get('gcn') else "N/A" for d in datasets])
gcn_row += " \\\\"
print(gcn_row)

# HP row
hp_row = "HP ($\\alpha=0$) & "
hp_row += " & ".join([f"{results[d]['hp']:.2f}" if results[d].get('hp') else "N/A" for d in datasets])
hp_row += " \\\\"
print(hp_row)

# Adaptive row
adaptive_row = "\\textbf{AdaptiveMix (ours)} & "
adaptive_row += " & ".join([f"\\textbf{{{results[d]['adaptive']:.2f}}}" if results[d].get('adaptive') else "N/A" for d in datasets])
adaptive_row += " \\\\"
print(adaptive_row)

print("\\midrule")

# Alpha row
alpha_row = "Final $\\alpha$ & "
alpha_row += " & ".join([f"{results[d]['alpha']:.2f}" if results[d].get('alpha') else "N/A" for d in datasets])
alpha_row += " \\\\"
print(alpha_row)

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Node classification accuracy and learned $\\alpha$ values.}")
print("\\label{tab:results}")
print("\\end{table}")
print("="*70)

# Generate markdown table
print("\n" + "="*70)
print("RESULTS TABLE (Markdown Format)")
print("="*70)
print("| Model | Cora | CiteSeer | PubMed |")
print("|-------|------|----------|--------|")
print(f"| GCN (α=1) | {results['cora'].get('gcn', 'N/A'):.2f}% | {results['citeseer'].get('gcn', 'N/A'):.2f}% | {results['pubmed'].get('gcn', 'N/A'):.2f}% |")
print(f"| HP (α=0) | {results['cora'].get('hp', 'N/A'):.2f}% | {results['citeseer'].get('hp', 'N/A'):.2f}% | {results['pubmed'].get('hp', 'N/A'):.2f}% |")
print(f"| **AdaptiveMix (ours)** | **{results['cora'].get('adaptive', 'N/A'):.2f}%** | **{results['citeseer'].get('adaptive', 'N/A'):.2f}%** | **{results['pubmed'].get('adaptive', 'N/A'):.2f}%** |")
print(f"| Final α | {results['cora'].get('alpha', 'N/A'):.2f} | {results['citeseer'].get('alpha', 'N/A'):.2f} | {results['pubmed'].get('alpha', 'N/A'):.2f} |")
print("="*70)

# Save results to CSV
import csv
with open('results/summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'GCN', 'HighPass', 'AdaptiveMix', 'FinalAlpha'])
    for dataset in datasets:
        writer.writerow([
            dataset.capitalize(),
            results[dataset].get('gcn', 'N/A'),
            results[dataset].get('hp', 'N/A'),
            results[dataset].get('adaptive', 'N/A'),
            results[dataset].get('alpha', 'N/A')
        ])

print("\n✓ Summary saved to: results/summary.csv")
PYTHON_SCRIPT

echo ""
echo "=============================================================="
echo "ALL EXPERIMENTS COMPLETED ✓"
echo "=============================================================="
echo ""
echo "Generated Files:"
echo "  - results/alpha_evolution_*.csv    (Alpha evolution data)"
echo "  - figures/alpha_evolution_*.pdf    (Publication plots)"
echo "  - logs/*.log                       (Detailed training logs)"
echo "  - results/summary.csv              (Results table)"
echo ""
echo "Next Steps:"
echo "  1. Review figures/ for paper plots"
echo "  2. Copy LaTeX table from above into paper"
echo "  3. Check logs/ for detailed training info"
echo "  4. Use results/summary.csv for further analysis"
echo ""
echo "Happy writing! 📝"
echo "=============================================================="
