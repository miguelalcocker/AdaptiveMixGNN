# AdaptiveMixGNN: Graph Neural Network with a Filter Bank for Heterophilic Graphs

Implementation of a node-classification model based on principles from Graph Signal Processing (GSP).

## Overview

AdaptiveMixGNN addresses the challenge of heterophily in graph neural networks through a **Filter Bank** architecture that processes low-frequency (homophilic) and high-frequency (heterophilic) signals.

### Key Innovation

The model learns a **node-wise mixing parameter α** (α_i ∈ [0,1]) that adaptively balances:
- **Low-pass filter** (S_LP): aggregates information from similar neighbors (homophily)
- **High-pass filter** (S_HP): captures patterns from dissimilar neighbors (heterophily)

**Hypothesis**: α → 1 in homophilic graphs, α → 0 in heterophilic graphs

## Mathematical Specification

### Signal Propagation

For each layer l, signal propagation is defined as:

```text
z_LP = S_LP · X_{l-1}
z_HP = S_HP · X_{l-1}
α_i = σ(x_i · θ + b)           (per node)
z_mix = α ⊙ z_LP + (1-α) ⊙ z_HP
X_l = σ(z_mix · W + bias)
```

**Graph Shift Operators (GSOs):**

1. **Low-pass GSO (S_LP)**: GCN-style normalized adjacency
   ```text
   S_LP = D̃^(-1/2) · Ã · D̃^(-1/2)
   where Ã = A + I
   ```

2. **High-pass GSO (S_HP)**: spectral difference filter
   ```text
   S_HP = I - S_LP
   ```

## Installation

```bash
# Create environment
conda create -n adaptivemix python=3.10
conda activate adaptivemix

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python train.py --dataset Cora --epochs 200 --verbose
```

### AdaptiveMixGNN Benchmark

```bash
# Only our model on all datasets
python benchmark.py

# Comparison with baselines (MLP, GCN, GAT)
python benchmark.py --baselines

# More runs for statistics
python benchmark.py --baselines --runs 10
```

### Figure Generation

```bash
# Bar plot comparing baselines
python plot.py --baselines

# Alpha distribution histograms (requires CSVs)
python plot.py --alpha

# All figures
python plot.py --all
```

### Alpha Distribution Extraction

To generate alpha histograms, you first need to extract the data:

```bash
# Generate CSV for Cora
python train.py --dataset Cora --save_alpha_distribution --epochs 200
mv alpha_distribution_results.csv alpha_cora.csv

# Generate CSV for Texas
python train.py --dataset Texas --save_alpha_distribution --epochs 200
mv alpha_distribution_results.csv alpha_texas.csv

# Generate figure
python plot.py --alpha
```

## Command-Line Arguments

### train.py

```text
Architecture:
  --hidden_dim        Hidden layer dimension (default: 64)
  --num_layers        Number of GNN layers (default: 2)
  --dropout           Dropout rate (default: 0.5)

Training:
  --epochs            Training epochs (default: 200)
  --lr                Learning rate (default: 0.01)
  --weight_decay      L2 regularization (default: 5e-4)
  --patience          Early stopping patience (default: 50)
  --warmup_epochs     Warmup epochs for α (default: 20)

Dataset:
  --dataset           Cora, CiteSeer, Texas, Wisconsin, Cornell (default: Cora)

Logging:
  --save_alpha_distribution  Save node-wise α distribution to CSV
  --log_alpha               Save α evolution per epoch
  --verbose                 Print detailed progress
```

### benchmark.py

```text
  --baselines         Include comparison with MLP, GCN, GAT
  --runs              Number of runs (default: 5)
  --epochs            Epochs per run (default: 200)
```

## Results

### Node Classification Accuracy

| Dataset | Type | Test Accuracy | Avg. α |
|---------|------|---------------|--------|
| Cora | Homophilic | 79.54 ± 0.33% | 0.897 |
| CiteSeer | Homophilic | 68.14 ± 0.57% | 0.842 |
| Texas | Heterophilic | 80.00 ± 1.32% | 0.480 |
| Wisconsin | Heterophilic | 80.78 ± 0.78% | 0.450 |

### Baseline Comparison

| Model | Cora | CiteSeer | Texas | Wisconsin |
|--------|------|----------|-------|-----------|
| MLP | 56.54 ± 1.02 | 57.10 ± 1.07 | 77.84 ± 1.08 | 79.22 ± 1.57 |
| GCN | 81.22 ± 0.76 | 68.42 ± 0.52 | 62.70 ± 2.02 | 53.73 ± 2.00 |
| GAT | 80.44 ± 1.07 | 67.72 ± 0.95 | 61.08 ± 5.82 | 51.76 ± 5.05 |
| **AdaptiveMixGNN** | 79.54 ± 0.33 | 68.14 ± 0.57 | **80.00 ± 1.32** | **80.78 ± 0.78** |

### Hypothesis Validation

| Dataset | Type | Expected α | Observed α | Status |
|---------|------|------------|------------|--------|
| Cora | Homophilic | α > 0.5 | 0.897 | CONFIRMED |
| CiteSeer | Homophilic | α > 0.5 | 0.842 | CONFIRMED |
| Texas | Heterophilic | α < 0.5 | 0.480 | CONFIRMED |
| Wisconsin | Heterophilic | α < 0.5 | 0.450 | CONFIRMED |

## Code Structure

```text
.
├── model.py           # AdaptiveMixGNN implementation
│   ├── compute_graph_shift_operators()  # Pre-compute S_LP, S_HP
│   ├── AdaptiveMixGNNLayer              # Layer with node-wise α mixing
│   ├── AdaptiveMixGNN                   # Full model
│   └── get_optimizer()                  # Optimizer with differentiated LR
│
├── train.py           # Training script
│   ├── Dataset loading (Planetoid, WebKB)
│   ├── Training loop with early stopping
│   └── Alpha distribution extraction
│
├── benchmark.py       # Benchmark suite
│   ├── run_benchmark_ours()      # AdaptiveMixGNN only
│   └── run_benchmark_baselines() # MLP/GCN/GAT comparison
│
├── plot.py            # Figure generation
│   ├── plot_baselines_comparison()  # Bar chart
│   └── plot_alpha_distribution()    # α histograms
│
├── figures/           # Generated figures
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## References

### Graph Signal Processing
- Kipf & Welling (2017): Semi-Supervised Classification with GCNs
- Sandryhaila & Moura (2013): Discrete Signal Processing on Graphs
- Defferrard et al. (2016): Convolutional Neural Networks on Graphs

### Heterophily in GNNs
- Zhu et al. (2020): Beyond Homophily in Graph Neural Networks
- Chien et al. (2021): Adaptive Universal Generalized PageRank GNN
- Bo et al. (2021): Beyond Low-frequency Information in GCNs

## License

This code is provided for research and educational purposes.
