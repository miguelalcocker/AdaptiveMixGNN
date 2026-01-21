"""
Benchmark Suite for AdaptiveMixGNN

Incluye:
1. Benchmark de nuestro modelo solo (run_benchmark)
2. Comparativa con baselines: MLP, GCN, GAT (run_baselines)

Usage:
    python benchmark.py                    # Solo AdaptiveMixGNN
    python benchmark.py --baselines        # Comparativa con MLP, GCN, GAT
    python benchmark.py --runs 5           # 5 ejecuciones para estadisticas

Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from pathlib import Path
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import GCNConv, GATConv
import warnings
warnings.filterwarnings('ignore')

from model import AdaptiveMixGNN, get_optimizer, count_parameters


# ============================================================================
# MODELOS BASELINE
# ============================================================================

class MLP(nn.Module):
    """MLP que ignora la estructura del grafo."""
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GCN(nn.Module):
    """GCN estandar de Kipf & Welling (2017)."""
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    """Graph Attention Network (Velickovic et al., 2018)."""
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_features, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark AdaptiveMixGNN')
    parser.add_argument('--baselines', action='store_true',
                        help='Incluir comparativa con MLP, GCN, GAT')
    parser.add_argument('--runs', type=int, default=5,
                        help='Numero de ejecuciones (default: 5)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epocas de entrenamiento (default: 200)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension oculta (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Numero de capas (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout (default: 0.5)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Warmup epochs (default: 100)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_dataset(name, device):
    """Carga dataset y devuelve data en el dispositivo."""
    path = Path('./data')

    if name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=str(path), name=name)
        data = dataset[0].to(device)
        if data.train_mask.dim() == 2:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
        graph_type = "Heterophilic"
    else:
        dataset = Planetoid(root=str(path), name=name)
        data = dataset[0].to(device)
        graph_type = "Homophilic"

    return data, dataset, graph_type


def train_epoch(model, optimizer, data, is_mlp=False):
    model.train()
    optimizer.zero_grad()
    out = model(data.x) if is_mlp else model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, is_mlp=False):
    model.eval()
    out = model(data.x) if is_mlp else model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    results = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        results[split] = correct / total if total > 0 else 0.0
    return results


def run_single_experiment(model, optimizer, data, args, is_mlp=False):
    """Entrena un modelo con early stopping."""
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0
    epoch_times = []

    for epoch in range(args.epochs):
        start = time.time()
        train_epoch(model, optimizer, data, is_mlp)
        epoch_times.append((time.time() - start) * 1000)

        results = evaluate(model, data, is_mlp)

        if results['val'] > best_val_acc:
            best_val_acc = results['val']
            best_test_acc = results['test']
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience and epoch > args.warmup:
            break

    return best_test_acc, np.mean(epoch_times)


# ============================================================================
# BENCHMARKS
# ============================================================================

def run_benchmark_ours(args):
    """Benchmark solo de AdaptiveMixGNN."""
    datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']
    device = torch.device(args.device)
    results = {}

    print("\n" + "="*70)
    print("BENCHMARK: AdaptiveMixGNN")
    print("="*70)
    print(f"Device: {device}, Runs: {args.runs}, Epochs: {args.epochs}")

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print('='*50)

        data, dataset, graph_type = load_dataset(ds_name, device)
        test_accs, alphas = [], []

        for run in range(args.runs):
            torch.manual_seed(run * 42)
            np.random.seed(run * 42)

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

            optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
            test_acc, _ = run_single_experiment(model, optimizer, data, args)
            avg_alpha = np.mean(model.get_alpha_values(data.x))

            test_accs.append(test_acc)
            alphas.append(avg_alpha)
            print(f"  Run {run+1}: {test_acc*100:.2f}%, alpha={avg_alpha:.3f}")

        results[ds_name] = {
            'type': graph_type,
            'acc_mean': np.mean(test_accs) * 100,
            'acc_std': np.std(test_accs) * 100,
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
            'params': count_parameters(model)
        }

    # Imprimir resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"{'Dataset':<12} {'Tipo':<12} {'Accuracy':<18} {'Alpha':<12}")
    print("-"*54)
    for ds, r in results.items():
        print(f"{ds:<12} {r['type']:<12} {r['acc_mean']:.2f} +/- {r['acc_std']:.2f}%   {r['alpha_mean']:.3f}")

    return results


def run_benchmark_baselines(args):
    """Benchmark comparativo: MLP, GCN, GAT, AdaptiveMixGNN."""
    datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']
    models_config = ['MLP', 'GCN', 'GAT', 'AdaptiveMixGNN']
    device = torch.device(args.device)

    results = {ds: {m: {} for m in models_config} for ds in datasets}

    print("\n" + "="*70)
    print("BENCHMARK COMPARATIVO: MLP vs GCN vs GAT vs AdaptiveMixGNN")
    print("="*70)
    print(f"Device: {device}, Runs: {args.runs}")

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print('='*60)

        data, dataset, _ = load_dataset(ds_name, device)

        for model_name in models_config:
            accs = []

            for run in range(args.runs):
                torch.manual_seed(run * 42)
                np.random.seed(run * 42)

                if model_name == 'MLP':
                    model = MLP(dataset.num_features, args.hidden_dim, dataset.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                    is_mlp = True
                elif model_name == 'GCN':
                    model = GCN(dataset.num_features, args.hidden_dim, dataset.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                    is_mlp = False
                elif model_name == 'GAT':
                    model = GAT(dataset.num_features, args.hidden_dim, dataset.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
                    is_mlp = False
                else:  # AdaptiveMixGNN
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
                    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
                    is_mlp = False

                test_acc, _ = run_single_experiment(model, optimizer, data, args, is_mlp)
                accs.append(test_acc)

            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            results[ds_name][model_name] = {'mean': mean_acc, 'std': std_acc}
            print(f"  {model_name:20s}: {mean_acc:.2f} +/- {std_acc:.2f}%")

    # Tabla Markdown
    print("\n" + "="*70)
    print("TABLA MARKDOWN")
    print("="*70)
    print("| Modelo | " + " | ".join(datasets) + " |")
    print("|--------|" + "|".join(["--------"] * len(datasets)) + "|")
    for m in models_config:
        row = f"| {m} |"
        for ds in datasets:
            r = results[ds][m]
            row += f" {r['mean']:.2f} +/- {r['std']:.2f} |"
        print(row)

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.baselines:
        run_benchmark_baselines(args)
    else:
        run_benchmark_ours(args)
