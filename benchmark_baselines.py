"""
Benchmark: AdaptiveMixGNN vs Baselines (MLP, GCN, GAT)

Compara nuestro modelo contra baselines estándar para demostrar:
1. MLP < GNN (el grafo aporta valor)
2. GCN funciona bien en homofílicos pero mal en heterofílicos
3. AdaptiveMixGNN funciona bien en ambos escenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import GCNConv, GATConv
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_RUNS = 5
N_EPOCHS = 200
HIDDEN_DIM = 64
PATIENCE = 50
WARMUP_EPOCHS = 100

print(f"Device: {DEVICE}")
print(f"Runs: {N_RUNS}, Epochs: {N_EPOCHS}, Hidden: {HIDDEN_DIM}")
print("=" * 70)


# ============================================================================
# MODELOS BASELINE
# ============================================================================

class MLP(nn.Module):
    """MLP que ignora completamente la estructura del grafo."""
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
    """GCN estándar de Kipf & Welling (2017)."""
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
    """Graph Attention Network (Veličković et al., 2018)."""
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
# NUESTRO MODELO (importado)
# ============================================================================

from model_ref import AdaptiveMixGNN, get_optimizer


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_epoch(model, optimizer, data, is_mlp=False):
    model.train()
    optimizer.zero_grad()

    if is_mlp:
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)

    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, is_mlp=False):
    model.eval()

    if is_mlp:
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)

    results = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        results[split] = correct / total if total > 0 else 0.0

    return results


def run_experiment(model, optimizer, data, n_epochs, patience, warmup, is_mlp=False):
    """Entrena un modelo con early stopping."""
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

    for epoch in range(n_epochs):
        loss = train_epoch(model, optimizer, data, is_mlp)
        results = evaluate(model, data, is_mlp)

        if results['val'] > best_val_acc:
            best_val_acc = results['val']
            best_test_acc = results['test']
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping después del warmup
        if patience_counter >= patience and epoch > warmup:
            break

    return best_test_acc


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_dataset(name):
    """Carga un dataset y devuelve data en el dispositivo."""
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'data/{name}', name=name)
    elif name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=f'data/{name}', name=name)
    else:
        raise ValueError(f"Dataset {name} no soportado")

    data = dataset[0].to(DEVICE)

    # Para WebKB, usar split 0
    if name in ['Texas', 'Wisconsin', 'Cornell']:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    return data, dataset.num_features, dataset.num_classes


# ============================================================================
# BENCHMARK PRINCIPAL
# ============================================================================

def run_benchmark():
    datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']
    models_config = ['MLP', 'GCN', 'GAT', 'AdaptiveMixGNN']

    results = {ds: {m: [] for m in models_config} for ds in datasets}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print('='*70)

        data, num_features, num_classes = load_dataset(dataset_name)
        print(f"  Nodes: {data.num_nodes}, Features: {num_features}, Classes: {num_classes}")

        for model_name in models_config:
            accs = []

            for run in range(N_RUNS):
                torch.manual_seed(run * 42)
                np.random.seed(run * 42)

                # Crear modelo
                if model_name == 'MLP':
                    model = MLP(num_features, HIDDEN_DIM, num_classes).to(DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                    is_mlp = True

                elif model_name == 'GCN':
                    model = GCN(num_features, HIDDEN_DIM, num_classes).to(DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                    is_mlp = False

                elif model_name == 'GAT':
                    model = GAT(num_features, HIDDEN_DIM, num_classes).to(DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
                    is_mlp = False

                elif model_name == 'AdaptiveMixGNN':
                    model = AdaptiveMixGNN(
                        num_features=num_features,
                        hidden_dim=HIDDEN_DIM,
                        num_classes=num_classes,
                        num_layers=2,
                        edge_index=data.edge_index,
                        num_nodes=data.num_nodes,
                        device=DEVICE,
                        dropout=0.5
                    ).to(DEVICE)
                    optimizer = get_optimizer(model, lr=0.01, weight_decay=5e-4)
                    is_mlp = False

                # Entrenar
                test_acc = run_experiment(
                    model, optimizer, data,
                    N_EPOCHS, PATIENCE, WARMUP_EPOCHS,
                    is_mlp=is_mlp
                )
                accs.append(test_acc)

            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            results[dataset_name][model_name] = (mean_acc, std_acc)

            print(f"  {model_name:20s}: {mean_acc:.2f} ± {std_acc:.2f}%")

    return results


def print_latex_table(results):
    """Genera tabla en formato LaTeX."""
    print("\n" + "="*70)
    print("TABLA LATEX")
    print("="*70)

    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())

    # Header
    print("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    print("\\toprule")
    print("Modelo & " + " & ".join(datasets) + " \\\\")
    print("\\midrule")

    # Rows
    for model in models:
        row = model
        for ds in datasets:
            mean, std = results[ds][model]
            row += f" & {mean:.2f} ± {std:.2f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")


def print_markdown_table(results):
    """Genera tabla en formato Markdown."""
    print("\n" + "="*70)
    print("TABLA MARKDOWN")
    print("="*70)

    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())

    # Header
    print("| Modelo | " + " | ".join(datasets) + " |")
    print("|--------|" + "|".join(["--------"] * len(datasets)) + "|")

    # Rows
    for model in models:
        row = f"| {model} |"
        for ds in datasets:
            mean, std = results[ds][model]
            row += f" {mean:.2f} ± {std:.2f} |"
        print(row)


if __name__ == "__main__":
    results = run_benchmark()
    print_markdown_table(results)
    print_latex_table(results)

    # Guardar resultados
    print("\n" + "="*70)
    print("ANÁLISIS")
    print("="*70)

    for ds in results:
        print(f"\n{ds}:")
        mlp_acc = results[ds]['MLP'][0]
        gcn_acc = results[ds]['GCN'][0]
        gat_acc = results[ds]['GAT'][0]
        our_acc = results[ds]['AdaptiveMixGNN'][0]

        print(f"  - MLP vs GCN: {'GCN mejor' if gcn_acc > mlp_acc else 'MLP mejor'} ({gcn_acc:.2f} vs {mlp_acc:.2f})")
        print(f"  - AdaptiveMixGNN vs GCN: {'Nuestro mejor' if our_acc > gcn_acc else 'GCN mejor'} ({our_acc:.2f} vs {gcn_acc:.2f})")
        print(f"  - AdaptiveMixGNN vs GAT: {'Nuestro mejor' if our_acc > gat_acc else 'GAT mejor'} ({our_acc:.2f} vs {gat_acc:.2f})")
