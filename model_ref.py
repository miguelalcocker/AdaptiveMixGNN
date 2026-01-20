"""
AdaptiveMixGNN: Filter Bank GNN for Heterophilic Graphs

Implementation of a node classification model based on Graph Signal Processing (GSP)
principles using separate low-pass and high-pass graph filters with node-wise α.

Mathematical Specification:
    α_i = sigmoid(x_i · θ + b)
    z_mix_i = α_i * (S_LP @ X)_i + (1-α_i) * (S_HP @ X)_i
    X_l = σ(W @ z_mix + bias)

Where:
    - S_LP: Low-pass graph shift operator (GCN normalized adjacency)
    - S_HP: High-pass graph shift operator (I - S_LP)
    - α_i: Node-wise learnable mixing parameter [0,1]
    - W: Shared weight matrix (applied after mixing)
    - σ: ReLU activation (except last layer)

Reference: Filter Bank GNNs for heterophily-aware graph learning
Workshop: GRaM @ ICLR 2026 (Tiny Papers Track)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
import csv
from pathlib import Path


def compute_graph_shift_operators(edge_index, num_nodes, device):
    """
    Pre-compute low-pass (S_LP) and high-pass (S_HP) graph shift operators.

    S_LP = D^(-1/2) * (A + I) * D^(-1/2)  [GCN normalization]
    S_HP = I - S_LP                        [High-pass filter]

    Complexity: O(E) instead of O(N²) - critical for large graphs.

    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes in the graph
        device: Device to place tensors on

    Returns:
        S_LP: Low-pass GSO as sparse COO tensor
        S_HP: High-pass GSO as sparse COO tensor
    """
    # Add self-loops: A_tilde = A + I
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Compute degree matrix D_tilde
    row, col = edge_index_with_loops
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Compute normalized adjacency: D^(-1/2) * A_tilde * D^(-1/2)
    # For edge (i,j): value = deg_inv_sqrt[i] * deg_inv_sqrt[j]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Create S_LP as sparse COO tensor
    S_LP = torch.sparse_coo_tensor(
        edge_index_with_loops,
        edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()

    # Compute S_HP = I - S_LP efficiently (same sparsity pattern)
    # S_HP[i,i] = 1 - S_LP[i,i] (diagonal)
    # S_HP[i,j] = -S_LP[i,j] (off-diagonal)
    S_HP_values = -edge_weight.clone()
    diag_mask = (row == col)
    S_HP_values[diag_mask] = 1.0 - edge_weight[diag_mask]

    S_HP = torch.sparse_coo_tensor(
        edge_index_with_loops,
        S_HP_values,
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()

    return S_LP, S_HP


def count_parameters(model):
    """
    Count total trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AdaptiveMixGNNLayer(nn.Module):
    """
    Single layer of AdaptiveMixGNN with adaptive frequency mixing.

    Implements: X_out = σ(W * (α * S_LP * X + (1-α) * S_HP * X) + b)

    Where α_i = sigmoid(x_i · θ) is computed per-node based on features.
    """
    def __init__(self, in_features, out_features, S_LP, S_HP, use_activation=True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            S_LP: Pre-computed low-pass graph shift operator (sparse)
            S_HP: Pre-computed high-pass graph shift operator (sparse)
            use_activation: Whether to apply ReLU activation (False for last layer)
        """
        super(AdaptiveMixGNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_activation = use_activation

        # Store pre-computed GSOs
        self.register_buffer('S_LP', S_LP)
        self.register_buffer('S_HP', S_HP)

        # Node-wise alpha predictor: α_i = sigmoid(x_i · θ + b)
        self.alpha_predictor = nn.Linear(in_features, 1, bias=True)

        # Shared weight matrix (applied after mixing)
        self.W = nn.Linear(in_features, out_features, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.alpha_predictor.weight)
        nn.init.zeros_(self.alpha_predictor.bias)

    def forward(self, x):
        """
        Forward pass with adaptive frequency mixing.

        Args:
            x: Input features [num_nodes, in_features]

        Returns:
            out: Output features [num_nodes, out_features]
            alpha: Per-node mixing coefficients [num_nodes, 1]
        """
        # Compute node-wise alpha
        alpha = torch.sigmoid(self.alpha_predictor(x))  # [N, 1]

        # Low-pass branch: S_LP * X
        z_lp = torch.sparse.mm(self.S_LP, x)  # [N, F_in]

        # High-pass branch: S_HP * X
        z_hp = torch.sparse.mm(self.S_HP, x)  # [N, F_in]

        # Adaptive mixing: α * low_pass + (1-α) * high_pass
        z_mix = alpha * z_lp + (1 - alpha) * z_hp  # [N, F_in]

        # Apply shared transformation
        out = self.W(z_mix)  # [N, F_out]

        # Apply activation (ReLU) if not last layer
        if self.use_activation:
            out = F.relu(out)

        return out, alpha


class AdaptiveMixGNN(nn.Module):
    """
    AdaptiveMixGNN: Multi-layer Filter Bank GNN for Node Classification.

    Adaptively balances homophily (low-pass) and heterophily (high-pass)
    signal propagation through learnable per-node mixing parameters.

    Architecture:
        - Pre-computes S_LP and S_HP once in __init__
        - Each layer learns α_i = sigmoid(x_i · θ) per node
        - Last layer outputs logits (no activation)
    """
    def __init__(
        self,
        num_features,
        hidden_dim,
        num_classes,
        num_layers=2,
        edge_index=None,
        num_nodes=None,
        device='cpu',
        dropout=0.5
    ):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GNN layers
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            device: Device to place model on
            dropout: Dropout probability
        """
        super(AdaptiveMixGNN, self).__init__()

        assert edge_index is not None, "edge_index must be provided"
        assert num_nodes is not None, "num_nodes must be provided"

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute graph shift operators (efficiency)
        self.S_LP, self.S_HP = compute_graph_shift_operators(
            edge_index, num_nodes, device
        )

        # Build layers
        self.layers = nn.ModuleList()
        dims = [num_features] + [hidden_dim] * (num_layers - 1) + [num_classes]

        for i in range(num_layers):
            self.layers.append(
                AdaptiveMixGNNLayer(
                    dims[i],
                    dims[i + 1],
                    self.S_LP,
                    self.S_HP,
                    use_activation=(i < num_layers - 1)
                )
            )

    def forward(self, x, edge_index=None, return_alpha=False):
        """
        Forward pass through all layers.

        Args:
            x: Input features [num_nodes, num_features]
            edge_index: Edge index (not used, kept for compatibility)
            return_alpha: If True, also return alpha values

        Returns:
            Logits [num_nodes, num_classes]
            (optionally) List of alpha tensors per layer
        """
        alphas = []

        # Forward through all layers
        for i, layer in enumerate(self.layers):
            x, alpha = layer(x)
            alphas.append(alpha)
            if i < len(self.layers) - 1:
                x = self.dropout(x)

        if return_alpha:
            return x, alphas
        return x

    def get_alpha_values(self, x=None):
        """
        Get current alpha values for all layers (for logging/analysis).

        Args:
            x: Input features (required for node-wise alpha computation)

        Returns:
            List of mean alpha values for each layer
        """
        if x is not None:
            with torch.no_grad():
                _, alphas = self.forward(x, return_alpha=True)
                return [alpha.mean().item() for alpha in alphas]
        return [0.5] * self.num_layers

    def print_model_info(self):
        """
        Print model architecture and parameter count (Simplicity criterion).
        """
        total_params = count_parameters(self)
        print("\n" + "="*60)
        print("AdaptiveMixGNN Model Information")
        print("="*60)
        print(f"Architecture:")
        print(f"  - Input features:    {self.num_features}")
        print(f"  - Hidden dimension:  {self.hidden_dim}")
        print(f"  - Output classes:    {self.num_classes}")
        print(f"  - Number of layers:  {self.num_layers}")
        print(f"\nParameters:")
        print(f"  - Total trainable:   {total_params:,}")
        print("="*60 + "\n")

        return total_params


class AlphaLogger:
    """
    Utility for logging alpha evolution during training.

    Saves alpha values to CSV for analysis and visualization in the paper.
    """
    def __init__(self, save_path='alpha_evolution.csv', num_layers=2):
        """
        Args:
            save_path: Path to save CSV file
            num_layers: Number of layers in the model
        """
        self.save_path = Path(save_path)
        self.num_layers = num_layers

        # Initialize CSV file with headers
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['epoch'] + [f'alpha_layer_{i}' for i in range(num_layers)]
            writer.writerow(headers)

    def log(self, epoch, alpha_values):
        """
        Log alpha values for current epoch.

        Args:
            epoch: Current epoch number
            alpha_values: List of alpha values (one per layer)
        """
        with open(self.save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch] + alpha_values
            writer.writerow(row)

    def __repr__(self):
        return f"AlphaLogger(save_path='{self.save_path}', num_layers={self.num_layers})"


# Utility function for quick parameter counting
def print_model_parameters(model):
    """
    Print detailed parameter breakdown for transparency.

    Args:
        model: AdaptiveMixGNN model instance
    """
    print("\nParameter Breakdown:")
    print("-" * 60)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:40s} {param.numel():>10,} params")
    print("-" * 60)
    total = count_parameters(model)
    print(f"{'TOTAL':40s} {total:>10,} params")
    print("-" * 60 + "\n")
    return total


def get_optimizer(model, lr=0.01, weight_decay=5e-4):
    """
    Optimizer with differential learning rates for α parameters.

    Args:
        model: AdaptiveMixGNN model instance
        lr: Base learning rate
        weight_decay: Weight decay for non-alpha parameters

    Returns:
        Configured Adam optimizer
    """
    alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
    other_params = [p for n, p in model.named_parameters() if 'alpha' not in n]

    return torch.optim.Adam([
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': alpha_params, 'lr': lr * 0.1, 'weight_decay': 0.0}
    ])
