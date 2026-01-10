"""
AdaptiveMixGNN: Filter Bank GNN for Heterophilic Graphs

Implementation of a node classification model based on Graph Signal Processing (GSP)
principles using separate low-pass and high-pass graph filters.

Mathematical Specification:
    X_l = σ(α^(l) * S_LP * X_{l-1} * W_LP^(l) + (1-α^(l)) * S_HP * X_{l-1} * W_HP^(l) + b^(l))

Where:
    - S_LP: Low-pass graph shift operator (GCN normalized adjacency)
    - S_HP: High-pass graph shift operator (I - S_LP)
    - α^(l): Learnable mixing parameter per layer [0,1]
    - W_LP, W_HP: Learnable weight matrices
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
    indices_LP = edge_index_with_loops
    values_LP = edge_weight
    S_LP = torch.sparse_coo_tensor(
        indices_LP,
        values_LP,
        size=(num_nodes, num_nodes),
        device=device
    )

    # Create identity matrix as sparse COO tensor
    identity_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(device)
    identity_values = torch.ones(num_nodes, device=device)
    I = torch.sparse_coo_tensor(
        identity_indices,
        identity_values,
        size=(num_nodes, num_nodes),
        device=device
    )

    # Compute S_HP = I - S_LP (sparse subtraction)
    # Convert to dense for subtraction, then back to sparse
    S_HP = I.to_dense() - S_LP.to_dense()
    S_HP_indices = S_HP.nonzero().t()
    S_HP_values = S_HP[S_HP_indices[0], S_HP_indices[1]]
    S_HP = torch.sparse_coo_tensor(
        S_HP_indices,
        S_HP_values,
        size=(num_nodes, num_nodes),
        device=device
    )

    return S_LP.coalesce(), S_HP.coalesce()


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

    Implements: X_out = σ(α * S_LP * X * W_LP + (1-α) * S_HP * X * W_HP + b)
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

        # Learnable mixing parameter α (unconstrained, will apply sigmoid)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

        # Learnable weight matrices for low-pass and high-pass branches
        self.W_LP = nn.Linear(in_features, out_features, bias=False)
        self.W_HP = nn.Linear(in_features, out_features, bias=False)

        # Shared bias term
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Initialize weights
        nn.init.xavier_uniform_(self.W_LP.weight)
        nn.init.xavier_uniform_(self.W_HP.weight)

    @property
    def alpha(self):
        """Get alpha value constrained to [0,1] via sigmoid."""
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x):
        """
        Forward pass with adaptive frequency mixing.

        Args:
            x: Input features [num_nodes, in_features]

        Returns:
            Output features [num_nodes, out_features]
        """
        # Low-pass branch: S_LP * X * W_LP
        x_lp = torch.sparse.mm(self.S_LP, x)  # [N, F_in]
        x_lp = self.W_LP(x_lp)                 # [N, F_out]

        # High-pass branch: S_HP * X * W_HP
        x_hp = torch.sparse.mm(self.S_HP, x)  # [N, F_in]
        x_hp = self.W_HP(x_hp)                 # [N, F_out]

        # Get current alpha value (constrained to [0,1])
        alpha_val = self.alpha

        # Adaptive mixing: α * low_pass + (1-α) * high_pass
        out = alpha_val * x_lp + (1 - alpha_val) * x_hp

        # Add bias
        out = out + self.bias

        # Apply activation (ReLU) if not last layer
        if self.use_activation:
            out = F.relu(out)

        return out


class AdaptiveMixGNN(nn.Module):
    """
    AdaptiveMixGNN: Multi-layer Filter Bank GNN for Node Classification.

    Adaptively balances homophily (low-pass) and heterophily (high-pass)
    signal propagation through learnable per-layer mixing parameters.

    Architecture:
        - Pre-computes S_LP and S_HP once in __init__
        - Each layer learns α^(l) to balance frequency components
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
        ablation_mode=None
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
            ablation_mode: If 'gcn', force α=1; if 'hp', force α=0; else learnable
        """
        super(AdaptiveMixGNN, self).__init__()

        assert edge_index is not None, "edge_index must be provided"
        assert num_nodes is not None, "num_nodes must be provided"

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.ablation_mode = ablation_mode

        # Pre-compute graph shift operators (efficiency)
        self.S_LP, self.S_HP = compute_graph_shift_operators(
            edge_index, num_nodes, device
        )

        # Build layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            AdaptiveMixGNNLayer(
                num_features,
                hidden_dim,
                self.S_LP,
                self.S_HP,
                use_activation=True
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                AdaptiveMixGNNLayer(
                    hidden_dim,
                    hidden_dim,
                    self.S_LP,
                    self.S_HP,
                    use_activation=True
                )
            )

        # Last layer (no activation - output logits)
        self.layers.append(
            AdaptiveMixGNNLayer(
                hidden_dim,
                num_classes,
                self.S_LP,
                self.S_HP,
                use_activation=False
            )
        )

    def forward(self, x, edge_index=None):
        """
        Forward pass through all layers.

        Args:
            x: Input features [num_nodes, num_features]
            edge_index: Edge index (not used, kept for compatibility)

        Returns:
            Logits [num_nodes, num_classes]
        """
        # Apply ablation mode if specified
        if self.ablation_mode == 'gcn':
            # Force all alphas to 1 (pure low-pass/GCN)
            for layer in self.layers:
                layer.alpha_raw.data.fill_(10.0)  # sigmoid(10) ≈ 1
        elif self.ablation_mode == 'hp':
            # Force all alphas to 0 (pure high-pass)
            for layer in self.layers:
                layer.alpha_raw.data.fill_(-10.0)  # sigmoid(-10) ≈ 0

        # Forward through all layers
        for layer in self.layers:
            x = layer(x)

        return x

    def get_alpha_values(self):
        """
        Get current alpha values for all layers (for logging/analysis).

        Returns:
            List of alpha values (post-sigmoid) for each layer
        """
        return [layer.alpha.item() for layer in self.layers]

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
        print(f"  - Ablation mode:     {self.ablation_mode if self.ablation_mode else 'None (learnable α)'}")
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
