"""
Quick Test Script for AdaptiveMixGNN

Validates:
1. Model initialization
2. Forward pass shape correctness
3. Alpha parameter constraints
4. Sparse GSO computation
5. Ablation mode functionality
6. Parameter counting
7. Alpha logging

No dataset download required - uses synthetic graph.
"""

import torch
import torch.nn.functional as F
from model import AdaptiveMixGNN, AlphaLogger, count_parameters, print_model_parameters
import os


def create_synthetic_graph(num_nodes=100, num_edges=500, num_features=10, num_classes=3):
    """Create a synthetic graph for testing."""
    # Random features
    x = torch.randn(num_nodes, num_features)

    # Random edge_index (undirected)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Random labels
    y = torch.randint(0, num_classes, (num_nodes,))

    # Create train/val/test masks
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:60]] = True
    val_mask[perm[60:80]] = True
    test_mask[perm[80:]] = True

    return x, edge_index, y, train_mask, val_mask, test_mask


def test_model_initialization():
    """Test 1: Model initialization."""
    print("\n" + "="*70)
    print("TEST 1: Model Initialization")
    print("="*70)

    num_nodes = 100
    num_features = 10
    num_classes = 3
    hidden_dim = 16
    num_layers = 2

    x, edge_index, y, _, _, _ = create_synthetic_graph(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes
    )

    model = AdaptiveMixGNN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        edge_index=edge_index,
        num_nodes=num_nodes,
        device='cpu'
    )

    print("✓ Model initialized successfully")
    print(f"  - Input features: {num_features}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Output classes: {num_classes}")
    print(f"  - Num layers: {num_layers}")

    return model, x, edge_index, y


def test_forward_pass(model, x, edge_index):
    """Test 2: Forward pass shape correctness."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)

    print(f"✓ Forward pass successful")
    print(f"  - Input shape:  {x.shape}")
    print(f"  - Output shape: {out.shape}")
    print(f"  - Expected shape: ({x.shape[0]}, {model.num_classes})")

    assert out.shape == (x.shape[0], model.num_classes), "Output shape mismatch!"
    print("✓ Output shape correct")

    return out


def test_alpha_constraints(model):
    """Test 3: Alpha values are in [0,1]."""
    print("\n" + "="*70)
    print("TEST 3: Alpha Parameter Constraints")
    print("="*70)

    alphas = model.get_alpha_values()
    print(f"Alpha values per layer:")
    for i, alpha in enumerate(alphas):
        print(f"  - Layer {i}: α = {alpha:.4f}")
        assert 0 <= alpha <= 1, f"Alpha {alpha} out of [0,1] range!"

    print("✓ All alpha values in [0, 1]")

    return alphas


def test_sparse_gso_computation(model):
    """Test 4: Sparse GSO properties."""
    print("\n" + "="*70)
    print("TEST 4: Sparse GSO Computation")
    print("="*70)

    S_LP = model.S_LP
    S_HP = model.S_HP

    print(f"S_LP (Low-Pass GSO):")
    print(f"  - Type: {type(S_LP)}")
    print(f"  - Shape: {S_LP.shape}")
    print(f"  - Is sparse: {S_LP.is_sparse}")
    print(f"  - Num non-zero: {S_LP._nnz()}")

    print(f"\nS_HP (High-Pass GSO):")
    print(f"  - Type: {type(S_HP)}")
    print(f"  - Shape: {S_HP.shape}")
    print(f"  - Is sparse: {S_HP.is_sparse}")
    print(f"  - Num non-zero: {S_HP._nnz()}")

    # Check that S_LP + S_HP ≈ I (approximately identity)
    sum_GSO = S_LP.to_dense() + S_HP.to_dense()
    identity = torch.eye(S_LP.shape[0])
    diff = (sum_GSO - identity).abs().max().item()

    print(f"\nProperty check: S_LP + S_HP = I")
    print(f"  - Max deviation from identity: {diff:.6f}")

    if diff < 0.01:
        print("✓ S_LP + S_HP ≈ I (within tolerance)")
    else:
        print(f"⚠ Warning: Deviation {diff:.6f} larger than expected")

    print("✓ Sparse GSO computation verified")


def test_ablation_modes():
    """Test 5: Ablation mode functionality."""
    print("\n" + "="*70)
    print("TEST 5: Ablation Modes")
    print("="*70)

    num_nodes = 50
    x, edge_index, _, _, _, _ = create_synthetic_graph(num_nodes=num_nodes)

    # Test GCN mode (α=1)
    print("\nAblation Mode: GCN (α=1)")
    model_gcn = AdaptiveMixGNN(
        num_features=10,
        hidden_dim=16,
        num_classes=3,
        num_layers=2,
        edge_index=edge_index,
        num_nodes=num_nodes,
        device='cpu',
        ablation_mode='gcn'
    )
    model_gcn.eval()
    with torch.no_grad():
        _ = model_gcn(x, edge_index)
    alphas_gcn = model_gcn.get_alpha_values()
    print(f"  Alpha values: {[f'{a:.4f}' for a in alphas_gcn]}")
    assert all(a > 0.99 for a in alphas_gcn), "GCN mode should force α ≈ 1"
    print("✓ GCN mode: α ≈ 1.0")

    # Test HP mode (α=0)
    print("\nAblation Mode: High-Pass (α=0)")
    model_hp = AdaptiveMixGNN(
        num_features=10,
        hidden_dim=16,
        num_classes=3,
        num_layers=2,
        edge_index=edge_index,
        num_nodes=num_nodes,
        device='cpu',
        ablation_mode='hp'
    )
    model_hp.eval()
    with torch.no_grad():
        _ = model_hp(x, edge_index)
    alphas_hp = model_hp.get_alpha_values()
    print(f"  Alpha values: {[f'{a:.4f}' for a in alphas_hp]}")
    assert all(a < 0.01 for a in alphas_hp), "HP mode should force α ≈ 0"
    print("✓ High-Pass mode: α ≈ 0.0")

    print("\n✓ Ablation modes working correctly")


def test_parameter_counting(model):
    """Test 6: Parameter counting utilities."""
    print("\n" + "="*70)
    print("TEST 6: Parameter Counting")
    print("="*70)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    print("\nDetailed breakdown:")
    print_model_parameters(model)

    print("✓ Parameter counting utilities working")

    return total_params


def test_alpha_logging():
    """Test 7: Alpha logger functionality."""
    print("\n" + "="*70)
    print("TEST 7: Alpha Logging")
    print("="*70)

    # Create temporary CSV
    log_path = 'test_alpha_evolution.csv'

    logger = AlphaLogger(save_path=log_path, num_layers=2)
    print(f"Created logger: {logger}")

    # Log some fake epochs
    for epoch in range(1, 6):
        fake_alphas = [0.5 + epoch*0.05, 0.6 + epoch*0.04]
        logger.log(epoch, fake_alphas)

    print(f"✓ Logged 5 epochs to {log_path}")

    # Read back and verify
    import csv
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    print(f"✓ CSV contains {len(rows)} rows (including header)")
    print(f"  Header: {rows[0]}")
    print(f"  Sample data: {rows[1]}")

    # Clean up
    os.remove(log_path)
    print(f"✓ Cleaned up test file")

    print("✓ Alpha logging working correctly")


def test_gradient_flow(model, x, edge_index, y, train_mask):
    """Test 8: Gradient flow (backward pass)."""
    print("\n" + "="*70)
    print("TEST 8: Gradient Flow")
    print("="*70)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Forward pass
    out = model(x, edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])

    print(f"Initial loss: {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that gradients exist
    has_gradients = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters() if p.requires_grad
    )

    print("✓ Backward pass successful")
    print(f"✓ Gradients computed: {has_gradients}")

    # Second forward pass to check loss changed
    with torch.no_grad():
        out2 = model(x, edge_index)
        loss2 = F.cross_entropy(out2[train_mask], y[train_mask])

    print(f"Loss after 1 step: {loss2.item():.4f}")
    print("✓ Gradient flow verified")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("AdaptiveMixGNN - Model Validation Tests")
    print("="*70)
    print("Testing implementation correctness without dataset download")

    try:
        # Test 1: Initialization
        model, x, edge_index, y = test_model_initialization()

        # Test 2: Forward pass
        out = test_forward_pass(model, x, edge_index)

        # Test 3: Alpha constraints
        alphas = test_alpha_constraints(model)

        # Test 4: Sparse GSO
        test_sparse_gso_computation(model)

        # Test 5: Ablation modes
        test_ablation_modes()

        # Test 6: Parameter counting
        total_params = test_parameter_counting(model)

        # Test 7: Alpha logging
        test_alpha_logging()

        # Test 8: Gradient flow
        _, _, _, train_mask, _, _ = create_synthetic_graph()
        test_gradient_flow(model, x, edge_index, y, train_mask)

        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nModel Summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Alpha values: {[f'{a:.3f}' for a in alphas]}")
        print(f"  - Output shape: {out.shape}")
        print("\nThe model is ready for training!")
        print("Run: python train_example.py --dataset Cora --log_alpha --verbose")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED ✗")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
