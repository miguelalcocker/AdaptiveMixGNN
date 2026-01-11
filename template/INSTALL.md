# Installation Guide for AdaptiveMixGNN

## Quick Start (Recommended)

### Option 1: Using Conda (Easiest)

```bash
# Create new environment
conda create -n adaptivemix python=3.10 -y
conda activate adaptivemix

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
pip install wandb matplotlib seaborn pandas networkx
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other requirements
pip install -r requirements.txt
```

### Option 3: GPU Support

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
```

## Test the Model

```bash
# Run validation tests (no dataset download required)
python test_model.py

# If successful, try training
python train_example.py --dataset Cora --epochs 10 --log_alpha --verbose
```

## Troubleshooting

### "No module named 'torch_geometric'"
- Make sure you installed PyTorch Geometric: `pip install torch-geometric`
- For conda: `conda install pyg -c pyg`

### "No module named 'torch_scatter'"
- Install PyG extensions: `pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`

### CUDA version mismatch
- Check your CUDA version: `nvcc --version` or `nvidia-smi`
- Install matching PyTorch/PyG versions (see PyG installation guide)

### Import errors during test
- Ensure all dependencies from requirements.txt are installed
- Try: `pip install -r requirements.txt`

## Minimal Installation (Testing Only)

If you just want to test the model without training:

```bash
pip install torch torch-geometric
python test_model.py
```

## Using Existing Environment

If you already have the `rl-unrolling` environment set up:

```bash
conda activate rl-unrolling  # or your environment name
pip install torch-geometric pyg_lib torch_scatter torch_sparse
```

## Next Steps

After successful installation:

1. **Quick test**: `python test_model.py`
2. **Training**: `python train_example.py --dataset Cora --log_alpha --verbose`
3. **Ablation study**: See README.md for detailed commands

## Resources

- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Issues: Open an issue in the repository if you encounter problems
