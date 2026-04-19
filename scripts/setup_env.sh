#!/bin/bash
set -e

echo "=== NeMo EMR De-identification Setup ==="

# Check Python
python3 --version || { echo "Python3 not found"; exit 1; }

# Check GPU
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create virtual environment
echo "--- Creating virtual environment ---"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.x for Blackwell/DGX Spark)
echo "--- Installing PyTorch ---"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install NeMo base + other dependencies
# NOTE: nemo_toolkit[nlp] requires mamba-ssm which needs CUDA toolkit headers.
# For full NLP support, use the official NeMo Docker container (recommended for DGX Spark):
#   docker pull nvcr.io/nvidia/nemo:24.09
# For local dev without Docker, we install base nemo + NLP deps individually below.
echo "--- Installing NeMo base ---"
pip install nemo_toolkit==2.1.0

echo "--- Installing NLP dependencies ---"
pip install transformers sentencepiece lightning

echo "--- Installing API and utility dependencies ---"
pip install fastapi==0.115.0 "uvicorn[standard]==0.30.0" pydantic==2.9.0 \
    python-dotenv==1.0.0 PyYAML==6.0.2 "tqdm>=4.66.3" \
    "datasets>=3.0.0" "pandas==2.2.0"

# Verify NeMo install
echo "--- Verifying NeMo ---"
python3 -c "import nemo; print(f'NeMo version: {nemo.__version__}')"

echo ""
echo "=== Setup complete ==="
echo "Run: source venv/bin/activate"
echo ""
echo "For full NLP support (recommended on DGX Spark):"
echo "  docker pull nvcr.io/nvidia/nemo:24.09"
