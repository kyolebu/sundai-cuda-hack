#!/bin/bash
set -e
cd "$(dirname "$0")"

ENV_NAME="cuda-challenge"

# Activate conda environment
if conda info --envs 2>/dev/null | grep -q "$ENV_NAME"; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "ERROR: Conda environment '$ENV_NAME' not found. Run 'conda activate cuda-challenge' first."
    exit 1
fi

if [ ! -f flux_dump/weights.pt ] || [ ! -f flux_dump/activations_1024x1024.pt ]; then
    echo "flux_dump/ missing - downloading from Hugging Face..."
    python download_data.py
fi

python benchmark.py "$@"
