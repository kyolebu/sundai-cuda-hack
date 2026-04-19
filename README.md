# CUDA Challenge: W4A4 Quantized GEMM

Optimize INT4 quantization and GEMM kernels for real FLUX.1-schnell model layers.

**See [GETTING_STARTED.md](GETTING_STARTED.md) for setup instructions and how to begin.**

## The Challenge

You are given FP16 activation and weight tensors from a FLUX.1-schnell diffusion model:

| Layer | M | N | K |
|---|---|---|---|
| attn_to_qkv | 4096 | 9216 | 3072 |
| attn_to_out | 4096 | 3072 | 3072 |
| ff_up | 4096 | 12288 | 3072 |
| ff_down | 4096 | 3072 | 12288 |

Your job:

1. **Offline weight quantization** (Python) -- convert FP16 weights to packed INT4. Not timed.
2. **Online forward pass** (CUDA) -- quantize FP16 activations to INT4 and compute the INT4 GEMM. **This is timed.**

## Rules

- **Edit only** `your_solution/quantize.py` and `your_solution/kernel.cu`
- **Do not modify** `reference/`, `benchmark.py`, or `benchmark.sh`
- **External libraries allowed** -- cuBLAS, CUTLASS, Thrust, CUB, etc.
- **Must pass correctness** -- cosine similarity vs FP16 matmul must exceed per-layer thresholds
- **C++ wrapper signatures are fixed** -- do not change `quantize_int4_custom` or `gemm_int4_custom`

## Scoring

Your score is the **average GEMM TOPs** across all 4 target shapes. Higher is better.

Baselines below are measured on an **NVIDIA RTX A6000** (Ampere, INT4 dense peak ~619 TOPS):

| Baseline | Avg GEMM TOPs | |
|---|---|---|
| Naive SIMT | ~1.1 | Starting point |
| MMA starter | ~58 | Copy from `reference/gemm_int4_mma.cu` |
| Optimized (nunchaku) | ~305 | Target to beat |

## How to run it

We already preinstalled conda on each of the servers and configured the environment called `cuda-challenge`.
```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate cuda-challenge
```

At first download the data. Make sure the python that you are using is from the `cuda-challenge` environment. 
```bash
python download_data.py
```

Then run the benchmark.
```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate cuda-challenge && python benchmark.py
``` 

## Repository Structure

```
cuda-challenge/
  GETTING_STARTED.md    # How to set up and begin
  download_data.py      # Downloads benchmark tensors from Hugging Face
  benchmark.py          # Correctness + performance measurement
  benchmark.sh          # Entry point for benchmarking
  reference/
    quantize.py         # Reference offline weight quantization (read-only)
    quantize_int4.cu    # Naive SIMT quantization kernel (read-only)
    gemm_int4.cu        # Naive SIMT GEMM kernel (~1 GB/s)
    gemm_int4_mma.cu    # MMA-based GEMM starting code (~63 GB/s)
    pybind.cpp          # Python bindings
  your_solution/
    quantize.py         # YOUR offline weight quantization
    kernel.cu           # YOUR CUDA kernels
    pybind.cpp          # Python bindings
  optimized/            # High-performance nunchaku-based baseline
  flux_dump/            # Generated data (run download_data.py)
```
