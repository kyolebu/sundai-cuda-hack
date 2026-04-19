# Getting Started

## 1. Download Benchmark Data

```bash
python download_data.py
```

Pulls ~390 MB of activation/weight tensors from [`nunchaku-ai/cuda-challenge-flux-dump`](https://huggingface.co/datasets/nunchaku-ai/cuda-challenge-flux-dump) into `flux_dump/`. `./benchmark.sh` will also do this automatically on first run.

The 4 target GEMM shapes:

```
  attn_to_qkv       4096 x  9216 x  3072
  attn_to_out       4096 x  3072 x  3072
  ff_up             4096 x 12288 x  3072
  ff_down           4096 x  3072 x 12288
```

## 2. Run the Benchmark

```bash
./benchmark.sh
```

This compiles your CUDA kernels, runs correctness checks (per-layer cosine thresholds), and reports throughput in TOPs. On a fresh start with the naive kernel you should see ~1.1 TOPs GEMM and 1.00x speedup on an RTX A6000.

## 3. Edit Your Solution

You submit two files:

**`your_solution/kernel.cu`** -- CUDA kernels for online activation quantization + GEMM.

Two C++ wrapper signatures are fixed (do not change these):
```cpp
std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size);
torch::Tensor gemm_int4_custom(torch::Tensor A_packed, torch::Tensor B_packed,
                                torch::Tensor scales_A, torch::Tensor scales_B, int group_size);
```

**`your_solution/quantize.py`** -- Python script for offline weight quantization (not timed):
```python
def quantize_weights(weight: torch.Tensor, group_size: int = 64) -> dict:
    # Returns: {"weight_packed": [N, K//2] uint8,
    #           "weight_scales": [N, K//group_size] float16,
    #           "group_size": int}
```

## 4. Choose a Starting Point

Two reference GEMM kernels are provided in `reference/`:

| File | Approach | ~TOPs (A6000) | |
|---|---|---|---|
| `gemm_int4.cu` | Naive SIMT | ~1.1 | One thread per output element |
| `gemm_int4_mma.cu` | MMA + cp.async | ~58 | Tensor core m16n8k64, double-buffered shared memory |

Your `kernel.cu` starts as a copy of the naive version. To use the MMA version as your starting point, copy the GEMM function from `reference/gemm_int4_mma.cu` into your `kernel.cu` (replacing `gemm_int4_kernel` and `gemm_int4_custom`).

## Quantization Format

Both the Python offline quantizer and the CUDA online quantizer produce the same packed format:

- **Packed data**: `uint8 [rows, K/2]` -- two signed INT4 values per byte (low nibble = even element, high nibble = odd element)
- **Scales**: `float16 [rows, K/group_size]` -- one scale per group
- **Algorithm**: symmetric per-group quantization, `scale = max(|x|) / 7`, round-to-nearest, clamp to [-8, 7]

The GEMM kernel computes `C[M,N] = A[M,K] @ B[N,K]^T` where both A (activation) and B (weight) are in this packed format.

## Accuracy Reference

Cosine similarity vs FP16 matmul, measured with nunchaku quantization kernels on real FLUX.1-schnell weights:

| | attn_to_qkv | attn_to_out | ff_up | ff_down |
|---|---|---|---|---|
| | 4096×9216×3072 | 4096×3072×3072 | 4096×12288×3072 | 4096×3072×12288 |
| INT4 g=128 | 0.9870 | 0.9905 | 0.9745 | 0.9714 |
| INT4 g=64 | 0.9910 | 0.9921 | 0.9819 | 0.9822 |
| **Threshold** | **0.989** | **0.991** | **0.978** | **0.977** |

Per-layer thresholds are set between g=128 and g=64 accuracy. Your solution must pass all 4.

## Tips

- The naive kernel's bottleneck is the per-element GEMM loop -- tensor cores (MMA) give ~60x speedup out of the box
- After MMA, the next wins come from tiling, shared memory reuse, warp-level scale broadcasting, and output register transposition
- `cuBLAS`, `CUTLASS`, `Thrust`, `CUB` are all fair game
- Run `./benchmark.sh` often -- it recompiles automatically
