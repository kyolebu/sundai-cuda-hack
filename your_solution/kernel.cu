#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// =====================================================================
// INT4 Quantization Kernel (unchanged from baseline)
// =====================================================================

__global__ void quantize_int4_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int M,
    int K,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;

    if (row >= M) return;

    int num_groups = K / group_size;
    int k_start = group * group_size;

    float max_abs = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float val = __half2float(input[row * K + k_start + i]);
        float abs_val = fabsf(val);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    float scale = max_abs / 7.0f;
    scales[row * num_groups + group] = __float2half(scale);

    float rscale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    int out_offset = row * (K / 2) + k_start / 2;
    for (int i = 0; i < group_size; i += 2) {
        float val_even = __half2float(input[row * K + k_start + i]);
        float val_odd  = __half2float(input[row * K + k_start + i + 1]);

        int q_even = __float2int_rn(val_even * rscale);
        int q_odd  = __float2int_rn(val_odd * rscale);

        q_even = max(-8, min(7, q_even));
        q_odd  = max(-8, min(7, q_odd));

        uint8_t packed = (uint8_t)((q_odd & 0xF) << 4) | (uint8_t)(q_even & 0xF);
        output[out_offset + i / 2] = packed;
    }
}

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    auto output = torch::empty({M, K / 2}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    int num_groups = K / group_size;
    auto scales = torch::empty({M, num_groups}, torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 block(256);
    dim3 grid((M + 255) / 256, num_groups);

    quantize_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size
    );

    return {output, scales};
}

// =====================================================================
// MMA-based INT4 GEMM Kernel with Tensor Cores
// Double-buffered shared memory + cp.async pipelining
// + XOR-based swizzle to eliminate shared memory bank conflicts
// =====================================================================

// Configuration
static constexpr int BLOCK_M   = 128;
static constexpr int BLOCK_N   = 128;
static constexpr int BLOCK_K   = 64;   // one quantization group per K-step
static constexpr int WARP_SZ   = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M    = BLOCK_M / NUM_WARPS;  // 16
static constexpr int TILES_N   = BLOCK_N / 16;         // 8 (16-col tiles)

// Shared memory stride: 32 bytes data + 16 bytes padding = 48 bytes per row
// But we use a power-of-2 stride (64 bytes) for better swizzle alignment
// Actually, let's use XOR swizzle with the existing 48-byte stride approach
// 
// Swizzle approach: XOR the column (in units of 16 bytes = 128 bits) with
// bits from the row index. This distributes accesses across banks.
//
// Ampere shared memory has 32 banks, 4 bytes each, so 128 bytes per bank cycle.
// For our 32-byte rows (BLOCK_K/2 = 32), we have 2 x 16-byte chunks per row.
// With 16 rows in a warp's fragment, threads accessing same column across
// different rows cause bank conflicts.
//
// Swizzle: when storing row r at 16-byte chunk c, store at chunk (c ^ (r & 1))
// This flips the chunk index based on row parity, distributing across banks.
//
// For a more aggressive swizzle with 48-byte stride, we use padding approach:
// Use 64-byte stride (power of 2) which allows clean XOR swizzle.

// Use 64-byte stride for power-of-2 alignment enabling clean swizzle
static constexpr int SMEM_STRIDE = 64;  // 64 bytes per row (32 data + 32 padding)

// Swizzle function: XOR row bits into the byte offset within a row
// This remaps which bank each row's data lands in
// We XOR bits [4:5] of the offset with bits [0:1] of the row index (shifted to bit positions 4-5)
// Each bank is 4 bytes, so bit 4 selects between bank groups of 16 bytes
__device__ __forceinline__ int swizzle_offset(int row, int col_bytes) {
    // XOR bits from row into the column byte offset
    // row & 0x3 gives 2 bits, shift left by 4 to affect bank selection
    // This ensures consecutive rows map to different banks
    return (col_bytes ^ ((row & 0x3) << 4));
}

// MMA wrapper: m16n8k64 INT4xINT4 -> INT32
__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#else
    asm volatile("{"
        ".reg .b32 t0,t1,t2,t3;\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t0,t1},{%4},{%8},{%0,%1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t2,t3},{%5},{%8},{%2,%3};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1},{%6},{%9},{t0,t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%2,%3},{%7},{%9},{t2,t3};\n"
        "}\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#endif
}


// cp.async: 16-byte async global->shared copy
__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.cg.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit()  { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_wait_one() { asm volatile("cp.async.wait_group 1;\n"); }


// Load MMA A-fragment from shared memory with swizzle
__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride, int base_row) {
    int lane = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col    = (lane % 4) * 4; // byte offset within 16-byte half
    uint4 a;
    // First 16 bytes (col offset 0..15), second 16 bytes (col offset 16..31)
    int swiz_lo_0 = swizzle_offset(base_row + row_lo, col);
    int swiz_hi_0 = swizzle_offset(base_row + row_hi, col);
    int swiz_lo_1 = swizzle_offset(base_row + row_lo, 16 + col);
    int swiz_hi_1 = swizzle_offset(base_row + row_hi, 16 + col);
    a.x = *(const uint32_t*)(base + row_lo * stride + swiz_lo_0);
    a.y = *(const uint32_t*)(base + row_hi * stride + swiz_hi_0);
    a.z = *(const uint32_t*)(base + row_lo * stride + swiz_lo_1);
    a.w = *(const uint32_t*)(base + row_hi * stride + swiz_hi_1);
    return a;
}

// Load MMA B-fragment from shared memory with swizzle
__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride, int base_row) {
    int lane = threadIdx.x % WARP_SZ;
    int row  = lane / 4;
    int col  = (lane % 4) * 4;
    uint2 b;
    int swiz_0 = swizzle_offset(base_row + row, col);
    int swiz_1 = swizzle_offset(base_row + row, 16 + col);
    b.x = *(const uint32_t*)(base + row * stride + swiz_0);
    b.y = *(const uint32_t*)(base + row * stride + swiz_1);
    return b;
}


// Main GEMM kernel - double buffered with swizzled shared memory
__global__ void gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,
    const half    *__restrict__ scales_B,
    half          *__restrict__ C,
    int M, int N, int K, int group_size)
{
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int halfK = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    // Double-buffered shared memory
    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;
    const int tileB = BLOCK_N * SMEM_STRIDE;
    uint8_t *sA0 = smem, *sB0 = smem + tileA;
    uint8_t *sA1 = smem + tileA + tileB, *sB1 = sA1 + tileA;
    uint8_t *sA[2] = {sA0, sA1};
    uint8_t *sB[2] = {sB0, sB1};

    // FP32 accumulators: [n_tile][mma_half=0,1][4 values]
    float acc[TILES_N][2][4];
    #pragma unroll
    for (int j = 0; j < TILES_N; j++)
        for (int h = 0; h < 2; h++)
            acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;

    // Cooperative tile loader with swizzled writes
    // A: 128 rows x 32 bytes = 4096 bytes = 256 x 16B chunks
    // B: 128 rows x 32 bytes = 4096 bytes = 256 x 16B chunks
    // With 256 threads, each thread loads 1 chunk for A and 1 chunk for B
    auto load_tile = [&](int kt, int buf) {
        int kb = kt * (BLOCK_K / 2);  // byte offset in row
        // A: 128 rows x 32 bytes, 256 threads each load 16B
        {
            int row = tid / 2, col16 = tid % 2;
            // Apply swizzle: remap col16 based on row
            int dst_col = swizzle_offset(row, col16 * 16);
            bool p = (bm + row < M) && (kb + col16 * 16 < halfK);
            cp_async_16(sA[buf] + row * SMEM_STRIDE + dst_col,
                        A + (size_t)(bm + row) * halfK + kb + col16 * 16, p);
        }
        // B: 128 rows x 32 bytes
        {
            int row = tid / 2, col16 = tid % 2;
            int dst_col = swizzle_offset(row, col16 * 16);
            bool p = (bn + row < N) && (kb + col16 * 16 < halfK);
            cp_async_16(sB[buf] + row * SMEM_STRIDE + dst_col,
                        B + (size_t)(bn + row) * halfK + kb + col16 * 16, p);
        }
        cp_commit();
    };

    // Prefetch first tile into buffer 0
    if (num_k_tiles > 0) load_tile(0, 0);

    // Main K-loop with double buffering
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur_buf = kt & 1;

        // Start loading next tile into the other buffer
        if (kt + 1 < num_k_tiles) {
            load_tile(kt + 1, (kt + 1) & 1);
        }

        // Wait for current tile to be ready
        if (kt + 1 < num_k_tiles) {
            cp_wait_one();
        } else {
            cp_wait_all();
        }
        __syncthreads();

        // Group index for scales
        int g = (kt * BLOCK_K) / group_size;

        // Activation scales for this warp's rows
        int m_lo = bm + warpId * WARP_M + laneId / 4;
        int m_hi = m_lo + 8;
        float sa_lo = (m_lo < M) ? __half2float(scales_A[m_lo * num_groups + g]) : 0.f;
        float sa_hi = (m_hi < M) ? __half2float(scales_A[m_hi * num_groups + g]) : 0.f;

        // Load A-fragment with swizzled reads
        // base_row for swizzle is the row offset within the tile
        uint4 af = load_a_frag(sA[cur_buf] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE, warpId * WARP_M);

        // Process each 16-column N-tile
        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;

            // Two m16n8k64 MMAs per 16-col tile, with swizzled reads
            uint2 bf0 = load_b_frag(sB[cur_buf] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE, n_off + 0);
            uint2 bf1 = load_b_frag(sB[cur_buf] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE, n_off + 8);

            int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            // Scale results
            int c0 = bn + n_off + (laneId % 4) * 2;
            int c1 = c0 + 1;
            int c2 = c0 + 8;
            int c3 = c2 + 1;
            float sb0 = (c0 < N) ? __half2float(scales_B[c0 * num_groups + g]) : 0.f;
            float sb1 = (c1 < N) ? __half2float(scales_B[c1 * num_groups + g]) : 0.f;
            float sb2 = (c2 < N) ? __half2float(scales_B[c2 * num_groups + g]) : 0.f;
            float sb3 = (c3 < N) ? __half2float(scales_B[c3 * num_groups + g]) : 0.f;

            acc[nt][0][0] += (float)p0[0] * sa_lo * sb0;
            acc[nt][0][1] += (float)p0[1] * sa_lo * sb1;
            acc[nt][0][2] += (float)p0[2] * sa_hi * sb0;
            acc[nt][0][3] += (float)p0[3] * sa_hi * sb1;
            acc[nt][1][0] += (float)p1[0] * sa_lo * sb2;
            acc[nt][1][1] += (float)p1[1] * sa_lo * sb3;
            acc[nt][1][2] += (float)p1[2] * sa_hi * sb2;
            acc[nt][1][3] += (float)p1[3] * sa_hi * sb3;
        }
        __syncthreads();
    }

    // Write results to global memory
    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;
    #pragma unroll
    for (int nt = 0; nt < TILES_N; nt++) {
        int n_off = nt * 16;
        int c0 = bn + n_off + (laneId % 4) * 2;
        int c1 = c0 + 1;
        int c2 = c0 + 8;
        int c3 = c2 + 1;
        if (m_lo < M) {
            if (c0 < N) C[m_lo * N + c0] = __float2half(acc[nt][0][0]);
            if (c1 < N) C[m_lo * N + c1] = __float2half(acc[nt][0][1]);
        }
        if (m_hi < M) {
            if (c0 < N) C[m_hi * N + c0] = __float2half(acc[nt][0][2]);
            if (c1 < N) C[m_hi * N + c1] = __float2half(acc[nt][0][3]);
        }
        if (m_lo < M) {
            if (c2 < N) C[m_lo * N + c2] = __float2half(acc[nt][1][0]);
            if (c3 < N) C[m_lo * N + c3] = __float2half(acc[nt][1][1]);
        }
        if (m_hi < M) {
            if (c2 < N) C[m_hi * N + c2] = __float2half(acc[nt][1][2]);
            if (c3 < N) C[m_hi * N + c3] = __float2half(acc[nt][1][3]);
        }
    }
}

torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed,
    torch::Tensor B_packed,
    torch::Tensor scales_A,
    torch::Tensor scales_B,
    int group_size)
{
    int M = A_packed.size(0);
    int N = B_packed.size(0);
    int K = A_packed.size(1) * 2;

    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    dim3 block(NUM_WARPS * WARP_SZ);  // 256 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    // 2 buffers x (tileA + tileB)
    int smem_bytes = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);

    cudaFuncSetAttribute(gemm_int4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    gemm_int4_kernel<<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);

    return C;
}
