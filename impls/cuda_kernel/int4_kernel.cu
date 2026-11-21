// INT4 x FP16 matrix multiply in CUDA, version 1 (impls/cuda_kernel).
//
//   out[m, n] = sum_k x[m, k] * w[k, n],     w[k, n] = (q[k, n] - 8) * s[k / G, n]
//
// x: [M, K] FP16 activations, q: 4-bit weights, s: one FP16 scale per group of
// G consecutive k values in a column. FP32 accumulation, FP16 output.
//
// ---------------------------------------------------------------- the idea
// During decode M is small (1..64 tokens), so the work is dominated by READING
// the weights from HBM. The kernel is designed around reading them as fast as
// possible:
//
//   * ONE WARP (32 threads) computes ONE output column n, for M_TILE rows.
//     Each of the 32 lanes takes a different part of K, then the 32 partial
//     sums are added with warp shuffles (registers only, no shared memory).
//   * Weight layout (built in prepare()): column-major, K contiguous, 8 values
//     per 32-bit word. One column = K/2 bytes in a row in memory.
//   * Each lane loads 16 bytes (= 32 weights) at a time, and lanes 0..31 load
//     32 NEIGHBOURING 16-byte pieces: one fully coalesced 512-byte read per warp.
//   * The 32 weights of one piece all lie in ONE quantization group (G is a
//     multiple of 32), so they share one scale: we sum x * (q - 8) first and
//     multiply by the scale once per piece, not once per weight.
//   * 8 warps per block = 8 columns per block. N = 4096 gives 512 blocks, enough
//     to fill all SMs without split-K (K is already split across 32 lanes).
//
// This uses ordinary CUDA cores (FP32 FMAs), not tensor cores. That is fine
// for small M (memory-bound), but at large M the math becomes the bottleneck
// and tensor-core kernels (cuBLAS, Marlin) pull ahead. That trade-off is one of
// the things the study measures.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;    // = output columns per block
constexpr int CHUNK = 32;             // weights per 16-byte load (32 x 4 bits = 128 bits)

// M_TILE (rows of x handled by one warp) is a template parameter, so the
// compiler knows it and keeps acc[M_TILE] in registers with fully unrolled loops.
template <int M_TILE>
__global__ void int4_matmul_kernel(
    const half* __restrict__ x,        // [M, K]
    const uint4* __restrict__ wq,      // [N, K/32] 16-byte pieces (= [N, K/8] int32)
    const half* __restrict__ scales,   // [N, K/G]  (transposed in prepare())
    half* __restrict__ out,            // [M, N]
    int M, int N, int K, int group_size)
{
    // Which column does this warp own, and which rows of x?
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int m0 = blockIdx.y * M_TILE;
    if (n >= N) return;      // the whole warp leaves together, so the shuffles below stay safe

    const int chunks = K / CHUNK;                        // 16-byte pieces per column
    const uint4* wcol = wq + (size_t)n * chunks;         // start of column n's weights
    const half* scol = scales + (size_t)n * (K / group_size);

    float acc[M_TILE];
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) acc[m] = 0.f;

    // Lane l handles pieces l, l + 32, l + 64, ...: in every iteration the 32
    // lanes read 32 neighbouring pieces, i.e. one contiguous 512-byte block.
    for (int c = lane; c < chunks; c += WARP_SIZE) {
        const uint4 w = wcol[c];                         // 32 weights, k = 32c .. 32c + 31
        const float s = __half2float(scol[(c * CHUNK) / group_size]);

        // Unpack once, reuse for every row of x. In each 32-bit word, nibble j
        // (bits 4j .. 4j+3) is the weight for k = (first k of the word) + j.
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
        float q[CHUNK];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                q[i * 8 + j] = (float)((int)((words[i] >> (4 * j)) & 0xF) - 8);   // -8 .. 7

#pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int row = m0 + m;
            if (row < M) {
                // The matching 32 activations: 64 bytes = four 16-byte loads.
                const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * K + c * CHUNK);
                float part = 0.f;                        // sum of x * (q - 8) over this piece
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 xv = xr[i];              // 8 FP16 values
                    const half2* h = reinterpret_cast<const half2*>(&xv);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(h[j]);   // exact FP16 -> FP32
                        part += f.x * q[i * 8 + 2 * j] + f.y * q[i * 8 + 2 * j + 1];
                    }
                }
                acc[m] += s * part;                      // one scale multiply per piece
            }
        }
    }

    // Add up the 32 lanes' partial sums. __shfl_down_sync reads a register of
    // the lane `offset` places higher; after 5 halving steps lane 0 holds the
    // total. 0xffffffff = all 32 lanes take part.
#pragma unroll
    for (int m = 0; m < M_TILE; ++m)
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);

    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M_TILE; ++m)
            if (m0 + m < M) out[(size_t)(m0 + m) * N + n] = __float2half(acc[m]);
    }
}

template <int M_TILE>
static void launch(const torch::Tensor& x, const torch::Tensor& wq, const torch::Tensor& scales,
                   torch::Tensor& out, int group_size) {
    const int M = x.size(0), K = x.size(1), N = wq.size(0);
    const dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    const dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, (M + M_TILE - 1) / M_TILE);
    // Launch on PyTorch's current stream, so CUDA-event timing sees this kernel.
    int4_matmul_kernel<M_TILE><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const uint4*>(wq.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Python entry point: int4_matmul(x [M, K] fp16, wq [N, K/8] int32, scales [N, K/G] fp16, G)
torch::Tensor int4_matmul(torch::Tensor x, torch::Tensor wq, torch::Tensor scales, int64_t group_size) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.is_contiguous(),
                "x must be a contiguous 2-D FP16 CUDA tensor");
    TORCH_CHECK(wq.is_cuda() && wq.scalar_type() == at::kInt && wq.is_contiguous(),
                "wq must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kHalf && scales.is_contiguous(),
                "scales must be a contiguous FP16 CUDA tensor");
    const int64_t M = x.size(0), K = x.size(1), N = wq.size(0);
    TORCH_CHECK(K % CHUNK == 0 && group_size % CHUNK == 0 && K % group_size == 0,
                "K and the group size must be multiples of 32, and G must divide K");
    TORCH_CHECK(wq.size(1) == K / 8, "wq must be [N, K/8]");
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == K / group_size, "scales must be [N, K/G]");
    // The kernel reads x and wq with 16-byte loads, which need 16-byte-aligned addresses.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(wq.data_ptr()) % 16 == 0,
                "x and wq must be 16-byte aligned");

    auto out = torch::empty({M, N}, x.options());
    // Pick the row tile: small M uses exactly what it needs; larger M is split
    // into tiles of 8 rows (more rows per warp would need more registers).
    if (M <= 1)      launch<1>(x, wq, scales, out, (int)group_size);
    else if (M <= 2) launch<2>(x, wq, scales, out, (int)group_size);
    else if (M <= 4) launch<4>(x, wq, scales, out, (int)group_size);
    else             launch<8>(x, wq, scales, out, (int)group_size);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_matmul", &int4_matmul, "INT4 x FP16 matmul (symmetric, FP32 accumulation)");
}
