// Fused FP8 (e4m3) activation quantizers for the fast Anima DiT path: each kernel reads a BF16 row once and writes
// the FP8 row plus its scale (amax / 448), so the projections can run as row-wise scaled FP8 GEMMs (_scaled_mm).
//   modulate: LayerNorm (no affine) * (1 + scale) + shift, in FP32   (inputs of q/k/v, cross q, mlp.layer1)
//   gelu:     exact (erf) GELU                                       (input of mlp.layer2)
//   quant:    plain row quantization                                 (attention outputs -> output_proj)
// One 256-thread block per row; a thread keeps PER contiguous elements in registers.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr float kFp8Max = 448.0f;

template <bool kMax>
__device__ float block_reduce(float v, float* shared) {
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, o);
        v = kMax ? fmaxf(v, other) : v + other;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = lane < kThreads / 32 ? shared[lane] : 0.0f;  // identity for both (amax >= 0)
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, o);
        v = kMax ? fmaxf(v, other) : v + other;
    }
    __syncthreads();
    return v;
}

template <int PER>
__device__ void load_row(const __nv_bfloat16* row, float (&v)[PER]) {
    const uint4* src = reinterpret_cast<const uint4*>(row + threadIdx.x * PER);
#pragma unroll
    for (int i = 0; i < PER / 8; ++i) {
        const uint4 raw = src[i];
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[i * 8 + j] = __bfloat162float(h[j]);
    }
}

// amax over the row, then FP8 values and the row scale
template <int PER>
__device__ void store_fp8(float (&v)[PER], float* shared, __nv_fp8_e4m3* out_row, float* row_scale) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = block_reduce<true>(amax, shared);
    const float scale = fmaxf(amax / kFp8Max, 1e-12f);
    const float inv = 1.0f / scale;
    __align__(8) __nv_fp8_e4m3 q[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) q[i] = __nv_fp8_e4m3(v[i] * inv);
    uint2* dst = reinterpret_cast<uint2*>(out_row + threadIdx.x * PER);
#pragma unroll
    for (int i = 0; i < PER / 8; ++i) dst[i] = reinterpret_cast<const uint2*>(q)[i];
    if (threadIdx.x == 0) *row_scale = scale;
}

template <int PER>
__global__ void __launch_bounds__(kThreads) modulate_kernel(const __nv_bfloat16* x, const __nv_bfloat16* shift,
                                                            const __nv_bfloat16* scale, int64_t rows_per_batch,
                                                            float eps, __nv_fp8_e4m3* out, float* row_scale) {
    __shared__ float shared[kThreads / 32];
    constexpr int dim = kThreads * PER;
    const int64_t row = blockIdx.x;
    const int64_t batch = row / rows_per_batch;
    float v[PER];
    load_row<PER>(x + row * dim, v);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i) sum += v[i];
    const float mean = block_reduce<false>(sum, shared) / dim;
    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i) { v[i] -= mean; sq += v[i] * v[i]; }
    const float rstd = rsqrtf(block_reduce<false>(sq, shared) / dim + eps);
    float sc[PER], sh[PER];
    load_row<PER>(scale + batch * dim, sc);
    load_row<PER>(shift + batch * dim, sh);
#pragma unroll
    for (int i = 0; i < PER; ++i) v[i] = v[i] * rstd * (1.0f + sc[i]) + sh[i];
    store_fp8<PER>(v, shared, out + row * dim, row_scale + row);
}

template <int PER, bool kGelu>
__global__ void __launch_bounds__(kThreads) quant_kernel(const __nv_bfloat16* x, __nv_fp8_e4m3* out, float* row_scale) {
    __shared__ float shared[kThreads / 32];
    constexpr int dim = kThreads * PER;
    const int64_t row = blockIdx.x;
    float v[PER];
    load_row<PER>(x + row * dim, v);
    if (kGelu) {
#pragma unroll
        for (int i = 0; i < PER; ++i) v[i] = 0.5f * v[i] * (1.0f + erff(v[i] * 0.70710678118654752f));
    }
    store_fp8<PER>(v, shared, out + row * dim, row_scale + row);
}

}  // namespace

// dim must be 2048 (modulate, quant) or 8192 (gelu); x rows contiguous; shift/scale [batch, dim] contiguous.
extern "C" int dsh_anima_modulate_fp8(const void* x, const void* shift, const void* scale, int64_t rows,
                                      int64_t rows_per_batch, int dim, float eps, void* out, float* row_scale,
                                      cudaStream_t stream) {
    if (dim != kThreads * 8) return 1;
    modulate_kernel<8><<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(shift),
        static_cast<const __nv_bfloat16*>(scale), rows_per_batch, eps, static_cast<__nv_fp8_e4m3*>(out), row_scale);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int dsh_anima_quant_fp8(const void* x, int64_t rows, int dim, int gelu, void* out, float* row_scale,
                                   cudaStream_t stream) {
    const auto* in = static_cast<const __nv_bfloat16*>(x);
    auto* q = static_cast<__nv_fp8_e4m3*>(out);
    if (dim == kThreads * 8 && !gelu) quant_kernel<8, false><<<rows, kThreads, 0, stream>>>(in, q, row_scale);
    else if (dim == kThreads * 32 && gelu) quant_kernel<32, true><<<rows, kThreads, 0, stream>>>(in, q, row_scale);
    else return 1;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
