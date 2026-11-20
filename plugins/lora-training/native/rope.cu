// Fused rotary embedding for BF16 [B, S, H, D] tensors with [S, D] BF16 cos/sin.
// One thread owns a rotation pair (d, d + D/2): it reads both inputs once,
// computes in FP32 and rounds each output to BF16 once.
//   forward:  y_lo = x_lo*c_lo - x_hi*s_lo      y_hi = x_hi*c_hi + x_lo*s_hi
//   backward: g_lo = dy_lo*c_lo + dy_hi*s_hi    g_hi = dy_hi*c_hi - dy_lo*s_lo

namespace {

__device__ float from_bfloat16(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ unsigned short to_bfloat16(float value) {
    unsigned short rounded;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(rounded) : "f"(value));
    return rounded;
}

}  // namespace

extern "C" __global__ void rope_bf16(
    const unsigned short* __restrict__ input, const unsigned short* __restrict__ cosine,
    const unsigned short* __restrict__ sine, unsigned short* __restrict__ output,
    long long pairs, int sequence, int heads, int half, int backward) {
    const long long pair = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    const int d = static_cast<int>(pair % half);
    const long long row = pair / half;                   // index over (b, s, h)
    const int s = static_cast<int>((row / heads) % sequence);
    const long long lo = row * 2 * half + d;
    const long long hi = lo + half;
    const int table = s * 2 * half + d;
    const float c_lo = from_bfloat16(cosine[table]), c_hi = from_bfloat16(cosine[table + half]);
    const float s_lo = from_bfloat16(sine[table]), s_hi = from_bfloat16(sine[table + half]);
    const float a = from_bfloat16(input[lo]), b = from_bfloat16(input[hi]);
    if (backward) {
        output[lo] = to_bfloat16(a * c_lo + b * s_hi);
        output[hi] = to_bfloat16(b * c_hi - a * s_lo);
    } else {
        output[lo] = to_bfloat16(a * c_lo - b * s_lo);
        output[hi] = to_bfloat16(b * c_hi + a * s_hi);
    }
}
