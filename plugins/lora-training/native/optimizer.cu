namespace {

__device__ float from_bfloat16(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ unsigned short to_bfloat16(float value) {
    unsigned short rounded;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(rounded) : "f"(value));
    return rounded;
}

__device__ float round_bfloat16(float value) {
    return from_bfloat16(to_bfloat16(value));
}

}  // namespace

// Fuse Optimi's CUDA AdamW sequence while preserving each BF16 tensor write.
extern "C" __global__ void adamw_update(
    unsigned short* params, const unsigned short* grads, unsigned short* first_moment,
    unsigned short* second_moment, unsigned short* residual, const float* grad_norm,
    long long count, float lr, float decay, float beta1_complement,
    float beta2, float beta2_complement) {
    const auto index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const float clip = fminf(1.0f, 1.0f / (*grad_norm + 1e-6f));
    const float gradient = round_bfloat16(from_bfloat16(grads[index]) * clip);
    const float m_previous = from_bfloat16(first_moment[index]);
    const float m = round_bfloat16(beta1_complement < 0.5f
        ? fmaf(beta1_complement, gradient - m_previous, m_previous)
        : fmaf(-(1.0f - beta1_complement), gradient - m_previous, gradient));
    const float v_previous = round_bfloat16(from_bfloat16(second_moment[index]) * beta2);
    const float v = round_bfloat16(fmaf(beta2_complement * gradient, gradient, v_previous));
    const float denominator = round_bfloat16(round_bfloat16(sqrtf(v)) + 1e-8f);
    const float previous = round_bfloat16(from_bfloat16(params[index]) * decay);
    const float update = round_bfloat16(from_bfloat16(residual[index]) - lr * m / denominator);
    const auto next = to_bfloat16(previous + update);

    first_moment[index] = to_bfloat16(m);
    second_moment[index] = to_bfloat16(v);
    residual[index] = to_bfloat16(update + round_bfloat16(previous - from_bfloat16(next)));
    params[index] = next;
}
