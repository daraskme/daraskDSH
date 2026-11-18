#include "rope_ptx.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace anima {
namespace {

cudaKernel_t rope_kernel() {
    static const auto kernel = [] {
        cudaLibrary_t library;
        C10_CUDA_CHECK(cudaLibraryLoadData(&library, rope_ptx, nullptr, nullptr, 0, nullptr, nullptr, 0));
        cudaKernel_t result;
        C10_CUDA_CHECK(cudaLibraryGetKernel(&result, library, "rope_bf16"));
        return result;
    }();
    return kernel;
}

}  // namespace

// Rotary embedding of a BF16 CUDA tensor [B, S, H, D] with [.., S, .., D] BF16
// cos/sin tables (see rope.cu). `backward` applies the transposed rotation.
at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin, bool backward) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 4 && x.size(3) % 2 == 0,
                "Fused RoPE needs a BF16 CUDA tensor [B,S,H,D] with even D");
    const auto input = x.contiguous();
    const auto cosine = cos.to(at::kBFloat16).reshape({x.size(1), x.size(3)}).contiguous();
    const auto sine = sin.to(at::kBFloat16).reshape({x.size(1), x.size(3)}).contiguous();
    auto output = at::empty_like(input);
    const c10::cuda::CUDAGuard guard(input.device());
    const auto* in = input.const_data_ptr<at::BFloat16>();
    const auto* c = cosine.const_data_ptr<at::BFloat16>();
    const auto* s = sine.const_data_ptr<at::BFloat16>();
    auto* out = output.data_ptr<at::BFloat16>();
    long long pairs = input.numel() / 2;
    int sequence = static_cast<int>(x.size(1)), heads = static_cast<int>(x.size(2));
    int half = static_cast<int>(x.size(3) / 2), direction = backward ? 1 : 0;
    void* args[] = {&in, &c, &s, &out, &pairs, &sequence, &heads, &half, &direction};
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((pairs + threads - 1) / threads);
    C10_CUDA_CHECK(cudaLaunchKernel(rope_kernel(), blocks, threads, args, 0,
                                    c10::cuda::getCurrentCUDAStream(input.get_device())));
    return output;
}

}  // namespace anima
