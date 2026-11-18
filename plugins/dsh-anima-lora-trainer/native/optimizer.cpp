#include "optimizer_ptx.h"

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>

namespace anima {
namespace {

cudaKernel_t optimizer_kernel() {
    // The CUDA runtime loads this device-only library for all active contexts.
    static const auto kernel = [] {
        cudaLibrary_t library;
        C10_CUDA_CHECK(cudaLibraryLoadData(&library, adamw_ptx, nullptr, nullptr, 0, nullptr, nullptr, 0));
        cudaKernel_t result;
        C10_CUDA_CHECK(cudaLibraryGetKernel(&result, library, "adamw_update"));
        return result;
    }();
    return kernel;
}

}  // namespace

void adamw_step(
    at::Tensor& params, const at::Tensor& grads, at::Tensor& first_moment,
    at::Tensor& second_moment, at::Tensor& residual, const at::Tensor& grad_norm,
    double lr, double weight_decay, int64_t step) {
    const double beta1_power = std::pow(0.9, step);
    const double beta2_power = std::pow(0.99, step);
    const double beta1 = (beta1_power - 0.9) / (beta1_power - 1);
    const double beta2 = (beta2_power - 0.99) / (beta2_power - 1);

    auto* p = params.data_ptr<at::BFloat16>();
    auto* g = grads.const_data_ptr<at::BFloat16>();
    auto* m = first_moment.data_ptr<at::BFloat16>();
    auto* v = second_moment.data_ptr<at::BFloat16>();
    auto* k = residual.data_ptr<at::BFloat16>();
    auto* norm = grad_norm.const_data_ptr<float>();
    auto count = params.numel();
    auto rate = static_cast<float>(lr);
    auto decay = static_cast<float>(1.0 - lr * weight_decay);
    auto b2 = static_cast<float>(beta2);
    auto b1_comp = static_cast<float>(1.0 - beta1);
    auto b2_comp = static_cast<float>(1.0 - beta2);
    void* args[] = {&p, &g, &m, &v, &k, &norm, &count, &rate, &decay, &b1_comp, &b2, &b2_comp};

    constexpr int threads = 256;
    const auto blocks = (count + threads - 1) / threads;
    const auto stream = c10::cuda::getCurrentCUDAStream(params.get_device());
    C10_CUDA_CHECK(cudaLaunchKernel(optimizer_kernel(), blocks, threads, args, 0, stream));
}

}  // namespace anima
