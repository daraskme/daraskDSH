#include "engine.h"
#include "model.h"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

namespace anima {

void adamw_step(
    at::Tensor& params, const at::Tensor& grads, at::Tensor& first_moment,
    at::Tensor& second_moment, at::Tensor& residual, const at::Tensor& grad_norm,
    double lr, double weight_decay, int64_t step);

ModuleGroup module_group(const std::string& name) {
    if (name.find(".self_attn.") != std::string::npos) return self_attention;
    if (name.find(".cross_attn.") != std::string::npos) return cross_attention;
    if (name.find(".mlp.") != std::string::npos) return mlp;
    if (name.find(".adaln_modulation") != std::string::npos) return adaln_modulation;
    return other_modules;
}

double schedule_factor(const TrainOptions& options, int64_t step, int64_t steps) {
    const auto warmup = static_cast<int64_t>(std::ceil(options.warmup * static_cast<double>(steps)));
    if (step < warmup) {
        return static_cast<double>(step + 1) / static_cast<double>(warmup);
    }
    if (options.lr_cycles <= 0 || steps <= warmup) {
        return 1.0;
    }
    // Hugging Face cosine_with_hard_restarts_schedule_with_warmup.
    const auto progress = static_cast<double>(step - warmup) / static_cast<double>(steps - warmup);
    const auto phase = std::fmod(static_cast<double>(options.lr_cycles) * progress, 1.0);
    return std::max(0.0, 0.5 * (1.0 + std::cos(M_PI * phase)));
}

namespace {

using Dictionary = TensorDictionary;
using TensorList = std::vector<at::Tensor>;

struct TimingEvent {
    cudaEvent_t event{};

    TimingEvent() { C10_CUDA_CHECK(cudaEventCreate(&event)); }
    ~TimingEvent() { cudaEventDestroy(event); }

    void record(cudaStream_t stream) const {
        C10_CUDA_CHECK(cudaEventRecord(event, stream));
    }

    float elapsed_to(const TimingEvent& end) const {
        float elapsed = 0;
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed, event, end.event));
        return elapsed;
    }
};

TensorMap as_map(const Dictionary& tensors) {
    TensorMap result;
    result.reserve(tensors.size());
    for (const auto& item : tensors) {
        result.emplace(item.key(), item.value());
    }
    return result;
}

at::Tensor predict_op(
    const Dictionary& weights, const Dictionary& adapters, const at::Tensor& noisy,
    const at::Tensor& time, const at::Tensor& context) {
    return predict(as_map(weights), as_map(adapters), noisy, time, context);
}

void validate_samples(const TensorList& latents, const TensorList& contexts) {
    TORCH_CHECK(!latents.empty() && latents.size() == contexts.size(),
                "Anima needs one context batch per latent batch and at least one batch");
    for (size_t i = 0; i < latents.size(); ++i) {
        const auto& latent = latents[i];
        const auto& context = contexts[i];
        TORCH_CHECK(latent.dim() == 5 && latent.size(0) > 0 && latent.size(1) == 16 &&
                        latent.size(2) == 1 && latent.size(3) > 0 && latent.size(4) > 0 &&
                        latent.size(3) % 2 == 0 && latent.size(4) % 2 == 0 && latent.is_floating_point(),
                    "Anima latent batch ", i, " must have shape [B,16,1,H,W] with positive even H and W");
        TORCH_CHECK(context.dim() == 3 && context.size(0) == latent.size(0) && context.size(1) > 0 &&
                        context.size(2) == 1024 && context.is_floating_point(),
                    "Anima context batch ", i, " must have shape [B,L,1024] with the same B as its latent batch");
    }
}

void validate_overrides(
    const TensorList& latents, int64_t steps, const std::optional<TensorList>& noises,
    const std::optional<TensorList>& times) {
    TORCH_CHECK(noises.has_value() == times.has_value(), "Supply both Anima noise and time overrides");
    if (!noises) {
        return;
    }
    TORCH_CHECK(noises->size() == static_cast<size_t>(steps) && times->size() == noises->size(),
                "Anima needs one noise and time override per step");
    for (size_t i = 0; i < noises->size(); ++i) {
        const auto& latent = latents[i % latents.size()];
        TORCH_CHECK((*noises)[i].sizes() == latent.sizes() && (*noises)[i].is_floating_point(),
                    "Anima noise override ", i, " must match its latent shape");
        const auto& time = (*times)[i];
        TORCH_CHECK(time.dim() == 2 && time.size(0) == latent.size(0) && time.size(1) == 1 &&
                        time.is_floating_point(),
                    "Anima time override ", i, " must have shape [B,1] with the same B as its latent batch");
    }
}

at::Tensor frozen_to(const at::Tensor& input, const at::TensorOptions& options) {
    // Same-device casts need a copy of inference tensors so autograd can save them for backward.
    return input.detach().to(options, /*non_blocking=*/true, /*copy=*/input.is_inference());
}

struct AdapterParameters {
    TensorMap tensors;
    std::vector<std::string> names;
    TensorList parameters;
    at::Tensor flat;

    AdapterParameters(const Dictionary& initial, const TensorMap& weights, const at::TensorOptions& options) {
        constexpr std::string_view suffix_a = ".lora_A.weight";
        constexpr std::string_view suffix_b = ".lora_B.weight";
        int64_t count = 0;
        names.reserve(initial.size());
        for (const auto& item : initial) {
            const auto& name = item.key();
            const auto& value = item.value();
            TORCH_CHECK(name.size() > suffix_a.size(), "Invalid Anima adapter key: ", name);
            const auto suffix = std::string_view(name).substr(name.size() - suffix_a.size());
            TORCH_CHECK(suffix == suffix_a || suffix == suffix_b, "Invalid Anima adapter key: ", name);
            const auto prefix = name.substr(0, name.size() - suffix.size());
            const auto pair = prefix + std::string(suffix == suffix_a ? suffix_b : suffix_a);
            TORCH_CHECK(initial.contains(pair), "Missing Anima adapter tensor: ", pair);
            const auto weight = weights.find(prefix + ".weight");
            TORCH_CHECK(weight != weights.end() && weight->second.dim() == 2,
                        "Anima adapter has no matching linear weight: ", name);
            TORCH_CHECK(value.dim() == 2 && value.numel() > 0 && value.is_floating_point(),
                        "Anima adapter must be a nonempty floating-point matrix: ", name);
            if (suffix == suffix_a) {
                const auto other = initial.at(pair);
                TORCH_CHECK(other.dim() == 2 && value.size(1) == weight->second.size(1) &&
                                other.size(0) == weight->second.size(0) && other.size(1) == value.size(0),
                            "Anima LoRA A/B shapes do not match the linear weight: ", prefix);
            }
            names.push_back(name);
            count += value.numel();
        }

        // Group-contiguous layout so each module group can take its own learning rate.
        std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
            const auto ga = module_group(a), gb = module_group(b);
            return ga != gb ? ga < gb : a < b;
        });
        parameters.reserve(names.size());
        tensors.reserve(names.size());
        flat = at::empty({count}, options);
        int64_t offset = 0;
        for (const auto& name : names) {
            const auto value = initial.at(name);
            auto parameter = flat.narrow(0, offset, value.numel()).view(value.sizes()).detach();
            parameter.copy_(value.detach(), /*non_blocking=*/true);
            parameter.set_requires_grad(true);
            parameters.push_back(parameter);
            tensors.emplace(name, parameter);
            const auto group = module_group(name);
            if (segments.empty() || segments.back().group != group) {
                segments.push_back({group, offset, 0});
            }
            segments.back().count += value.numel();
            offset += value.numel();
        }
    }

    struct Segment {
        ModuleGroup group;
        int64_t offset;
        int64_t count;
    };
    std::vector<Segment> segments;

    at::Tensor take_gradients() const {
        TensorList gradients;
        gradients.reserve(parameters.size());
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& gradient = parameters[i].mutable_grad();
            TORCH_CHECK(gradient.defined(), "Missing training gradient for Anima adapter: ", names[i]);
            gradients.push_back(gradient.reshape({-1}));
            gradient = at::Tensor();
        }
        return at::cat(gradients);
    }

    Dictionary to_cpu() const {
        const auto buffer = flat.to(at::kCPU);
        Dictionary output;
        int64_t offset = 0;
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto& parameter = parameters[i];
            output.insert(names[i], buffer.narrow(0, offset, parameter.numel()).view(parameter.sizes()));
            offset += parameter.numel();
        }
        return output;
    }
};

}  // namespace

NativeTrainingResult train(
    const Dictionary& initial_weights, const Dictionary& initial_adapters,
    const TensorList& latents, const TensorList& contexts, int64_t steps, double lr, int64_t seed,
    const std::optional<TensorList>& noises, const std::optional<TensorList>& times, int64_t log_every) {
    return train_with_options(initial_weights, initial_adapters, latents, contexts, steps, lr, seed,
                              noises, times, log_every, TrainOptions{});
}

NativeTrainingResult train_with_options(
    const Dictionary& initial_weights, const Dictionary& initial_adapters,
    const TensorList& latents, const TensorList& contexts, int64_t steps, double lr, int64_t seed,
    const std::optional<TensorList>& noises, const std::optional<TensorList>& times, int64_t log_every,
    const TrainOptions& options) {
    TORCH_CHECK(options.weight_decay >= 0 && options.timestep_scale > 0 && options.warmup >= 0 &&
                    options.warmup < 1 && options.lr_cycles >= 0,
                "Invalid Anima schedule, timestep or weight decay option");
    TORCH_CHECK(steps > 0, "Anima training steps must be positive");
    TORCH_CHECK(std::isfinite(lr) && lr > 0 && lr <= std::numeric_limits<float>::max(),
                "Anima learning rate must be finite, positive and representable in float32");
    TORCH_CHECK(log_every >= 0, "Anima log interval must be nonnegative");
    TORCH_CHECK(!initial_weights.empty() && !initial_adapters.empty(), "Anima needs DiT weights and initial LoRA adapters");
    validate_samples(latents, contexts);
    validate_overrides(latents, steps, noises, times);
    TORCH_CHECK(c10::cuda::device_count() > 0, "Native Anima training needs an available CUDA GPU");

    const c10::cuda::CUDAGuard device_guard(c10::cuda::current_device());
    const auto device = device_guard.current_device();
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    const auto gpu_float = at::TensorOptions().device(device).dtype(at::kFloat);
    const auto gpu_bfloat = gpu_float.dtype(at::kBFloat16);
    const auto cpu_float = at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    const auto cpu_long = cpu_float.dtype(at::kLong);
    const c10::InferenceMode inference_guard(false);
    const c10::impl::ExcludeDispatchKeyGuard autocast_guard(c10::DispatchKey::AutocastCUDA);
    const at::AutoGradMode enable_grad(true);

    auto weights = as_map(initial_weights);
    for (auto& [name, value] : weights) {
        value = frozen_to(value, gpu_bfloat);
    }
    AdapterParameters adapters(initial_adapters, weights, gpu_bfloat);
    auto first_moment = at::zeros_like(adapters.flat);
    auto second_moment = at::zeros_like(adapters.flat);
    auto residual = at::zeros_like(adapters.flat);

    auto order_generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));
    auto time_generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));
    auto noise_generator = at::cuda::detail::createCUDAGenerator(device.index());
    noise_generator.set_current_seed(static_cast<uint64_t>(seed));
    at::Tensor order;

    constexpr int64_t metric_chunk = 32;
    auto statistics = at::empty({metric_chunk, 2}, gpu_float);
    auto losses = at::empty({steps}, cpu_float);
    auto elapsed = at::empty({steps}, cpu_float);
    auto* loss_values = losses.data_ptr<float>();
    auto* elapsed_values = elapsed.data_ptr<float>();
    std::array<TimingEvent, metric_chunk> starts, ends;
    int64_t first_metric = 0;
    const int64_t batch_count = latents.size();

    for (int64_t step = 0; step < steps; ++step) {
        const int64_t slot = step - first_metric;
        starts[slot].record(stream);
        auto sample = step % batch_count;
        if (!noises) {
            if (sample == 0) {
                order = at::randperm(batch_count, order_generator, cpu_long);
            }
            sample = order.const_data_ptr<int64_t>()[sample];
        }
        const auto latent = frozen_to(latents[sample], gpu_float);
        const auto context = frozen_to(contexts[sample], gpu_bfloat);
        const auto batch = latent.size(0);
        const auto time = times ? frozen_to((*times)[step], gpu_float)
                                : at::randn({batch, 1}, time_generator, cpu_float)
                                      .mul(options.timestep_scale).sigmoid().to(gpu_float);
        const auto noise = noises ? frozen_to((*noises)[step], gpu_float)
                                  : at::randn(latent.sizes(), noise_generator, gpu_float);
        const auto fraction = time.view({batch, 1, 1, 1, 1});
        const auto noisy = (1 - fraction) * latent + fraction * noise;
        const auto target = noise - latent;
        const auto predicted = predict(weights, adapters.tensors, noisy, time, context);
        const auto loss = at::mse_loss(predicted.to(at::kFloat), target);
        loss.backward();

        {
            const at::NoGradGuard no_grad;
            const auto grads = adapters.take_gradients();
            const auto norm = at::linalg_vector_norm(grads, 2, std::nullopt, false, at::kFloat);
            // The learning rate is per image: a batch of B images takes a B-times step.
            const auto factor = schedule_factor(options, step, steps) * static_cast<double>(batch);
            for (const auto& segment : adapters.segments) {
                const auto group_lr = options.group_lr[segment.group];
                const auto rate = (group_lr < 0 ? lr : group_lr) * factor;
                auto params = adapters.flat.narrow(0, segment.offset, segment.count);
                auto first = first_moment.narrow(0, segment.offset, segment.count);
                auto second = second_moment.narrow(0, segment.offset, segment.count);
                auto rest = residual.narrow(0, segment.offset, segment.count);
                adamw_step(params, grads.narrow(0, segment.offset, segment.count), first, second, rest, norm,
                           rate, options.weight_decay, step + 1);
            }
            statistics[slot][0].copy_(loss.detach());
            statistics[slot][1].copy_(norm);
        }
        ends[slot].record(stream);

        const auto completed = step + 1;
        const bool report = log_every > 0 && (step == 0 || completed % log_every == 0 || completed == steps);
        if (report || completed == steps || slot + 1 == metric_chunk) {
            const auto observed = statistics.narrow(0, 0, slot + 1).to(at::kCPU);
            const auto values = observed.accessor<float, 2>();
            for (int64_t j = 0; j <= slot; ++j) {
                const auto recorded_step = first_metric + j;
                TORCH_CHECK(std::isfinite(values[j][0]) && std::isfinite(values[j][1]),
                            "Non-finite Anima loss or gradient at step ", recorded_step + 1);
                loss_values[recorded_step] = values[j][0];
                elapsed_values[recorded_step] = starts[j].elapsed_to(ends[j]);
            }
            if (report) {
                std::fprintf(stderr, "step %" PRId64 "/%" PRId64 " loss=%.6g elapsed_ms=%.3f\n",
                             completed, steps, loss_values[step], elapsed_values[step]);
                std::fflush(stderr);
            }
            first_metric = completed;
        }
    }

    TORCH_CHECK(at::isfinite(adapters.flat).all().item<bool>(), "Non-finite Anima adapter weights after training");
    return {adapters.to_cpu(), losses, elapsed};
}

}  // namespace anima

TORCH_LIBRARY(anima_native, library) {
    library.def("predict(Dict(str, Tensor) weights, Dict(str, Tensor) adapters, Tensor noisy, Tensor time, Tensor context) -> Tensor");
    library.def("train(Dict(str, Tensor) weights, Dict(str, Tensor) adapters, Tensor[] latents, Tensor[] contexts, "
                "int steps, float lr, int seed, Tensor[]? noises=None, Tensor[]? times=None, int log_every=0) "
                "-> (Dict(str, Tensor), Tensor, Tensor)", TORCH_FN(anima::train));
}

TORCH_LIBRARY_IMPL(anima_native, CompositeImplicitAutograd, library) {
    library.impl("predict", TORCH_FN(anima::predict_op));
}
