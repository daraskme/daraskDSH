#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace anima {

// LoRA module groups, in the order adapters are laid out for the optimizer.
enum ModuleGroup : int { self_attention, cross_attention, mlp, adaln_modulation, other_modules, module_group_count };
ModuleGroup module_group(const std::string& adapter_name);

struct TrainOptions {
    // Per-group learning rate; negative uses the base rate. Each is per image.
    std::array<double, module_group_count> group_lr{-1, -1, -1, -1, -1};
    double weight_decay = 0.01;
    double timestep_scale = 1.0;  // t = sigmoid(scale * N(0, 1))
    double warmup = 0.0;          // fraction of steps with a linear warmup
    int64_t lr_cycles = 0;        // 0: constant; n: cosine with n hard restarts
};

using TensorDictionary = c10::Dict<std::string, at::Tensor>;
using NativeTrainingResult = std::tuple<TensorDictionary, at::Tensor, at::Tensor>;

NativeTrainingResult train(
    const TensorDictionary& weights, const TensorDictionary& adapters,
    const std::vector<at::Tensor>& latents, const std::vector<at::Tensor>& contexts,
    int64_t steps, double lr, int64_t seed, const std::optional<std::vector<at::Tensor>>& noises,
    const std::optional<std::vector<at::Tensor>>& times, int64_t log_every);

NativeTrainingResult train_with_options(
    const TensorDictionary& weights, const TensorDictionary& adapters,
    const std::vector<at::Tensor>& latents, const std::vector<at::Tensor>& contexts,
    int64_t steps, double lr, int64_t seed, const std::optional<std::vector<at::Tensor>>& noises,
    const std::optional<std::vector<at::Tensor>>& times, int64_t log_every, const TrainOptions& options);

}  // namespace anima
