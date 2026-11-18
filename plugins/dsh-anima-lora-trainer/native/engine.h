#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace anima {

using TensorDictionary = c10::Dict<std::string, at::Tensor>;
using NativeTrainingResult = std::tuple<TensorDictionary, at::Tensor, at::Tensor>;

NativeTrainingResult train(
    const TensorDictionary& weights, const TensorDictionary& adapters,
    const std::vector<at::Tensor>& latents, const std::vector<at::Tensor>& contexts,
    int64_t steps, double lr, int64_t seed, const std::optional<std::vector<at::Tensor>>& noises,
    const std::optional<std::vector<at::Tensor>>& times, int64_t log_every);

}  // namespace anima
