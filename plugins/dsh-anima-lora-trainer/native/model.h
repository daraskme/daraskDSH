#pragma once

#include <ATen/core/Tensor.h>
#include <string>
#include <unordered_map>

namespace anima {

using TensorMap = std::unordered_map<std::string, at::Tensor>;

// Base keys omit the checkpoint prefix. Adapters use <linear>.lora_{A,B}.weight
// and scale 1. Context is the output of the frozen LLM adapter.
at::Tensor predict(
    const TensorMap& weights,
    const TensorMap& adapters,
    const at::Tensor& noisy,
    const at::Tensor& time,
    const at::Tensor& context);

}  // namespace anima
