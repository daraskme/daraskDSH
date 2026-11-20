#include "engine.h"
#include "standalone.h"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>

namespace {

using Clock = std::chrono::steady_clock;

at::ScalarType tensor_dtype(uint32_t dtype) {
    switch (dtype) {
        case DSH_ANIMA_BF16: return at::kBFloat16;
        case DSH_ANIMA_F32: return at::kFloat;
        case DSH_ANIMA_F16: return at::kHalf;
    }
    TORCH_CHECK(false, "Unsupported tensor data type in Anima input: ", dtype);
}

at::Tensor tensor_view(const DshAnimaTensor& tensor) {
    TORCH_CHECK(tensor.data && tensor.shape && tensor.dimensions > 0,
                "Missing Anima tensor data or shape");
    const auto dtype = tensor_dtype(tensor.dtype);
    const at::IntArrayRef shape(tensor.shape, tensor.dimensions);
    size_t bytes = c10::elementSize(dtype);
    for (const auto size : shape) {
        TORCH_CHECK(size > 0 && static_cast<uint64_t>(size) <= std::numeric_limits<size_t>::max() / bytes,
                    "Invalid Anima tensor dimension: ", size);
        bytes *= size;
    }
    TORCH_CHECK(bytes == tensor.bytes, "Anima tensor byte size does not match its shape and dtype");
    return at::from_blob(const_cast<uint8_t*>(tensor.data), shape, at::TensorOptions().dtype(dtype));
}

DshAnimaTensor tensor_record(const at::Tensor& tensor, const char* name) {
    TORCH_INTERNAL_ASSERT(tensor.device().is_cpu() && tensor.is_contiguous() && tensor.scalar_type() == at::kBFloat16);
    return {name, static_cast<const uint8_t*>(tensor.const_data_ptr()), tensor.nbytes(),
            tensor.sizes().data(), static_cast<size_t>(tensor.dim()), DSH_ANIMA_BF16};
}

at::Tensor token_array(const int64_t* data, const at::Device& device) {
    TORCH_CHECK(data, "Missing token IDs or attention mask in Anima sample");
    return at::from_blob(const_cast<int64_t*>(data), {1, 512}, at::TensorOptions().dtype(at::kLong)).to(device);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, size_t> prepare_dataset(
    const DshAnimaConfig& config, const at::Device& device, void* user,
    DshAnimaReadSample read_sample, DshAnimaWriteCache write_cache) {
    const at::NoGradGuard no_grad;
    // The profiling executor changes the BF16 outputs of the traced conditioner.
    const torch::jit::GraphOptimizerEnabledGuard unoptimized_frontend(false);
    std::optional<torch::jit::Module> vae, conditioner;
    std::vector<at::Tensor> latents, contexts;
    latents.reserve(config.sample_count);
    contexts.reserve(config.sample_count);
    size_t hits = 0;
    const auto output_options = at::TensorOptions().device(at::kCPU).dtype(at::kBFloat16);

    for (size_t index = 0; index < config.sample_count; ++index) {
        DshAnimaSample sample{};
        TORCH_CHECK(read_sample(user, index, &sample) == 0, "Could not read Anima sample ", index + 1);

        if (!sample.rgb) {
            latents.push_back(tensor_view(sample.latent).clone());
            contexts.push_back(tensor_view(sample.context).clone());
            ++hits;
            continue;
        }

        if (!vae) {
            const std::filesystem::path folder(config.frontend_dir);
            const auto vae_path = folder / "vae.pt";
            const auto conditioner_path = folder / "conditioner.pt";
            TORCH_CHECK(std::filesystem::is_regular_file(vae_path) && std::filesystem::is_regular_file(conditioner_path),
                        "Missing Anima frontend export in ", folder.string(), " (expected vae.pt and conditioner.pt)");
            vae = torch::jit::load(vae_path.string(), device);
            vae->eval();
            conditioner = torch::jit::load(conditioner_path.string(), device);
            conditioner->eval();
        }

        TORCH_CHECK(sample.height > 0 && sample.height % 16 == 0 && sample.width > 0 && sample.width % 16 == 0,
                    "Anima image dimensions must be positive multiples of 16");
        const auto rgb = at::from_blob(const_cast<uint8_t*>(sample.rgb), {sample.height, sample.width, 3}, at::kByte);
        const auto pixels = rgb.to(device, at::kFloat).permute({2, 0, 1}).unsqueeze(0).unsqueeze(2).contiguous().div_(255);
        auto latent = vae->forward({pixels}).toTensor().to(output_options).contiguous();
        auto context = conditioner->forward({
            token_array(sample.qwen_ids, device), token_array(sample.qwen_mask, device),
            token_array(sample.t5_ids, device), token_array(sample.t5_mask, device),
        }).toTensor().to(output_options).contiguous();

        TORCH_CHECK(at::isfinite(latent).all().item<bool>() && at::isfinite(context).all().item<bool>(),
                    "Non-finite output from Anima frontend for sample ", index + 1);
        if (write_cache) {
            const auto latent_record = tensor_record(latent, "latents");
            const auto context_record = tensor_record(context, "context");
            TORCH_CHECK(write_cache(user, index, &latent_record, &context_record) == 0,
                        "Could not write Anima preprocessing cache for sample ", index + 1);
        }
        latents.push_back(std::move(latent));
        contexts.push_back(std::move(context));
    }

    return {std::move(latents), std::move(contexts), hits};
}

anima::TensorDictionary read_weights(const DshAnimaConfig& config) {
    anima::TensorDictionary weights;
    for (size_t index = 0; index < config.weight_count; ++index) {
        const auto& weight = config.weights[index];
        TORCH_CHECK(weight.name && *weight.name, "Unnamed tensor in Anima DiT weights");
        TORCH_CHECK(weights.insert(weight.name, tensor_view(weight)).second, "Duplicate Anima DiT weight: ", weight.name);
    }
    return weights;
}

anima::TensorDictionary initialize_adapters(
    const anima::TensorDictionary& weights, int64_t rank, int64_t seed, const anima::TrainOptions& options) {
    std::vector<std::string> names;
    for (const auto& item : weights) {
        const auto name = item.key();
        if (name.substr(0, 7) == "blocks." && name.size() > 7 && name.substr(name.size() - 7) == ".weight" &&
            item.value().dim() == 2 && options.group_lr[anima::module_group(name)] != 0) {
            names.push_back(item.key());
        }
    }
    TORCH_CHECK(!names.empty(), "No Anima LoRA targets: every module learning rate is 0 or the checkpoint has no blocks");
    std::sort(names.begin(), names.end());

    auto generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));
    anima::TensorDictionary adapters;
    const auto bf16 = at::TensorOptions().dtype(at::kBFloat16);
    for (const auto& name : names) {
        const auto weight = weights.at(name);
        const double bound = 1.0 / std::sqrt(static_cast<double>(weight.size(1)));
        const auto prefix = name.substr(0, name.size() - 7);
        auto first = at::empty({rank, weight.size(1)}, at::kFloat).uniform_(-bound, bound, generator);
        adapters.insert(prefix + ".lora_A.weight", first.to(at::kBFloat16));
        adapters.insert(prefix + ".lora_B.weight", at::zeros({weight.size(0), rank}, bf16));
    }
    return adapters;
}

// Measured on RTX PRO 6000 Blackwell: B=2 cuts ms/image by 27% at 512px
// (1,024 tokens) and gives nothing at 1024px, where a single image already
// saturates the GPU. Pair same-shape samples up to this many patch tokens.
constexpr int64_t max_batched_tokens = 1280;
constexpr int64_t small_batch = 2;

std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>> batch_samples(
    std::vector<at::Tensor> latents, std::vector<at::Tensor> contexts, int64_t seed) {
    std::map<std::vector<int64_t>, std::vector<size_t>> groups;
    auto generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed) ^ 0x5a17u);
    const auto order = at::randperm(static_cast<int64_t>(latents.size()), generator, at::kLong);
    for (int64_t i = 0; i < order.numel(); ++i) {
        const auto index = static_cast<size_t>(order.const_data_ptr<int64_t>()[i]);
        const auto shape = latents[index].sizes();
        groups[{shape.begin(), shape.end()}].push_back(index);
    }
    std::vector<at::Tensor> batched_latents, batched_contexts;
    for (const auto& [shape, members] : groups) {
        const auto tokens = (shape[3] / 2) * (shape[4] / 2);
        const size_t size = tokens <= max_batched_tokens ? small_batch : 1;
        for (size_t start = 0; start < members.size(); start += size) {
            std::vector<at::Tensor> latent_group, context_group;
            for (size_t j = start; j < std::min(start + size, members.size()); ++j) {
                latent_group.push_back(latents[members[j]]);
                context_group.push_back(contexts[members[j]]);
            }
            batched_latents.push_back(at::cat(latent_group));
            batched_contexts.push_back(at::cat(context_group));
        }
    }
    return {std::move(batched_latents), std::move(batched_contexts)};
}

void run(
    const DshAnimaConfig& config, void* user, DshAnimaReadSample read_sample,
    DshAnimaWriteCache write_cache, DshAnimaWriteOutput write_output) {
    TORCH_CHECK(config.sample_count && read_sample && write_output, "Anima needs a dataset and result writer");
    TORCH_CHECK(config.weights && config.weight_count, "Anima needs DiT weights");
    TORCH_CHECK(config.frontend_dir && *config.frontend_dir, "Anima needs a frontend export directory");
    TORCH_CHECK((config.steps > 0) != (config.epochs > 0) && config.steps >= 0 && config.epochs >= 0 &&
                    config.rank > 0 && config.log_every >= 0 &&
                    config.lr > 0 && std::isfinite(config.lr), "Invalid Anima training configuration");
    TORCH_CHECK(c10::cuda::device_count() > 0, "Native Anima training needs an available CUDA GPU");

    at::globalContext().lazyInitDevice(at::kCUDA);
    const c10::cuda::CUDAGuard guard(c10::cuda::current_device());
    const auto device = guard.current_device();
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device.index()));
    TORCH_CHECK(properties.major >= 8, "Native Anima needs a CUDA GPU with native BF16 support");
    constexpr auto aggregate = static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    const auto prepare_start = Clock::now();
    auto [samples, sample_contexts, cache_hits] = prepare_dataset(config, device, user, read_sample, write_cache);
    auto [latents, contexts] = batch_samples(std::move(samples), std::move(sample_contexts), config.seed);
    const int64_t batch_count = static_cast<int64_t>(latents.size());
    TORCH_CHECK(config.steps > 0 || config.epochs <= std::numeric_limits<int64_t>::max() / batch_count,
                "Training step count is too large");
    const int64_t steps = config.steps > 0 ? config.steps : config.epochs * batch_count;
    std::fprintf(stderr, "Training %lld step(s) over %lld batch(es) of up to %lld image(s)\n",
                 static_cast<long long>(steps), static_cast<long long>(batch_count),
                 static_cast<long long>(std::max_element(latents.begin(), latents.end(), [](const auto& a, const auto& b) {
                     return a.size(0) < b.size(0);
                 })->size(0)));
    const auto prepare_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto train_start = Clock::now();
    const auto weights = read_weights(config);
    anima::TrainOptions options;
    options.group_lr = {config.self_attn_lr, config.cross_attn_lr, config.mlp_lr, config.adaln_lr, -1};
    options.weight_decay = config.weight_decay;
    options.timestep_scale = config.timestep_scale;
    options.warmup = config.warmup;
    options.lr_cycles = config.lr_cycles;
    const auto initial = initialize_adapters(weights, config.rank, config.seed, options);
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    auto [trained, losses, elapsed] = anima::train_with_options(
        weights, initial, latents, contexts, steps, config.lr, config.seed, {}, {}, config.log_every, options);
    const auto train_end = Clock::now();
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());

    std::vector<std::string> names;
    std::vector<DshAnimaTensor> tensors;
    names.reserve(trained.size());
    tensors.reserve(trained.size());
    for (const auto& item : trained) {
        names.push_back(item.key());
        tensors.push_back(tensor_record(item.value(), names.back().c_str()));
    }
    const DshAnimaMetrics metrics = {
        losses.const_data_ptr<float>(), elapsed.const_data_ptr<float>(), static_cast<size_t>(steps), cache_hits,
        std::chrono::duration<double>(train_start - prepare_start).count(),
        std::chrono::duration<double>(train_end - train_start).count(),
        static_cast<uint64_t>(stats.allocated_bytes[aggregate].peak),
        static_cast<uint64_t>(stats.reserved_bytes[aggregate].peak),
        static_cast<uint64_t>(prepare_stats.allocated_bytes[aggregate].peak),
        static_cast<uint64_t>(prepare_stats.reserved_bytes[aggregate].peak), properties.name,
    };
    TORCH_CHECK(write_output(user, tensors.data(), tensors.size(), &metrics) == 0, "Could not save the Anima LoRA");
}

}  // namespace

extern "C" uint32_t dsh_anima_abi_version() {
    return 3;
}

extern "C" int dsh_anima_run(
    const DshAnimaConfig* config, void* user, DshAnimaReadSample read_sample,
    DshAnimaWriteCache write_cache, DshAnimaWriteOutput write_output, DshAnimaError on_error) {
    try {
        TORCH_CHECK(config, "Missing Anima training configuration");
        run(*config, user, read_sample, write_cache, write_output);
        return 0;
    } catch (const c10::Error& error) {
        if (on_error) {
            on_error(user, error.what_without_backtrace());
        }
    } catch (const std::exception& error) {
        if (on_error) {
            on_error(user, error.what());
        }
    } catch (...) {
        if (on_error) {
            on_error(user, "Unknown native Anima training failure");
        }
    }
    return 1;
}
