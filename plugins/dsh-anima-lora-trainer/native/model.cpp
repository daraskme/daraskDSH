// Architecture and operation order follow the Cosmos Predict2 DiT from
// diffusion-pipe (NVIDIA CORPORATION & AFFILIATES, Apache-2.0).

#include "model.h"

#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <cmath>
#include <string_view>
#include <vector>

namespace anima {

at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin, bool backward);

namespace {

using at::Tensor;

constexpr int64_t latent_channels = 16;
constexpr int64_t context_channels = 1024;
constexpr int64_t patch_size = 2;
constexpr int64_t max_patch_extent = 512;
constexpr double norm_epsilon = 1e-6;

// y = x W^T + (x A^T) B^T with the LoRA term passed to cuBLAS as the GEMM's C
// input, in both directions: forward adds B(Ax) and backward adds dX_A inside the
// frozen GEMM instead of separate elementwise adds. W is frozen (no gradient).
struct LoraLinear : torch::autograd::Function<LoraLinear> {
    static Tensor forward(
        torch::autograd::AutogradContext* ctx, const Tensor& x, const Tensor& weight,
        const Tensor& a, const Tensor& b) {
        const auto rows = x.reshape({-1, x.size(-1)});
        const auto low_rank = rows.mm(a.t());
        auto output = at::addmm(low_rank.mm(b.t()), rows, weight.t());
        ctx->save_for_backward({rows, low_rank, weight, a, b});
        auto shape = x.sizes().vec();
        shape.back() = weight.size(0);
        return output.view(shape);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grads) {
        const auto saved = ctx->get_saved_variables();
        const auto& rows = saved[0];
        const auto& low_rank = saved[1];
        const auto& weight = saved[2];
        const auto& a = saved[3];
        const auto& b = saved[4];
        const auto grad = grads[0].reshape({-1, weight.size(0)});
        const auto grad_low_rank = grad.mm(b);
        Tensor grad_x;
        if (ctx->needs_input_grad(0)) {
            auto shape = grads[0].sizes().vec();
            shape.back() = weight.size(1);
            grad_x = at::addmm(grad_low_rank.mm(a), grad, weight).view(shape);
        }
        return {grad_x, Tensor(), grad_low_rank.t().mm(rows), grad.t().mm(low_rank)};
    }
};

struct Projections {
    const TensorMap& weights;
    const TensorMap& adapters;

    Tensor frozen(const std::string& name) const {
        const auto it = weights.find(name);
        TORCH_CHECK(it != weights.end(), "Missing Anima weight ", name);
        return it->second.requires_grad() ? it->second.detach() : it->second;
    }

    Tensor linear(const Tensor& x, const std::string& name) const {
        const auto weight = frozen(name + ".weight");
        const auto dtype = weight.scalar_type();
        // Cast once and share it between the frozen and LoRA branches.
        const auto input = x.scalar_type() == dtype ? x : x.to(dtype);
        const auto a = adapters.find(name + ".lora_A.weight");
        const auto b = adapters.find(name + ".lora_B.weight");
        TORCH_CHECK((a == adapters.end()) == (b == adapters.end()),
                    "Incomplete Anima LoRA pair for ", name);
        if (a == adapters.end()) {
            return at::linear(input, weight);
        }
        // Reference CUDA autocast performs the two LoRA linears at the
        // model's compute dtype even when the adapter leaves are FP32.
        return LoraLinear::apply(input, weight, a->second.to(dtype), b->second.to(dtype));
    }
};

Tensor rms_norm(const Tensor& x, const Tensor& weight) {
    auto fp32 = x.to(at::kFloat);
    auto normalized = fp32 * at::rsqrt(at::pow(fp32, 2).mean(-1, true) + norm_epsilon);
    return normalized.to(x.scalar_type()) * weight;
}

// Fused BF16 CUDA rotary embedding; the backward is the transposed rotation.
struct FusedRope : torch::autograd::Function<FusedRope> {
    static Tensor forward(torch::autograd::AutogradContext* ctx, const Tensor& x, const Tensor& cos,
                          const Tensor& sin) {
        ctx->save_for_backward({cos, sin});
        return rope_cuda(x, cos, sin, false);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grads) {
        const auto saved = ctx->get_saved_variables();
        return {rope_cuda(grads[0], saved[0], saved[1], true), Tensor(), Tensor()};
    }
};

struct RotaryEmbedding {
    Tensor cos;
    Tensor sin;

    Tensor apply(const Tensor& x) const {
        if (x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 4) {
            return FusedRope::apply(x, cos, sin);
        }
        auto halves = x.chunk(2, -1);
        auto rotated = at::cat({-halves[1], halves[0]}, -1);
        return x * cos + rotated * sin;
    }
};

RotaryEmbedding image_rope(
    int64_t height,
    int64_t width,
    int64_t head_dim,
    at::TensorOptions options) {
    // The reference partitions each head into time, height and width bands.
    // Images always use time index zero and a spatial extrapolation ratio of 4.
    const auto spatial_dim = head_dim / 6 * 2;
    const auto temporal_dim = head_dim - 2 * spatial_dim;
    const auto theta = 10000.0 * std::pow(4.0, double(spatial_dim) / (spatial_dim - 2));
    const auto float_options = options.dtype(at::kFloat);
    const auto exponent = at::arange(0, spatial_dim, 2, float_options) / spatial_dim;
    const auto frequencies = at::reciprocal(at::pow(theta, exponent));
    const auto height_angles = at::outer(at::arange(height, float_options), frequencies);
    const auto width_angles = at::outer(at::arange(width, float_options), frequencies);
    const auto time_angles = at::zeros({height, width, temporal_dim / 2}, float_options);
    auto half = at::cat({
        time_angles,
        height_angles.unsqueeze(1).expand({height, width, spatial_dim / 2}),
        width_angles.unsqueeze(0).expand({height, width, spatial_dim / 2}),
    }, -1);
    auto angles = at::cat({half, half}, -1).reshape({1, height * width, 1, head_dim});
    return {at::cos(angles).to(options), at::sin(angles).to(options)};
}

Tensor attention(
    const Projections& projections,
    const Tensor& query,
    const Tensor& context,
    const std::string& prefix,
    int64_t heads,
    int64_t head_dim,
    const RotaryEmbedding* rope = nullptr) {
    const auto project = [&](const Tensor& x, const char* key) {
        return projections.linear(x, prefix + key).reshape({x.size(0), x.size(1), heads, head_dim});
    };
    // Self-attention projects one FP32 input three times; cast it once for Q/K/V.
    const auto dtype = projections.frozen(prefix + ".q_proj.weight").scalar_type();
    const auto query_input = query.to(dtype);
    const auto context_input = context.is_same(query) ? query_input : context.to(dtype);
    // ATen's fused RMSNorm (one kernel each way) applies the weight before the
    // single BF16 rounding; the reference rounds before the weight multiply.
    const auto head_norm = [&](const Tensor& x, const char* key) {
        return at::rms_norm(x, {head_dim}, projections.frozen(prefix + key), norm_epsilon);
    };
    auto q = head_norm(project(query_input, ".q_proj"), ".q_norm.weight");
    auto k = head_norm(project(context_input, ".k_proj"), ".k_norm.weight");
    auto v = project(context_input, ".v_proj");
    if (rope != nullptr) {
        q = rope->apply(q);
        k = rope->apply(k);
    }
    auto attended = at::scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2));
    auto combined = attended.transpose(1, 2).reshape({query.size(0), query.size(1), heads * head_dim});
    return projections.linear(combined, prefix + ".output_proj");
}

Tensor modulation(const Projections& projections, const Tensor& emb, const std::string& prefix) {
    return projections.linear(projections.linear(emb, prefix + ".1"), prefix + ".2");
}

Tensor modulated_norm(const Tensor& x, const Tensor& shift, const Tensor& scale) {
    // CUDA autocast promotes this affine-free LayerNorm to FP32. The following
    // scale-and-shift stays FP32 until the next autocast linear.
    // (1 + scale) and shift are per sample, so each sample's modulation is
    // LayerNorm's own affine weight and bias: one kernel each way instead of
    // separate multiply/add kernels and their gradient reductions.
    const auto channels = x.size(-1);
    const auto input = x.to(at::kFloat);
    const auto weight = (1 + scale).to(at::kFloat);
    const auto bias = shift.to(at::kFloat);
    if (x.size(0) == 1) {
        return at::layer_norm(input, {channels}, weight.reshape({channels}), bias.reshape({channels}), norm_epsilon);
    }
    std::vector<Tensor> samples;
    for (int64_t i = 0; i < x.size(0); ++i) {
        samples.push_back(at::layer_norm(input[i], {channels}, weight[i].reshape({channels}),
                                         bias[i].reshape({channels}), norm_epsilon));
    }
    return at::stack(samples);
}

Tensor transformer_block(
    const Projections& projections,
    Tensor x,
    const Tensor& context,
    const Tensor& activated_timestep,
    const Tensor& timestep_adaln,
    const std::string& prefix,
    int64_t heads,
    int64_t head_dim,
    const RotaryEmbedding& rope) {
    const auto modulate = [&](const char* key) {
        return (modulation(projections, activated_timestep, prefix + key) + timestep_adaln).chunk(3, -1);
    };
    const auto self = modulate(".adaln_modulation_self_attn");
    auto normalized = modulated_norm(x, self[0], self[1]);
    auto attended = attention(projections, normalized, normalized, prefix + ".self_attn", heads, head_dim, &rope);
    x = x + self[2] * attended;

    const auto cross = modulate(".adaln_modulation_cross_attn");
    normalized = modulated_norm(x, cross[0], cross[1]);
    attended = attention(projections, normalized, context, prefix + ".cross_attn", heads, head_dim);
    x = x + cross[2] * attended;

    const auto mlp = modulate(".adaln_modulation_mlp");
    normalized = modulated_norm(x, mlp[0], mlp[1]);
    auto hidden = projections.linear(normalized, prefix + ".mlp.layer1");
    auto result = projections.linear(at::gelu(hidden), prefix + ".mlp.layer2");
    return x + mlp[2] * result;
}

}  // namespace

Tensor predict(
    const TensorMap& weights,
    const TensorMap& adapters,
    const Tensor& noisy,
    const Tensor& time,
    const Tensor& context) {
    TORCH_CHECK(noisy.dim() == 5 && noisy.size(1) == latent_channels && noisy.size(2) == 1,
                "Anima noisy latents must have shape [B, 16, 1, H, W]");
    const auto batch = noisy.size(0);
    const auto height = noisy.size(3) / patch_size;
    const auto width = noisy.size(4) / patch_size;
    TORCH_CHECK(batch > 0 && height > 0 && width > 0 && noisy.size(3) % patch_size == 0 &&
                    noisy.size(4) % patch_size == 0 && height <= max_patch_extent && width <= max_patch_extent,
                "Anima latents require positive even spatial dimensions up to ", max_patch_extent * patch_size);
    TORCH_CHECK(time.dim() == 2 && time.size(0) == batch && time.size(1) == 1,
                "Anima timesteps must have shape [B, 1]");
    TORCH_CHECK(context.dim() == 3 && context.size(0) == batch && context.size(1) > 0 &&
                    context.size(2) == context_channels,
                "Anima context must have shape [B, L, ", context_channels, "]");

    const Projections projections{weights, adapters};
    const auto patch_weight = projections.frozen("x_embedder.proj.1.weight");
    const auto hidden = patch_weight.size(0);
    const auto head_dim = projections.frozen("blocks.0.self_attn.q_norm.weight").size(0);
    TORCH_CHECK(head_dim >= 12 && head_dim % 2 == 0 && hidden % head_dim == 0,
                "Unsupported Anima attention dimensions ", hidden, " and ", head_dim);

    // Patch features are flattened in (channel, patch row, patch column) order.
    // The extra input channel is the all-zero padding mask used for training.
    const auto mask = at::zeros_like(noisy.narrow(1, 0, 1));
    auto patches = at::cat({noisy, mask}, 1)
        .reshape({batch, latent_channels + 1, height, patch_size, width, patch_size})
        .permute({0, 2, 4, 1, 3, 5})
        .reshape({batch, height * width, (latent_channels + 1) * patch_size * patch_size});
    auto x = projections.linear(patches, "x_embedder.proj.1");
    const auto rope = image_rope(height, width, head_dim, x.options());

    const auto half = hidden / 2;
    const auto time_exponent = -std::log(10000.0) * at::arange(half, time.options().dtype(at::kFloat)) / half;
    const auto time_angles = time.to(at::kFloat).unsqueeze(-1) * at::exp(time_exponent);
    const auto sinusoid = at::cat({at::cos(time_angles), at::sin(time_angles)}, -1);
    const auto timestep_adaln = projections.linear(
        at::silu(projections.linear(sinusoid, "t_embedder.1.linear_1")), "t_embedder.1.linear_2");
    // Every block's adaLN modulation reads this frozen embedding; cast it once
    // instead of once per modulation linear (values are identical).
    const auto activated_timestep = at::silu(rms_norm(sinusoid, projections.frozen("t_embedding_norm.weight")))
        .to(x.scalar_type());

    int64_t block_count = 0;
    constexpr std::string_view block_prefix = "blocks.";
    constexpr std::string_view block_suffix = ".self_attn.q_proj.weight";
    for (const auto& entry : weights) {
        const auto& name = entry.first;
        if (name.size() >= block_prefix.size() + block_suffix.size() &&
            name.compare(0, block_prefix.size(), block_prefix) == 0 &&
            name.compare(name.size() - block_suffix.size(), block_suffix.size(), block_suffix) == 0) {
            ++block_count;
        }
    }
    for (int64_t i = 0; i < block_count; ++i) {
        x = transformer_block(
            projections, x, context, activated_timestep, timestep_adaln,
            "blocks." + std::to_string(i), hidden / head_dim, head_dim, rope);
    }

    const auto final = (
        modulation(projections, activated_timestep, "final_layer.adaln_modulation") +
        timestep_adaln.narrow(-1, 0, 2 * hidden)).chunk(2, -1);
    auto output = projections.linear(modulated_norm(x, final[0], final[1]), "final_layer.linear");
    // The output projection stores (patch row, patch column, channel) order.
    return output.reshape({batch, height, width, patch_size, patch_size, latent_channels})
        .permute({0, 5, 1, 3, 2, 4})
        .reshape({batch, latent_channels, 1, height * patch_size, width * patch_size});
}

}  // namespace anima
