// MiniMax H3 / P-H3 image text conditioning: the Qwen3-VL language model (32B truncated after layer 50 for
// H3, 4B for P-H3; text only). Shapes come from the weights. Written from ComfyUI's comfy/text_encoders/{llama,qwen3vl,minimax}.py
// (GPL-3.0) and comfy-kitchen's INT8 ConvRot linear semantics (Apache-2.0).
//
// Weights are the released int8 ConvRot checkpoint: every projection is int8 [N, K] with a
// per-output-row FP32 scale and was rotated offline by a 256-wide regular Hadamard. As in
// ComfyUI, each projection is dequantized on the fly and the encoder runs in FP32.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

constexpr int64_t kHeadDim = 128, kH3Layers = 50;
constexpr double kEps = 1e-6, kRopeTheta = 5000000.0;
constexpr int64_t kGroup = 256;

Tensor hadamard(const at::TensorOptions& options) {
    // Normalized regular Hadamard (kron powers of the 4x4 regular matrix); symmetric, so H^T = H.
    auto h4 = at::tensor({1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1}, at::kFloat).view({4, 4});
    auto h = h4;
    while (h.size(0) < kGroup) h = at::kron(h, h4);
    return (h / std::sqrt(double(kGroup))).to(options);
}

struct Int8Linear {
    Tensor weight;  // int8 [N, K]
    Tensor scale;   // f32 [1, N]
};

// ComfyUI runs this checkpoint by dequantizing each projection (W = (q * scale) H) and computing
// in FP32 without quantizing activations. H is symmetric, so x W^T = (x H) (q * scale)^T: rotate
// the few activation rows instead of the whole weight, and fold the row scale into the output.
Tensor dequant_linear(const Tensor& x, const Int8Linear& w, const Tensor& h) {
    auto shape = x.sizes().vec();
    const auto k = shape.back();
    auto rows = x.reshape({-1, k});
    auto rotated = at::matmul(rows.view({rows.size(0), k / kGroup, kGroup}), h).view({rows.size(0), k});
    auto out = at::linear(rotated, w.weight.to(at::kFloat)) * w.scale;
    shape.back() = w.weight.size(0);
    return out.view(shape);
}

Int8Linear projection(const Weights& weights, const std::string& name) {
    return {weights.at(name + ".weight"), weights.at(name + ".weight_scale").to(at::kFloat).view({1, -1})};
}

Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin) {
    // split-half rotation in FP32 ([T, heads, 128] with cos/sin [T, 1, 128])
    auto f = x.to(at::kFloat);
    auto halves = f.chunk(2, -1);
    auto rotated = at::cat({-halves[1], halves[0]}, -1);
    return (f * cos + rotated * sin).to(x.scalar_type());
}

// Runs `layers` decoder layers. Returns the input of every layer in `taps` (in order) and, last, the output of
// the final layer (before the final norm). Weights are under `prefix` ("model." or "model.language_model.").
std::vector<Tensor> run(const Weights& weights, const std::string& prefix, const Tensor& token_ids, int64_t layers,
                        const std::vector<int64_t>& taps) {
    TORCH_CHECK(token_ids.dim() == 1 && token_ids.numel() > 0, "text encoder: expected 1-D token ids");
    const auto embed = weights.at(prefix + "embed_tokens.weight");
    const auto options = embed.options().dtype(at::kFloat);
    const int64_t kHidden = embed.size(1);
    const int64_t kHeads = weights.at(prefix + "layers.0.self_attn.q_proj.weight").size(0) / kHeadDim;
    const int64_t kKvHeads = weights.at(prefix + "layers.0.self_attn.k_proj.weight").size(0) / kHeadDim;
    const auto dtype = at::kFloat;
    const auto linear = dequant_linear;
    const auto h = hadamard(options);
    const auto t = token_ids.size(0);

    auto inv_freq = at::reciprocal(at::pow(kRopeTheta, at::arange(0, kHeadDim, 2, options.dtype(at::kFloat)) / kHeadDim));
    auto angles = at::outer(at::arange(t, options.dtype(at::kFloat)), inv_freq);
    angles = at::cat({angles, angles}, -1).unsqueeze(1);
    const auto cos = angles.cos(), sin = angles.sin();

    Tensor x;
    if (weights.contains(prefix + "embed_tokens.weight_scale")) {  // int8 convrot table: dequantize and un-rotate the rows
        auto ids = token_ids.to(embed.device());
        auto rows = embed.index_select(0, ids).to(at::kFloat) *
                    weights.at(prefix + "embed_tokens.weight_scale").to(at::kFloat).view({-1, 1}).index_select(0, ids);
        // ComfyUI dequantizes the rows into the checkpoint's original dtype (BF16)
        x = at::matmul(rows.view({t, kHidden / kGroup, kGroup}), h).view({t, kHidden}).to(at::kBFloat16).to(at::kFloat);
    } else {
        x = embed.index_select(0, token_ids.to(embed.device())).to(at::kFloat);
    }
    std::vector<Tensor> out;
    for (int64_t i = 0; i < layers; ++i) {
        if (std::find(taps.begin(), taps.end(), i) != taps.end()) out.push_back(x.clone());
        const auto p = prefix + "layers." + std::to_string(i);
        auto normed = at::rms_norm(x, {kHidden}, weights.at(p + ".input_layernorm.weight").to(dtype), kEps);
        auto q = linear(normed, projection(weights, p + ".self_attn.q_proj"), h).view({t, kHeads, kHeadDim});
        auto k = linear(normed, projection(weights, p + ".self_attn.k_proj"), h).view({t, kKvHeads, kHeadDim});
        auto v = linear(normed, projection(weights, p + ".self_attn.v_proj"), h).view({t, kKvHeads, kHeadDim});
        q = rope(at::rms_norm(q, {kHeadDim}, weights.at(p + ".self_attn.q_norm.weight").to(dtype), kEps), cos, sin);
        k = rope(at::rms_norm(k, {kHeadDim}, weights.at(p + ".self_attn.k_norm.weight").to(dtype), kEps), cos, sin);
        auto attended = at::scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
            {}, 0.0, /*is_causal=*/true, std::nullopt, /*enable_gqa=*/true);
        attended = attended.squeeze(0).transpose(0, 1).reshape({t, kHeads * kHeadDim});
        x = x + linear(attended, projection(weights, p + ".self_attn.o_proj"), h);

        normed = at::rms_norm(x, {kHidden}, weights.at(p + ".post_attention_layernorm.weight").to(dtype), kEps);
        auto gate = linear(normed, projection(weights, p + ".mlp.gate_proj"), h);
        auto up = linear(normed, projection(weights, p + ".mlp.up_proj"), h);
        x = x + linear(at::silu(gate) * up, projection(weights, p + ".mlp.down_proj"), h);
    }
    out.push_back(x);
    return out;
}

Tensor encode(const Weights& weights, const Tensor& token_ids) {
    TORCH_CHECK(token_ids.numel() > 16, "H3 text encoder: expected more than 16 tokens");
    return run(weights, "model.", token_ids, kH3Layers, {}).back();  // unnormalized hidden state after layer 50
}

Tensor hadamard256() { return hadamard(at::TensorOptions().dtype(at::kFloat).device(at::kCUDA)); }

std::vector<Tensor> encode_layers(const Weights& weights, const std::string& prefix, const Tensor& token_ids, int64_t layers,
                                  std::vector<int64_t> taps) {
    return run(weights, prefix, token_ids, layers, taps);
}


}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_h3, m) {
    m.def("encode_text(Dict(str, Tensor) weights, Tensor token_ids) -> Tensor", &encode);
    m.def("encode_text_layers(Dict(str, Tensor) weights, str prefix, Tensor token_ids, int layers, int[] taps) -> Tensor[]",
          &encode_layers);
    m.def("hadamard256() -> Tensor", &hadamard256);  // the ConvRot rotation (symmetric), FP32 on CUDA

}
