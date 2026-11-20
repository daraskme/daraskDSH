// MiniMax H3 / P-H3 image text conditioning: the Qwen3-VL language model (32B truncated after layer 50 for
// H3, 4B for P-H3), with Qwen3-VL image rows (MRoPE + DeepStack) when the prompt has pictures. Shapes come from the weights. Written from ComfyUI's comfy/text_encoders/{llama,qwen3vl,minimax}.py
// (GPL-3.0) and comfy-kitchen's INT8 ConvRot linear semantics (Apache-2.0).
//
// Weights are the released int8 ConvRot checkpoint: every projection is int8 [N, K] with a
// per-output-row FP32 scale and was rotated offline by a 256-wide regular Hadamard. As in
// ComfyUI, each projection is dequantized on the fly and the encoder runs in FP32.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <optional>
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

// Token embeddings in FP32 ([T, hidden]); the int8 convrot table is dequantized like ComfyUI (into BF16).
Tensor embed(const Weights& weights, const std::string& prefix, const Tensor& token_ids) {
    const auto table = weights.at(prefix + "embed_tokens.weight");
    const auto ids = token_ids.to(table.device());
    if (!weights.contains(prefix + "embed_tokens.weight_scale")) return table.index_select(0, ids).to(at::kFloat);
    const int64_t t = ids.size(0), hidden = table.size(1);
    auto rows = table.index_select(0, ids).to(at::kFloat) *
                weights.at(prefix + "embed_tokens.weight_scale").to(at::kFloat).view({-1, 1}).index_select(0, ids);
    const auto h = hadamard(table.options().dtype(at::kFloat));
    return at::matmul(rows.view({t, hidden / kGroup, kGroup}), h).view({t, hidden}).to(at::kBFloat16).to(at::kFloat);
}

// Runs `layers` decoder layers on FP32 input embeddings x [T, hidden]. Returns the input of every layer in `taps` (in
// order) and, last, the output of the final layer (before the final norm). Weights are under `prefix`.
// positions: [3, T] Qwen3-VL MRoPE ids (interleaved sections 24/20/20) or undefined for plain 0..T-1.
// deepstack: features added to the image rows (`visual`, bool [T]) after the first deepstack.size() layers.
std::vector<Tensor> run(const Weights& weights, const std::string& prefix, Tensor x, const std::optional<Tensor>& positions,
                        int64_t layers, const std::vector<int64_t>& taps, const std::vector<Tensor>& deepstack,
                        const std::optional<Tensor>& visual) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 0, "text encoder: expected [T, hidden] embeddings");
    const auto options = x.options().dtype(at::kFloat);
    const int64_t kHidden = x.size(1);
    const int64_t kHeads = weights.at(prefix + "layers.0.self_attn.q_proj.weight").size(0) / kHeadDim;
    const int64_t kKvHeads = weights.at(prefix + "layers.0.self_attn.k_proj.weight").size(0) / kHeadDim;
    const auto dtype = at::kFloat;
    const auto linear = dequant_linear;
    const auto h = hadamard(options);
    const auto t = x.size(0);

    auto inv_freq = at::reciprocal(at::pow(kRopeTheta, at::arange(0, kHeadDim, 2, options.dtype(at::kFloat)) / kHeadDim));
    Tensor angles;
    if (positions) {  // precompute_freqs_cis with interleaved MRoPE: T by default, H / W on every 3rd frequency
        auto freqs = positions->to(options).unsqueeze(-1) * inv_freq.view({1, 1, -1});  // [3, T, 64]
        angles = freqs[0].clone();
        using at::indexing::Slice;
        angles.index_put_({Slice(), Slice(1, 60, 3)}, freqs[1].index({Slice(), Slice(1, 60, 3)}));
        angles.index_put_({Slice(), Slice(2, 60, 3)}, freqs[2].index({Slice(), Slice(2, 60, 3)}));
    } else {
        angles = at::outer(at::arange(t, options.dtype(at::kFloat)), inv_freq);
    }
    angles = at::cat({angles, angles}, -1).unsqueeze(1);
    const auto cos = angles.cos(), sin = angles.sin();
    Tensor rows;
    if (!deepstack.empty()) rows = visual->to(x.device()).nonzero().squeeze(1);

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
        if (i < int64_t(deepstack.size())) x.index_add_(0, rows, deepstack[i].to(x));
    }
    out.push_back(x);
    return out;
}

std::vector<Tensor> run_ids(const Weights& weights, const std::string& prefix, const Tensor& token_ids, int64_t layers,
                            const std::vector<int64_t>& taps) {
    TORCH_CHECK(token_ids.dim() == 1 && token_ids.numel() > 0, "text encoder: expected 1-D token ids");
    return run(weights, prefix, embed(weights, prefix, token_ids), std::nullopt, layers, taps, {}, std::nullopt);
}

Tensor encode(const Weights& weights, const Tensor& token_ids) {
    TORCH_CHECK(token_ids.numel() > 16, "H3 text encoder: expected more than 16 tokens");
    return run_ids(weights, "model.", token_ids, kH3Layers, {}).back();  // unnormalized hidden state after layer 50
}

Tensor hadamard256() { return hadamard(at::TensorOptions().dtype(at::kFloat).device(at::kCUDA)); }

std::vector<Tensor> encode_layers(const Weights& weights, const std::string& prefix, const Tensor& token_ids, int64_t layers,
                                  std::vector<int64_t> taps) {
    return run_ids(weights, prefix, token_ids, layers, taps);
}

Tensor embed_tokens(const Weights& weights, const std::string& prefix, const Tensor& token_ids) {
    return embed(weights, prefix, token_ids);
}

std::vector<Tensor> encode_embeds(const Weights& weights, const std::string& prefix, const Tensor& embeds,
                                  const std::optional<Tensor>& positions, int64_t layers, std::vector<int64_t> taps,
                                  std::vector<Tensor> deepstack, const std::optional<Tensor>& visual) {
    return run(weights, prefix, embeds.to(at::kFloat), positions, layers, taps, deepstack, visual);
}


}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_h3, m) {
    m.def("encode_text(Dict(str, Tensor) weights, Tensor token_ids) -> Tensor", &encode);
    m.def("encode_text_layers(Dict(str, Tensor) weights, str prefix, Tensor token_ids, int layers, int[] taps) -> Tensor[]",
          &encode_layers);
    m.def("hadamard256() -> Tensor", &hadamard256);  // the ConvRot rotation (symmetric), FP32 on CUDA
    m.def("embed_tokens(Dict(str, Tensor) weights, str prefix, Tensor token_ids) -> Tensor", &embed_tokens);
    // embeds [T, hidden] with the image rows spliced in; positions [3, T] (MRoPE) or None; deepstack added at `visual`
    m.def("encode_embeds(Dict(str, Tensor) weights, str prefix, Tensor embeds, Tensor? positions, int layers, int[] taps, "
          "Tensor[] deepstack, Tensor? visual) -> Tensor[]", &encode_embeds);  // the ConvRot rotation (symmetric), FP32 on CUDA

}
