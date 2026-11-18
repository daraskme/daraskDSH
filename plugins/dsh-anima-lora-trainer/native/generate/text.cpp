// Anima generation text conditioning: Qwen3-0.6B (FP32, last layer + final norm) and the DiT's LLM adapter (BF16),
// then T5 weights on the adapter rows and zero padding to 512 -- the c_crossattn ComfyUI feeds the DiT.
// Written from ComfyUI's comfy/text_encoders/{llama,anima}.py and comfy/ldm/anima/model.py (GPL-3.0).
#include <ATen/ATen.h>
#include <torch/library.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <limits>
#include <string>

namespace dsh_kitchen {  // H3 plugin's kitchen/kitchen_ops.cpp (vendored comfy-kitchen kernel)
void apply_rope_split_half1(const at::Tensor& x, const at::Tensor& freqs, at::Tensor out);
}

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

constexpr int64_t kQwenLayers = 28, kQwenHeads = 16, kQwenKv = 8, kHeadDim = 128, kHidden = 1024;
constexpr double kQwenEps = 1e-6, kQwenTheta = 1000000.0;
constexpr int64_t kAdapterBlocks = 6, kAdapterHeads = 16, kAdapterHeadDim = 64, kContext = 512;

// ---- Qwen3-0.6B ---------------------------------------------------------------------------------

Tensor qwen_rope(const Tensor& x, const Tensor& matrix) {
    if (x.scalar_type() == at::kFloat) {  // the kitchen kernel is FP16/BF16 only; FP32 uses the eager split-half form
        const auto half = x.size(-1) / 2;
        auto a = x.narrow(-1, 0, half), b = x.narrow(-1, half, half);
        auto m = matrix.squeeze(0);  // [1, S, 64, 2, 2]
        // comfy-kitchen eager apply_rope_split_half1: diagonal product, then addcmul_ (same rounding order)
        auto out_a = a * m.select(-2, 0).select(-1, 0);
        out_a.addcmul_(b, m.select(-2, 0).select(-1, 1));
        auto out_b = b * m.select(-2, 1).select(-1, 1);
        out_b.addcmul_(a, m.select(-2, 1).select(-1, 0));
        return at::cat({out_a, out_b}, -1);
    }
    auto out = at::empty_like(x);
    dsh_kitchen::apply_rope_split_half1(x, matrix, out);
    return out;
}

Tensor qwen(const Weights& w, const Tensor& ids) {
    const auto& embed = w.at("model.embed_tokens.weight");
    const auto dtype = embed.scalar_type();
    const auto s = ids.size(0);
    const auto f32 = embed.options().dtype(at::kFloat);
    // precompute_freqs_cis + rope_matrix: [1, 1, S, 64, 2, 2] (cos, -sin, sin, cos) in FP32
    auto inv = at::reciprocal(at::pow(kQwenTheta, at::arange(0, kHeadDim, 2, f32) / kHeadDim));
    auto freqs = at::outer(at::arange(s, f32), inv);
    auto emb = at::cat({freqs, freqs}, -1);
    auto cos = emb.cos(), sin = emb.sin();
    const auto half = kHeadDim / 2;
    auto matrix = at::stack({cos.narrow(-1, 0, half), -sin.narrow(-1, half, half), sin.narrow(-1, 0, half),
                             cos.narrow(-1, half, half)}, -1).reshape({1, 1, s, half, 2, 2});
    // causal additive mask: finfo(dtype).min / 4 above the diagonal (as ComfyUI's llama.py)
    const double lowest = dtype == at::kBFloat16 ? double(std::numeric_limits<c10::BFloat16>::lowest())
                          : dtype == at::kHalf   ? double(std::numeric_limits<c10::Half>::lowest())
                                                 : double(std::numeric_limits<float>::lowest());
    auto mask = at::empty({s, s}, embed.options()).fill_(lowest / 4).triu_(1).view({1, 1, s, s});

    auto x = embed.index_select(0, ids.to(embed.device())).unsqueeze(0);  // [1, S, 1024]
    for (int64_t i = 0; i < kQwenLayers; ++i) {
        const auto p = "model.layers." + std::to_string(i);
        auto h = at::rms_norm(x, {kHidden}, w.at(p + ".input_layernorm.weight"), kQwenEps);
        auto q = at::linear(h, w.at(p + ".self_attn.q_proj.weight")).view({1, s, kQwenHeads, kHeadDim}).transpose(1, 2);
        auto k = at::linear(h, w.at(p + ".self_attn.k_proj.weight")).view({1, s, kQwenKv, kHeadDim}).transpose(1, 2);
        auto v = at::linear(h, w.at(p + ".self_attn.v_proj.weight")).view({1, s, kQwenKv, kHeadDim}).transpose(1, 2);
        q = qwen_rope(at::rms_norm(q, {kHeadDim}, w.at(p + ".self_attn.q_norm.weight"), kQwenEps), matrix);
        k = qwen_rope(at::rms_norm(k, {kHeadDim}, w.at(p + ".self_attn.k_norm.weight"), kQwenEps), matrix);
        k = k.repeat_interleave(kQwenHeads / kQwenKv, 1);
        v = v.repeat_interleave(kQwenHeads / kQwenKv, 1);
        auto a = at::scaled_dot_product_attention(q, k, v, mask);
        x = x + at::linear(a.transpose(1, 2).reshape({1, s, kQwenHeads * kHeadDim}), w.at(p + ".self_attn.o_proj.weight"));
        h = at::rms_norm(x, {kHidden}, w.at(p + ".post_attention_layernorm.weight"), kQwenEps);
        auto gate = at::linear(h, w.at(p + ".mlp.gate_proj.weight"));
        auto up = at::linear(h, w.at(p + ".mlp.up_proj.weight"));
        x = x + at::linear(at::silu(gate) * up, w.at(p + ".mlp.down_proj.weight"));
    }
    return at::rms_norm(x, {kHidden}, w.at("model.norm.weight"), kQwenEps).to(dtype);  // [1, S, 1024]
}

// ---- LLM adapter (comfy/ldm/anima/model.py) -----------------------------------------------------

std::pair<Tensor, Tensor> adapter_rope(int64_t length, const at::TensorOptions& options) {
    auto f32 = options.dtype(at::kFloat);
    auto inv = 1.0 / at::pow(10000.0, at::arange(0, kAdapterHeadDim, 2, options.dtype(at::kLong)).to(at::kFloat) / kAdapterHeadDim);
    auto freqs = at::outer(at::arange(length, f32), inv);
    auto emb = at::cat({freqs, freqs}, -1);
    return {emb.cos().to(options.dtype()).unsqueeze(0).unsqueeze(1), emb.sin().to(options.dtype()).unsqueeze(0).unsqueeze(1)};
}

Tensor rotate_half(const Tensor& x) {
    auto halves = x.chunk(2, -1);
    return at::cat({-halves[1], halves[0]}, -1);
}

Tensor adapter_attention(const Weights& w, const std::string& p, const Tensor& x, const Tensor& context,
                         const std::pair<Tensor, Tensor>& rope_q, const std::pair<Tensor, Tensor>& rope_k) {
    const auto lq = x.size(1), lk = context.size(1);
    auto q = at::rms_norm(at::linear(x, w.at(p + ".q_proj.weight")).view({1, lq, kAdapterHeads, kAdapterHeadDim}),
                          {kAdapterHeadDim}, w.at(p + ".q_norm.weight"), 1e-6).transpose(1, 2);
    auto k = at::rms_norm(at::linear(context, w.at(p + ".k_proj.weight")).view({1, lk, kAdapterHeads, kAdapterHeadDim}),
                          {kAdapterHeadDim}, w.at(p + ".k_norm.weight"), 1e-6).transpose(1, 2);
    auto v = at::linear(context, w.at(p + ".v_proj.weight")).view({1, lk, kAdapterHeads, kAdapterHeadDim}).transpose(1, 2);
    q = q * rope_q.first + rotate_half(q) * rope_q.second;
    k = k * rope_k.first + rotate_half(k) * rope_k.second;
    auto a = at::scaled_dot_product_attention(q, k, v);
    return at::linear(a.transpose(1, 2).reshape({1, lq, -1}).contiguous(), w.at(p + ".o_proj.weight"));
}

Tensor adapter(const Weights& w, const Tensor& hidden, const Tensor& t5_ids) {
    const std::string a = "net.llm_adapter.";
    auto x = w.at(a + "embed.weight").index_select(0, t5_ids.to(hidden.device())).unsqueeze(0).to(hidden.scalar_type());
    const auto rope_x = adapter_rope(x.size(1), x.options());
    const auto rope_c = adapter_rope(hidden.size(1), x.options());
    for (int64_t i = 0; i < kAdapterBlocks; ++i) {
        const auto p = a + "blocks." + std::to_string(i);
        auto normed = at::rms_norm(x, {kHidden}, w.at(p + ".norm_self_attn.weight"), 1e-6);
        x = x + adapter_attention(w, p + ".self_attn", normed, normed, rope_x, rope_x);
        normed = at::rms_norm(x, {kHidden}, w.at(p + ".norm_cross_attn.weight"), 1e-6);
        x = x + adapter_attention(w, p + ".cross_attn", normed, hidden, rope_x, rope_c);
        auto m = at::gelu(at::linear(at::rms_norm(x, {kHidden}, w.at(p + ".norm_mlp.weight"), 1e-6),
                                     w.at(p + ".mlp.0.weight"), w.at(p + ".mlp.0.bias")));
        x = x + at::linear(m, w.at(p + ".mlp.2.weight"), w.at(p + ".mlp.2.bias"));
    }
    return at::rms_norm(at::linear(x, w.at(a + "out_proj.weight"), w.at(a + "out_proj.bias")), {kHidden},
                        w.at(a + "norm.weight"), 1e-6);
}

// c_crossattn for one prompt: [1, max(512, L_t5), 1024] in the model dtype.
Tensor crossattn(const Weights& qwen_w, const Weights& dit_w, const Tensor& qwen_ids, const Tensor& t5_ids,
                 const Tensor& t5_weights) {
    // ComfyUI runs Qwen in FP32 (pass FP32 weights) and hands the DiT's adapter its BF16 cast.
    const auto model_dtype = dit_w.at("net.llm_adapter.embed.weight").scalar_type();
    auto hidden = qwen(qwen_w, qwen_ids).to(model_dtype);
    auto out = adapter(dit_w, hidden, t5_ids);
    out = out * t5_weights.to(out.device()).view({1, -1, 1}).to(out.scalar_type());
    if (out.size(1) < kContext) out = at::constant_pad_nd(out, {0, 0, 0, kContext - out.size(1)});
    return out;
}

Tensor qwen_op(const Weights& w, const Tensor& ids) { return qwen(w, ids); }
Tensor adapter_op(const Weights& w, const Tensor& hidden, const Tensor& t5_ids) { return adapter(w, hidden, t5_ids); }

}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_anima_gen, m) {
    m.def("qwen(Dict(str, Tensor) weights, Tensor ids) -> Tensor", &qwen_op);
    m.def("adapter(Dict(str, Tensor) weights, Tensor hidden, Tensor t5_ids) -> Tensor", &adapter_op);
    m.def("crossattn(Dict(str, Tensor) qwen, Dict(str, Tensor) dit, Tensor qwen_ids, Tensor t5_ids, Tensor t5_weights) -> Tensor",
          &crossattn);
}
