// Anima generation DiT: Cosmos Predict2 MiniTrainDIT forward (2048 wide, 28 blocks, 16 heads, AdaLN-LoRA 256),
// same op order and dtypes as ComfyUI so that a step matches it bit for bit. Inference only, BF16.
// Written from ComfyUI's comfy/ldm/cosmos/{predict2,position_embedding}.py (Apache-2.0) and the Anima config
// in comfy/model_detection.py (rope3d, h/w extrapolation 4.0, no fps modulation, padding-mask channel).
#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <string>

namespace dsh_kitchen {  // H3 plugin's kitchen/kitchen_ops.cpp (vendored comfy-kitchen kernel)
void rms_rope_split_half_(at::Tensor q, at::Tensor k, const at::Tensor& freqs, const at::Tensor& q_scale,
                          const at::Tensor& k_scale, double epsilon, int64_t rot_dim);
}

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

constexpr int64_t kDim = 2048, kHeads = 16, kHeadDim = 128, kOut = 16, kPatch = 2;
constexpr double kEps = 1e-6;

// VideoRopePosition3DEmb (image case): [T*H*W, 64, 2, 2] FP32, temporal 22 pairs, then h 21, then w 21.
Tensor rope_table(int64_t t, int64_t h, int64_t w, const at::TensorOptions& options) {
    const auto f32 = options.dtype(at::kFloat);
    constexpr int64_t dim_h = kHeadDim / 6 * 2, dim_t = kHeadDim - 2 * dim_h;
    auto spatial = at::arange(0, dim_h, 2, options.dtype(at::kLong)).narrow(0, 0, dim_h / 2).to(at::kFloat) / dim_h;
    auto temporal = at::arange(0, dim_t, 2, options.dtype(at::kLong)).narrow(0, 0, dim_t / 2).to(at::kFloat) / dim_t;
    const double spatial_theta = 10000.0 * std::pow(4.0, double(dim_h) / double(dim_h - 2));
    const double temporal_theta = 10000.0 * std::pow(1.0, double(dim_t) / double(dim_t - 2));
    auto h_freqs = at::reciprocal(at::pow(spatial_theta, spatial));  // Python's 1.0 / tensor
    auto t_freqs = at::reciprocal(at::pow(temporal_theta, temporal));
    auto seq = at::arange(std::max({h, w, t}), f32);
    auto table = [](const Tensor& angles) {
        auto c = angles.cos(), s = angles.sin();
        return at::stack({c, -s, s, c}, -1);  // [n, pairs, 4]
    };
    auto eh = table(at::outer(seq.narrow(0, 0, h), h_freqs));
    auto ew = table(at::outer(seq.narrow(0, 0, w), h_freqs));
    auto et = table(at::outer(seq.narrow(0, 0, t), t_freqs));
    auto em = at::cat({et.view({t, 1, 1, -1, 4}).expand({t, h, w, -1, 4}),
                       eh.view({1, h, 1, -1, 4}).expand({t, h, w, -1, 4}),
                       ew.view({1, 1, w, -1, 4}).expand({t, h, w, -1, 4})}, -2);
    return em.reshape({t * h * w, -1, 2, 2}).view({1, t * h * w, 1, -1, 2, 2});  // rope_emb_L_1_1_D.unsqueeze
}

// Timesteps (cos, sin) then TimestepEmbedding with AdaLN-LoRA: returns (emb [B,1,D] normed, adaln_lora [B,1,3D]).
std::pair<Tensor, Tensor> time_embedding(const Weights& w, const Tensor& timesteps, at::ScalarType dtype) {
    const auto half = kDim / 2;
    auto ts = timesteps.flatten().to(at::kFloat);
    auto exponent = at::arange(half, ts.options()) * (-std::log(10000.0));
    exponent = exponent / double(half);
    auto freqs = at::outer(ts, at::exp(exponent));
    auto sample = at::cat({freqs.cos(), freqs.sin()}, -1).unsqueeze(1).to(dtype);  // [B, 1, D]
    auto lora = at::linear(at::silu(at::linear(sample, w.at("net.t_embedder.1.linear_1.weight"))),
                           w.at("net.t_embedder.1.linear_2.weight"));
    return {at::rms_norm(sample, {kDim}, w.at("net.t_embedding_norm.weight"), kEps), lora};
}

Tensor modulation(const Weights& w, const std::string& p, const Tensor& emb, const Tensor& lora) {
    return at::linear(at::linear(at::silu(emb), w.at(p + ".1.weight")), w.at(p + ".2.weight")) + lora;
}

Tensor modulate(const Tensor& x, const Tensor& scale, const Tensor& shift) {
    return at::layer_norm(x, {kDim}, {}, {}, kEps) * (1 + scale) + shift;
}

// [B, H, S, D] attention through SDPA (ComfyUI's pytorch attention), back to [B, S, H*D]
Tensor attend(const Tensor& q, const Tensor& k, const Tensor& v) {
    auto out = at::scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2));
    return out.transpose(1, 2).reshape({q.size(0), q.size(1), kHeads * kHeadDim});
}

Tensor block(const Weights& w, const std::string& p, Tensor x, const Tensor& emb, const Tensor& lora,
             const Tensor& context, const Tensor& rope) {
    const auto b = x.size(0), l = x.size(1), lc = context.size(1);
    auto sa = modulation(w, p + ".adaln_modulation_self_attn", emb, lora).chunk(3, -1);
    auto ca = modulation(w, p + ".adaln_modulation_cross_attn", emb, lora).chunk(3, -1);
    auto ml = modulation(w, p + ".adaln_modulation_mlp", emb, lora).chunk(3, -1);  // shift, scale, gate

    auto h = modulate(x, sa[1], sa[0]);
    auto q = at::linear(h, w.at(p + ".self_attn.q_proj.weight")).view({b, l, kHeads, kHeadDim});
    auto k = at::linear(h, w.at(p + ".self_attn.k_proj.weight")).view({b, l, kHeads, kHeadDim});
    auto v = at::linear(h, w.at(p + ".self_attn.v_proj.weight")).view({b, l, kHeads, kHeadDim});
    dsh_kitchen::rms_rope_split_half_(q, k, rope, w.at(p + ".self_attn.q_norm.weight"),
                                      w.at(p + ".self_attn.k_norm.weight"), kEps, 0);
    auto r = at::linear(attend(q, k, v), w.at(p + ".self_attn.output_proj.weight"));
    x = at::addcmul(x, sa[2], r);

    h = modulate(x, ca[1], ca[0]);
    q = at::rms_norm(at::linear(h, w.at(p + ".cross_attn.q_proj.weight")).view({b, l, kHeads, kHeadDim}), {kHeadDim},
                     w.at(p + ".cross_attn.q_norm.weight"), kEps);
    k = at::rms_norm(at::linear(context, w.at(p + ".cross_attn.k_proj.weight")).view({b, lc, kHeads, kHeadDim}),
                     {kHeadDim}, w.at(p + ".cross_attn.k_norm.weight"), kEps);
    v = at::linear(context, w.at(p + ".cross_attn.v_proj.weight")).view({b, lc, kHeads, kHeadDim});
    r = at::linear(attend(q, k, v), w.at(p + ".cross_attn.output_proj.weight"));
    x = at::addcmul(x, ca[2], r);

    h = modulate(x, ml[1], ml[0]);
    r = at::linear(at::gelu(at::linear(h, w.at(p + ".mlp.layer1.weight"))), w.at(p + ".mlp.layer2.weight"));
    return at::addcmul(x, ml[2], r);
}

// x [B, 16, 1, H, W] latent (BF16), timesteps [B] (sigma for FLOW), context [B, 512, 1024] -> velocity [B, 16, 1, H, W]
Tensor forward(const Weights& w, const Tensor& x, const Tensor& timesteps, const Tensor& context) {
    TORCH_CHECK(x.dim() == 5 && x.size(1) == kOut && x.size(3) % kPatch == 0 && x.size(4) % kPatch == 0,
                "dsh_anima_gen.dit: x must be [B, 16, T, H, W] with even H and W");
    const auto dtype = w.at("net.x_embedder.proj.1.weight").scalar_type();
    const auto b = x.size(0), t = x.size(2), h = x.size(3) / kPatch, ww = x.size(4) / kPatch, l = t * h * ww;
    // concat_padding_mask (zeros), then "b c t (h m) (w n) -> b t h w (c m n)" and the patch linear
    auto padded = at::cat({x, at::zeros({b, 1, t, x.size(3), x.size(4)}, x.options())}, 1);
    auto patches = padded.view({b, kOut + 1, t, h, kPatch, ww, kPatch}).permute({0, 2, 3, 5, 1, 4, 6}).reshape({b, l, -1});
    auto hidden = at::linear(patches, w.at("net.x_embedder.proj.1.weight"));
    const auto rope = rope_table(t, h, ww, x.options());
    auto [emb, lora] = time_embedding(w, timesteps, dtype);
    for (int64_t i = 0; w.contains("net.blocks." + std::to_string(i) + ".mlp.layer1.weight"); ++i)
        hidden = block(w, "net.blocks." + std::to_string(i), hidden, emb, lora, context, rope);
    auto final_mod = modulation(w, "net.final_layer.adaln_modulation", emb, lora.narrow(-1, 0, 2 * kDim)).chunk(2, -1);
    auto out = at::linear(modulate(hidden.to(context.scalar_type()), final_mod[1], final_mod[0]),
                          w.at("net.final_layer.linear.weight"));  // [B, L, (p1 p2 t C)]
    return out.view({b, t, h, ww, kPatch, kPatch, 1, kOut}).permute({0, 7, 1, 6, 2, 4, 3, 5}).reshape(
        {b, kOut, t, h * kPatch, ww * kPatch});
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_anima_gen, m) {
    m.def("dit(Dict(str, Tensor) weights, Tensor x, Tensor timesteps, Tensor context) -> Tensor", &forward);
}
