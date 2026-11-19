// Anima generation DiT: Cosmos Predict2 MiniTrainDIT forward (2048 wide, 28 blocks, 16 heads, AdaLN-LoRA 256).
// dit: same op order and dtypes as ComfyUI, so a step matches it bit for bit (BF16).
// dit_fp8: the eight big projections per block as row-wise scaled FP8 GEMMs (weights per output channel from
// prepare_fp8, activations per row from the fused kernels in fp8.cu); everything else as in dit.
// Written from ComfyUI's comfy/ldm/cosmos/{predict2,position_embedding}.py (Apache-2.0) and the Anima config
// in comfy/model_detection.py (rope3d, h/w extrapolation 4.0, no fps modulation, padding-mask channel).
#include <ATen/ATen.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace dsh_kitchen {  // H3 plugin's kitchen/kitchen_ops.cpp (vendored comfy-kitchen kernel)
void rms_rope_split_half_(at::Tensor q, at::Tensor k, const at::Tensor& freqs, const at::Tensor& q_scale,
                          const at::Tensor& k_scale, double epsilon, int64_t rot_dim);
}

extern "C" int dsh_anima_modulate_fp8(const void* x, const void* shift, const void* scale, int64_t rows,
                                      int64_t rows_per_batch, int dim, float eps, void* out, float* row_scale,
                                      cudaStream_t stream);
extern "C" int dsh_anima_quant_fp8(const void* x, int64_t rows, int dim, int gelu, void* out, float* row_scale,
                                   cudaStream_t stream);

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

// ---- FP8 path -------------------------------------------------------------------------------------

const char* const kFp8Linears[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                                   "cross_attn.q_proj", "cross_attn.output_proj", "mlp.layer1", "mlp.layer2"};

struct Fp8 {  // FP8 rows [M, K] and their scales [M, 1]
    Tensor values, scales;
};

Fp8 fp8_empty(int64_t rows, int64_t dim, const Tensor& like) {
    return {at::empty({rows, dim}, like.options().dtype(at::kFloat8_e4m3fn)),
            at::empty({rows, 1}, like.options().dtype(at::kFloat))};
}

// LayerNorm * (1 + scale) + shift of x [B, L, D] -> FP8 rows (scale/shift [B, 1, D])
Fp8 modulate_fp8(const Tensor& x, const Tensor& scale, const Tensor& shift) {
    const auto rows = x.size(0) * x.size(1);
    auto out = fp8_empty(rows, kDim, x);
    auto sc = scale.contiguous(), sh = shift.contiguous();
    TORCH_CHECK(x.is_contiguous() && dsh_anima_modulate_fp8(x.data_ptr(), sh.data_ptr(), sc.data_ptr(), rows, x.size(1),
                                                            kDim, kEps, out.values.data_ptr(), out.scales.data_ptr<float>(),
                                                            c10::cuda::getCurrentCUDAStream()) == 0,
                "dsh_anima_gen: modulate_fp8 failed");
    return out;
}

Fp8 quant_fp8(const Tensor& x, bool gelu) {
    const auto dim = x.size(-1), rows = x.numel() / dim;
    auto in = x.contiguous();
    auto out = fp8_empty(rows, dim, x);
    TORCH_CHECK(dsh_anima_quant_fp8(in.data_ptr(), rows, dim, gelu, out.values.data_ptr(), out.scales.data_ptr<float>(),
                                    c10::cuda::getCurrentCUDAStream()) == 0, "dsh_anima_gen: quant_fp8 failed");
    return out;
}

Tensor linear_fp8(const Weights& w, const std::string& name, const Fp8& a, int64_t b) {
    return at::_scaled_mm(a.values, w.at(name + ".fp8").t(), a.scales, w.at(name + ".fp8_scale"), {}, {},
                          at::kBFloat16, false).view({b, a.values.size(0) / b, -1});
}

// Adds "<linear>.fp8" [N, K] e4m3 and "<linear>.fp8_scale" [1, N] (amax / 448 per output channel) for every
// block's big projections; the BF16 weights stay (the exact path and LoRA merges use them).
Weights prepare_fp8(const Weights& w) {
    auto out = w.copy();
    for (int64_t i = 0; w.contains("net.blocks." + std::to_string(i) + ".mlp.layer1.weight"); ++i) {
        for (const char* linear : kFp8Linears) {
            const auto name = "net.blocks." + std::to_string(i) + "." + linear;
            auto weight = w.at(name + ".weight").to(at::kFloat);
            auto scale = (weight.abs().amax({1}, true) / 448.0).clamp_min(1e-12);
            out.insert_or_assign(name + ".fp8", (weight / scale).to(at::kFloat8_e4m3fn));
            out.insert_or_assign(name + ".fp8_scale", scale.t().contiguous());
        }
    }
    return out;
}

// [B, H, S, D] attention through SDPA (ComfyUI's pytorch attention), back to [B, S, H*D]
Tensor attend(const Tensor& q, const Tensor& k, const Tensor& v) {
    auto out = at::scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2));
    return out.transpose(1, 2).reshape({q.size(0), q.size(1), kHeads * kHeadDim});
}

Tensor block(const Weights& w, const std::string& p, Tensor x, const Tensor& emb, const Tensor& lora,
             const Tensor& context, const Tensor& rope, bool fp8) {
    const auto b = x.size(0), l = x.size(1), lc = context.size(1);
    auto sa = modulation(w, p + ".adaln_modulation_self_attn", emb, lora).chunk(3, -1);
    auto ca = modulation(w, p + ".adaln_modulation_cross_attn", emb, lora).chunk(3, -1);
    auto ml = modulation(w, p + ".adaln_modulation_mlp", emb, lora).chunk(3, -1);  // shift, scale, gate

    // one projection of the modulated input: BF16 (exact) or FP8
    Tensor h;
    Fp8 h8;
    auto modulated = [&](const Tensor& scale, const Tensor& shift) {
        if (fp8) h8 = modulate_fp8(x, scale, shift);
        else h = modulate(x, scale, shift);
    };
    auto project = [&](const std::string& name) {
        return fp8 ? linear_fp8(w, p + "." + name, h8, b) : at::linear(h, w.at(p + "." + name + ".weight"));
    };
    auto output = [&](const std::string& name, const Tensor& a) {
        return fp8 ? linear_fp8(w, p + "." + name, quant_fp8(a, false), b) : at::linear(a, w.at(p + "." + name + ".weight"));
    };

    modulated(sa[1], sa[0]);
    auto q = project("self_attn.q_proj").view({b, l, kHeads, kHeadDim});
    auto k = project("self_attn.k_proj").view({b, l, kHeads, kHeadDim});
    auto v = project("self_attn.v_proj").view({b, l, kHeads, kHeadDim});
    dsh_kitchen::rms_rope_split_half_(q, k, rope, w.at(p + ".self_attn.q_norm.weight"),
                                      w.at(p + ".self_attn.k_norm.weight"), kEps, 0);
    auto r = output("self_attn.output_proj", attend(q, k, v));
    x = at::addcmul(x, sa[2], r);

    modulated(ca[1], ca[0]);
    q = at::rms_norm(project("cross_attn.q_proj").view({b, l, kHeads, kHeadDim}), {kHeadDim},
                     w.at(p + ".cross_attn.q_norm.weight"), kEps);
    k = at::rms_norm(at::linear(context, w.at(p + ".cross_attn.k_proj.weight")).view({b, lc, kHeads, kHeadDim}),
                     {kHeadDim}, w.at(p + ".cross_attn.k_norm.weight"), kEps);
    v = at::linear(context, w.at(p + ".cross_attn.v_proj.weight")).view({b, lc, kHeads, kHeadDim});
    r = output("cross_attn.output_proj", attend(q, k, v));
    x = at::addcmul(x, ca[2], r);

    modulated(ml[1], ml[0]);
    if (fp8) r = linear_fp8(w, p + ".mlp.layer2", quant_fp8(project("mlp.layer1"), true), b);
    else r = at::linear(at::gelu(project("mlp.layer1")), w.at(p + ".mlp.layer2.weight"));
    return at::addcmul(x, ml[2], r);
}

// x [B, 16, 1, H, W] latent (BF16), timesteps [B] (sigma for FLOW), context [B, 512, 1024] -> velocity [B, 16, 1, H, W]
Tensor forward(const Weights& w, const Tensor& x, const Tensor& timesteps, const Tensor& context, bool fp8) {
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
        hidden = block(w, "net.blocks." + std::to_string(i), hidden, emb, lora, context, rope, fp8);
    auto final_mod = modulation(w, "net.final_layer.adaln_modulation", emb, lora.narrow(-1, 0, 2 * kDim)).chunk(2, -1);
    auto out = at::linear(modulate(hidden.to(context.scalar_type()), final_mod[1], final_mod[0]),
                          w.at("net.final_layer.linear.weight"));  // [B, L, (p1 p2 t C)]
    return out.view({b, t, h, ww, kPatch, kPatch, 1, kOut}).permute({0, 7, 1, 6, 2, 4, 3, 5}).reshape(
        {b, kOut, t, h * kPatch, ww * kPatch});
}

Tensor dit_exact(const Weights& w, const Tensor& x, const Tensor& t, const Tensor& c) { return forward(w, x, t, c, false); }
Tensor dit_fp8(const Weights& w, const Tensor& x, const Tensor& t, const Tensor& c) { return forward(w, x, t, c, true); }

}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_anima_gen, m) {
    m.def("dit(Dict(str, Tensor) weights, Tensor x, Tensor timesteps, Tensor context) -> Tensor", &dit_exact);
    m.def("dit_fp8(Dict(str, Tensor) weights, Tensor x, Tensor timesteps, Tensor context) -> Tensor", &dit_fp8);
    m.def("prepare_fp8(Dict(str, Tensor) weights) -> Dict(str, Tensor)", &prepare_fp8);
}
