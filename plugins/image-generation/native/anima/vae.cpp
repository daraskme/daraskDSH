// Anima image decode and encode: the Wan 2.1 (qwen_image_vae) decoder/encoder for a single frame, as ComfyUI's
// VAEDecode / VAEEncode run it. Decode:
// T = 1, so no feature cache, every causal 3D conv keeps only its last temporal slice ("causal_zero"), and the
// upsample3d time convs are skipped. Weights in the VAE dtype (ComfyUI: BF16, bit-exact), output [H, W, 3]
// in [0, 1] FP32. Written from ComfyUI's comfy/ldm/wan/vae.py, comfy/sd.py and diffusionmodules/model.py.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

bool has(const Weights& w, const std::string& key) { return w.contains(key); }

// CausalConv3d with T = 1 and no cache: weight[:, :, -1:], padding (0, p, p)
Tensor causal_conv(const Weights& w, const std::string& p, const Tensor& x) {
    const auto& weight = w.at(p + ".weight");
    const int64_t pad = weight.size(3) / 2;
    return at::conv3d(x, weight.narrow(2, weight.size(2) - x.size(2), x.size(2)), w.at(p + ".bias"), {1, 1, 1},
                      {0, pad, pad});
}

// RMS_norm: F.normalize over channels * sqrt(dim) * gamma
Tensor rms(const Weights& w, const std::string& p, const Tensor& x) {
    const auto& gamma = w.at(p + ".gamma");
    auto denom = x.norm(2, {1}, true).clamp_min(1e-12).expand_as(x);
    return x / denom * std::sqrt(double(gamma.size(0))) * gamma.to(x.scalar_type());
}

Tensor residual(const Weights& w, const std::string& p, const Tensor& x) {
    auto h = causal_conv(w, p + ".residual.2", at::silu(rms(w, p + ".residual.0", x)));
    h = causal_conv(w, p + ".residual.6", at::silu(rms(w, p + ".residual.3", h)));
    return h + (has(w, p + ".shortcut.weight") ? causal_conv(w, p + ".shortcut", x) : x);
}

// Single-head spatial attention through SDPA with ComfyUI's backend priority (flash, cuDNN, efficient, math).
Tensor attention(const Weights& w, const std::string& p, const Tensor& x) {
    auto identity = x;
    auto h = rms(w, p + ".norm", x.squeeze(2));  // (b t) c h w with b = t = 1
    auto qkv = at::conv2d(h, w.at(p + ".to_qkv.weight"), w.at(p + ".to_qkv.bias")).chunk(3, 1);
    const auto shape = qkv[0].sizes().vec();
    const auto b = shape[0], c = shape[1];
    auto heads = [&](const Tensor& t) { return t.view({b, 1, c, -1}).transpose(2, 3).contiguous(); };
    auto& ctx = at::globalContext();
    const auto previous = ctx.sDPPriorityOrder();
    std::vector<int64_t> order = {int64_t(at::SDPBackend::flash_attention), int64_t(at::SDPBackend::cudnn_attention),
                                  int64_t(at::SDPBackend::efficient_attention), int64_t(at::SDPBackend::math)};
    std::vector<int64_t> old;
    for (auto backend : previous) old.push_back(int64_t(backend));
    for (auto backend : old)
        if (std::find(order.begin(), order.end(), backend) == order.end()) order.push_back(backend);
    ctx.setSDPPriorityOrder(order);
    auto a = at::scaled_dot_product_attention(heads(qkv[0]), heads(qkv[1]), heads(qkv[2]));
    ctx.setSDPPriorityOrder(old);
    a = a.transpose(2, 3).reshape(shape);
    return at::conv2d(a, w.at(p + ".proj.weight"), w.at(p + ".proj.bias")).unsqueeze(2) + identity;
}

Tensor upsample(const Weights& w, const std::string& p, const Tensor& x) {
    auto h = at::_upsample_nearest_exact2d(x.squeeze(2), {x.size(3) * 2, x.size(4) * 2}, 2.0, 2.0);
    return at::conv2d(h, w.at(p + ".resample.1.weight"), w.at(p + ".resample.1.bias"), {1, 1}, {1, 1}).unsqueeze(2);
}

// z: the KSampler latent [1, 16, 1, h, w] (already un-normalized) -> image [H, W, 3] FP32 in [0, 1]
Tensor decode(const Weights& w, const Tensor& z) {
    TORCH_CHECK(z.dim() == 5 && z.size(0) == 1 && z.size(2) == 1, "dsh_anima_gen.vae_decode: z must be [1, 16, 1, h, w]");
    auto x = causal_conv(w, "conv2", z.to(w.at("conv2.weight").scalar_type()));
    x = causal_conv(w, "decoder.conv1", x);
    x = residual(w, "decoder.middle.0", x);
    x = attention(w, "decoder.middle.1", x);
    x = residual(w, "decoder.middle.2", x);
    for (int i = 0;; ++i) {
        const auto p = "decoder.upsamples." + std::to_string(i);
        if (has(w, p + ".residual.0.gamma")) x = residual(w, p, x);
        else if (has(w, p + ".resample.1.weight")) x = upsample(w, p, x);
        else break;
    }
    x = causal_conv(w, "decoder.head.2", at::silu(rms(w, "decoder.head.0", x)));
    auto image = x.to(at::kFloat).add_(1.0).div_(2.0).clamp_(0.0, 1.0);  // process_output in the output dtype
    return image.squeeze(0).squeeze(1).permute({1, 2, 0}).contiguous();
}

// Encode, one frame: ComfyUI's WanVAE.encode with t = 1 runs the encoder without a feature cache, so every causal
// conv keeps its last temporal slice and the downsample3d time convs are skipped (as the decoder's upsample3d).
Tensor downsample(const Weights& w, const std::string& p, const Tensor& x) {
    auto h = at::constant_pad_nd(x.squeeze(2), {0, 1, 0, 1}, 0.0);  // ZeroPad2d((0, 1, 0, 1))
    return at::conv2d(h, w.at(p + ".resample.1.weight"), w.at(p + ".resample.1.bias"), {2, 2}).unsqueeze(2);
}

// image: [H, W, 3] FP32 in [0, 1] (H, W multiples of 8) -> mu [1, 16, 1, H/8, W/8] FP32 (not yet normalized)
Tensor encode(const Weights& w, const Tensor& image) {
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "dsh_anima_gen.vae_encode: image must be [H, W, 3]");
    const auto dtype = w.at("encoder.conv1.weight").scalar_type();
    auto x = image.permute({2, 0, 1}).unsqueeze(0).unsqueeze(2).mul(2.0).sub(1.0).to(dtype);  // process_input
    x = causal_conv(w, "encoder.conv1", x);
    for (int i = 0;; ++i) {
        const auto p = "encoder.downsamples." + std::to_string(i);
        if (has(w, p + ".residual.0.gamma")) x = residual(w, p, x);
        else if (has(w, p + ".resample.1.weight")) x = downsample(w, p, x);
        else break;
    }
    x = residual(w, "encoder.middle.0", x);
    x = attention(w, "encoder.middle.1", x);
    x = residual(w, "encoder.middle.2", x);
    x = causal_conv(w, "encoder.head.2", at::silu(rms(w, "encoder.head.0", x)));
    x = causal_conv(w, "conv1", x);
    return x.narrow(1, 0, x.size(1) / 2).to(at::kFloat);  // mu
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_anima_gen, m) {
    m.def("vae_decode(Dict(str, Tensor) weights, Tensor z) -> Tensor", &decode);
    m.def("vae_encode(Dict(str, Tensor) weights, Tensor image) -> Tensor", &encode);
}
