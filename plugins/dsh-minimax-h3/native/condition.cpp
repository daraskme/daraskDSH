// MiniMax H3 text conditioning after the Qwen encoder: condition_proj and the 2-block token
// refiner of the DiT checkpoint, with the LoRA merged into the weights. Written from ComfyUI's
// comfy/ldm/minimax/model.py and comfy/weight_adapter/lora.py (GPL-3.0).
#include <ATen/ATen.h>
#include <torch/library.h>

#include <string>

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

constexpr int64_t kHidden = 5376, kHeads = 56, kHeadDim = 128;
constexpr double kEps = 1e-5;

// ComfyUI (dynamic VRAM) patches a LoRA in the weight's own dtype:
// w + (strength * alpha / rank) * (up @ down), every step in BF16. Verified bit-exact.
Tensor merge_lora(const Tensor& weight, const Tensor& up, const Tensor& down, const Tensor& alpha, double strength) {
    const auto dtype = weight.scalar_type();
    auto diff = at::mm(up.to(dtype), down.to(dtype));
    auto scale = alpha.to(dtype) / down.size(0) * strength;
    return weight + scale * diff;
}

// Merge every LoRA into the matching "<path>.weight", in place (the base weights are not kept: the DiT alone is
// 38 GB). Keys: kohya (lora_unet_<path with _>.lora_up / lora_down / alpha) or diffusion-pipe / PEFT
// (diffusion_model.<path>.lora_B / lora_A, alpha = rank). int8 weights are skipped (loras.py attaches those as
// run-time LoRA). Returns the merged count.
int64_t merge_loras(const Weights& weights, const Weights& lora, double strength) {
    int64_t count = 0;
    for (const auto& item : weights) {
        const auto& name = item.key();
        if (name.size() <= 7 || name.substr(name.size() - 7) != ".weight") continue;
        auto value = item.value();
        if (value.scalar_type() == at::kChar) continue;
        const auto path = name.substr(0, name.size() - 7);
        auto key = "lora_unet_" + path;
        for (auto& c : key) if (c == '.') c = '_';
        if (lora.contains(key + ".lora_up.weight")) {
            value.copy_(merge_lora(value, lora.at(key + ".lora_up.weight"), lora.at(key + ".lora_down.weight"),
                                   lora.at(key + ".alpha"), strength));
        } else if (lora.contains("diffusion_model." + path + ".lora_B.weight")) {
            const auto& down = lora.at("diffusion_model." + path + ".lora_A.weight");
            value.copy_(merge_lora(value, lora.at("diffusion_model." + path + ".lora_B.weight"), down,
                                   at::scalar_tensor(double(down.size(0)), down.options()), strength));
        } else {
            continue;
        }
        ++count;
    }
    return count;
}

Tensor refine(const Weights& w, const Tensor& text_states) {
    // text_states: Qwen layer-50 hidden [L, 5120] (FP32). ComfyUI feeds condition_proj in BF16.
    const auto& proj = w.at("condition_proj.weight");
    auto x = at::linear(text_states.to(proj.scalar_type()), proj, w.at("condition_proj.bias"));
    const auto l = x.size(0);
    for (int i = 0; i < 2; ++i) {
        const auto p = "token_refiner.blocks." + std::to_string(i);
        auto h = at::rms_norm(x, {kHidden}, w.at(p + ".norm1.weight"), kEps);
        auto qkv = at::linear(h, w.at(p + ".attn.qkv_proj.weight")).split(kHeads * kHeadDim, -1);
        auto q = at::rms_norm(qkv[0].view({l, kHeads, kHeadDim}), {kHeadDim}, w.at(p + ".attn.q_norm.weight"), kEps);
        auto k = at::rms_norm(qkv[1].view({l, kHeads, kHeadDim}), {kHeadDim}, w.at(p + ".attn.k_norm.weight"), kEps);
        auto v = qkv[2].view({l, kHeads, kHeadDim});
        auto a = at::scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                  v.transpose(0, 1).unsqueeze(0));
        x = at::linear(a.squeeze(0).transpose(0, 1).reshape({l, kHeads * kHeadDim}), w.at(p + ".attn.out_proj.weight")) + x;
        h = at::rms_norm(x, {kHidden}, w.at(p + ".norm2.weight"), kEps);
        auto gate_up = at::linear(h, w.at(p + ".mlp.fc1.weight")).chunk(2, -1);
        x = at::linear(at::silu(gate_up[0]) * gate_up[1], w.at(p + ".mlp.fc2.weight")) + x;
    }
    return at::rms_norm(x, {kHidden}, w.at("token_refiner.final_norm.weight"), kEps);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_h3, m) {
    m.def("merge_loras(Dict(str, Tensor) weights, Dict(str, Tensor) lora, float strength) -> int",
          &merge_loras);
    m.def("refine_text(Dict(str, Tensor) weights, Tensor text_states) -> Tensor", &refine);
}
