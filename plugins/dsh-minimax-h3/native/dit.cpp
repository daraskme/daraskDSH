// MiniMax H3 audio-video DiT forward for text-to-video (batch 1, no keyframes/references/masks).
// Written from ComfyUI's comfy/ldm/minimax/model.py (GPL-3.0); every op keeps ComfyUI's dtype and
// order so the BF16 results can be compared bit for bit. Attention is dense SDPA here; the SLA
// sparse path replaces it in a later step.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace dsh_kitchen {  // kitchen/kitchen_ops.cpp (vendored comfy-kitchen kernel, bit-identical to ComfyUI)
void rms_rope_split_half_(at::Tensor q, at::Tensor k, const at::Tensor& freqs, const at::Tensor& q_scale,
                          const at::Tensor& k_scale, double epsilon, int64_t rot_dim);
}

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

constexpr int64_t kHidden = 5376, kHeads = 56, kHeadDim = 128, kRotDim = 96, kVideoC = 24, kAudioC = 32;
constexpr double kEps = 1e-5;
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr int kFramePerToken[5] = {1, 4, 4, 4, 4};
constexpr int kTagVideo = 0, kTagText = 1, kTagAudio = 2;

struct Segment {
    int64_t start, stop, row;  // row: mod-row index (timestep index * 3 + modality tag) or final-layer row
};

Tensor f32(const Weights& w, const std::string& name) { return w.at(name).to(at::kFloat); }

// ---- packed layout (PackedLayout, T2V) --------------------------------------------------------

Tensor axis_from_sqrt_area(int64_t dim, double sqrt_area) {
    const double ratio = double(dim) / sqrt_area;
    const int64_t n = dim / 2;
    return (at::arange(n, at::kDouble) * (ratio / double(n)) + (1.0 - ratio) / 2.0) * 32.0;
}

struct Layout {
    int64_t text, audio, video, seq;
    Tensor positions;  // [S, 3] float64 (t, h, w), CPU
};

Layout make_layout(int64_t text_len, int64_t latent_t, int64_t lat_h, int64_t lat_w, int64_t audio_t) {
    const double area = std::sqrt(double(lat_h * lat_w));
    auto hs = axis_from_sqrt_area(lat_h, area), ws = axis_from_sqrt_area(lat_w, area);
    auto mesh = at::meshgrid({hs, ws}, "ij");
    auto frame = at::stack({mesh[0].reshape(-1), mesh[1].reshape(-1)}, -1);  // [rows, 2]
    const int64_t frame_rows = frame.size(0);

    auto text = at::zeros({text_len, 3}, at::kDouble);
    text.select(1, 0).copy_(at::arange(text_len, at::kDouble));
    const double cursor = double(text_len);

    auto audio = at::zeros({audio_t * 2, 3}, at::kDouble);
    audio.select(1, 0).copy_((cursor + at::arange(audio_t, at::kDouble)).repeat({2}));
    audio.narrow(0, 0, audio_t).select(1, 2).fill_(ws[0].item<double>());
    audio.narrow(0, audio_t, audio_t).select(1, 2).fill_(ws[ws.size(0) - 1].item<double>());

    std::vector<double> spans(latent_t);
    for (int64_t k = 0; k < latent_t; ++k) spans[k] = kFrameRescale * kFramePerToken[k % 5];
    auto span = at::tensor(spans, at::kDouble);
    auto t_grid = cursor + at::cat({at::zeros({1}, at::kDouble), span.narrow(0, 0, latent_t - 1).cumsum(0)});
    auto video = at::empty({latent_t, frame_rows, 3}, at::kDouble);
    video.select(2, 0).copy_(t_grid.unsqueeze(1).expand({latent_t, frame_rows}));
    video.narrow(2, 1, 2).copy_(frame.unsqueeze(0).expand({latent_t, frame_rows, 2}));

    const int64_t n_video = latent_t * frame_rows;
    return {text_len, audio_t * 2, n_video, text_len + audio_t * 2 + n_video,
            at::cat({text, audio, video.reshape({-1, 3})})};
}

Tensor rope_table(const Layout& layout, const Tensor& inv_freq, at::ScalarType dtype) {
    // rope_freqs + rope_rotation_table: [1, S, 1, 48, 2, 2]
    auto pos = layout.positions.to(at::kFloat).to(inv_freq.device());
    auto per_axis = pos.unsqueeze(-1) * inv_freq.view({1, 1, -1});
    auto axes = per_axis.unbind(1);
    auto half = at::cat({axes[0], axes[1], axes[2]}, -1);   // [S, 48]
    auto c = at::cos(half), s = at::sin(half);
    return at::stack({c, -s, s, c}, -1).reshape({1, layout.seq, 1, half.size(1), 2, 2}).to(dtype);
}

// ---- timesteps ----------------------------------------------------------------------------------

Tensor time_shift_sigma(const Tensor& sigma, double from, double to) {
    auto base = sigma / (from + sigma * (1.0 - from));
    return to * base / (1.0 + (to - 1.0) * base);
}

Tensor time_embedding(const Weights& w, const std::vector<double>& unique_t, const at::Device& device) {
    auto table = f32(w, "adaln_t_table").to(device);
    auto t = at::tensor(std::vector<float>(unique_t.begin(), unique_t.end()), at::TensorOptions().dtype(at::kFloat).device(device));
    auto pos = t.clamp(0.0, 1.0) * double(table.size(0) - 1);
    auto i0 = pos.floor().to(at::kLong).clamp_max(table.size(0) - 2);
    return at::lerp(table.index_select(0, i0), table.index_select(0, i0 + 1), (pos - i0).unsqueeze(1));
}

std::vector<Tensor> adaln(const Weights& w, const std::string& prefix, const Tensor& t_emb, int64_t expand, int64_t modalities) {
    auto x = at::linear(t_emb, f32(w, prefix + ".linear.weight"), f32(w, prefix + ".linear.bias"));
    return x.view({x.size(0) * modalities, expand * kHidden}).chunk(expand, -1);
}

// ---- block --------------------------------------------------------------------------------------

void mod_scale_shift(Tensor& h, const Tensor& shift, const Tensor& scale, const std::vector<Segment>& segments) {
    for (const auto& s : segments) {
        h.narrow(0, s.start, s.stop - s.start).mul_(1.0 + scale[s.row].to(h.scalar_type())).add_(shift[s.row].to(h.scalar_type()));
    }
}

void mod_gate(Tensor& x, const Tensor& gate, const Tensor& other, const std::vector<Segment>& segments) {
    for (const auto& s : segments) {
        x.narrow(0, s.start, s.stop - s.start).addcmul_(other.narrow(0, s.start, s.stop - s.start), gate[s.row].to(x.scalar_type()));
    }
}

Tensor attention(const Weights& w, const std::string& p, const Tensor& h, const Tensor& table) {
    const auto s = h.size(0);
    auto qkv = at::linear(h, w.at(p + ".attn.qkv_proj.weight")).split(kHeads * kHeadDim, -1);
    // fused per-head RMSNorm + partial split-half RoPE, in place on the qkv buffer (as ComfyUI)
    auto q = qkv[0].view({1, s, kHeads, kHeadDim}), k = qkv[1].view({1, s, kHeads, kHeadDim});
    dsh_kitchen::rms_rope_split_half_(q, k, table, w.at(p + ".attn.q_norm.weight"), w.at(p + ".attn.k_norm.weight"), kEps,
                                      kRotDim);
    q = q[0];
    k = k[0];
    auto v = qkv[2].view({s, kHeads, kHeadDim});
    auto out = at::scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                v.transpose(0, 1).unsqueeze(0));
    return at::linear(out.transpose(1, 2).reshape({s, kHeads * kHeadDim}), w.at(p + ".attn.out_proj.weight"));
}

Tensor mlp(const Weights& w, const std::string& p, const Tensor& h) {
    auto gate_up = at::linear(h, w.at(p + ".mlp.fc1.weight")).chunk(2, -1);
    return at::linear(at::silu(gate_up[0]) * gate_up[1], w.at(p + ".mlp.fc2.weight"));
}

Tensor block(const Weights& w, int64_t index, Tensor x, const Tensor& t_emb, const std::vector<Segment>& segments,
             const Tensor& table) {
    const auto p = "blocks." + std::to_string(index);
    auto mod = adaln(w, p + ".adaln_proj", t_emb, 6, 3);  // shift/scale/gate msa, shift/scale/gate mlp
    auto h = at::rms_norm(x, {kHidden}, w.at(p + ".norm1.weight"), kEps);
    mod_scale_shift(h, mod[0], mod[1], segments);
    mod_gate(x, mod[2], attention(w, p, h, table), segments);
    h = at::rms_norm(x, {kHidden}, w.at(p + ".norm2.weight"), kEps);
    mod_scale_shift(h, mod[3], mod[4], segments);
    mod_gate(x, mod[5], mlp(w, p, h), segments);
    return x;
}

std::vector<Segment> segments_from(const std::vector<int64_t>& flat) {
    TORCH_CHECK(flat.size() % 3 == 0, "H3: segments must be (start, stop, row) triples");
    std::vector<Segment> segments;
    for (size_t i = 0; i < flat.size(); i += 3) segments.push_back({flat[i], flat[i + 1], flat[i + 2]});
    return segments;
}

// ---- forward ------------------------------------------------------------------------------------

std::vector<Tensor> forward_impl(const Weights& w, const Tensor& video_x, const Tensor& audio_x, const Tensor& timestep,
                                 const Tensor& context, const Tensor& token_tags, double shift_v, double shift_a) {
    TORCH_CHECK(video_x.dim() == 5 && video_x.size(0) == 1, "H3: video latent must be [1, 24, T, H, W]");
    const auto device = video_x.device();
    const auto dtype = context.scalar_type();
    const int64_t latent_t = video_x.size(2), lat_h = video_x.size(3), lat_w = video_x.size(4);
    const int64_t audio_t = audio_x.size(-1), text_len = context.size(1);
    TORCH_CHECK(lat_h % 2 == 0 && lat_w % 2 == 0, "H3: latent height/width must be even");
    const auto layout = make_layout(text_len, latent_t, lat_h, lat_w, audio_t);

    auto sigma_v = (timestep.flatten()[0] / 1000.0).to(at::kFloat).clamp_min(1e-6);
    const double t_v = (1.0 - sigma_v).item<float>();
    const double t_a = (1.0 - time_shift_sigma(sigma_v, shift_v, shift_a)).item<float>();
    std::set<double> unique_set{t_v, t_a};
    std::vector<double> unique_t(unique_set.begin(), unique_set.end());
    auto row_of = [&](double t) { return int64_t(std::find(unique_t.begin(), unique_t.end(), t) - unique_t.begin()); };

    // mod segments: text tag runs (all 1 for T2V), audio (tag 2), video (tag 0)
    std::vector<Segment> segments;
    auto tags = token_tags.to(at::kCPU).to(at::kLong);
    const auto* tag = tags.const_data_ptr<int64_t>();
    int64_t run = 0;
    for (int64_t i = 1; i <= text_len; ++i) {
        if (i == text_len || tag[i] != tag[run]) {
            segments.push_back({run, i, row_of(t_v) * 3 + tag[run]});
            run = i;
        }
    }
    const int64_t a0 = text_len, v0 = text_len + layout.audio;
    segments.push_back({a0, v0, row_of(t_a) * 3 + kTagAudio});
    segments.push_back({v0, layout.seq, row_of(t_v) * 3 + kTagVideo});

    // embed and pack [text | audio | video]
    auto video_rows = video_x.to(at::kFloat).reshape({1, kVideoC, latent_t, 1, lat_h / 2, 2, lat_w / 2, 2})
                          .permute({0, 2, 4, 6, 1, 3, 5, 7}).reshape({-1, kVideoC * 4});
    auto video_embed = at::linear(video_rows, f32(w, "video_patch_proj.weight"), f32(w, "video_patch_proj.bias")).to(dtype);
    auto audio_rows = audio_x.to(at::kFloat)[0].permute({1, 2, 0}).reshape({-1, kAudioC});
    auto audio_embed = at::linear(audio_rows, f32(w, "audio_patch_proj.weight"), f32(w, "audio_patch_proj.bias")).to(dtype);
    auto h = at::cat({context[0], audio_embed, video_embed});

    auto t_emb = time_embedding(w, unique_t, device);
    auto table = rope_table(layout, f32(w, "rope.inv_freq").to(device), dtype);
    for (int64_t i = 0; w.contains("blocks." + std::to_string(i) + ".norm1.weight"); ++i) {
        h = block(w, i, h, t_emb, segments, table);
    }

    // final layer: rows index t_emb directly (one modality)
    auto mod = adaln(w, "final_layer.adaln_proj", t_emb, 2, 1);  // shift, scale
    auto head = [&](int64_t a, int64_t b, double t, const std::string& name) {
        auto normed = at::rms_norm(h.narrow(0, a, b - a), {kHidden}, w.at("final_layer.norm.weight"), kEps);
        auto x = (normed * (1.0 + mod[1][row_of(t)]) + mod[0][row_of(t)]).to(at::kFloat);
        return at::linear(x, f32(w, name + ".weight"), f32(w, name + ".bias"));
    };
    auto v = head(v0, layout.seq, t_v, "final_layer.video_out");
    auto a = head(a0, v0, t_a, "final_layer.audio_out");
    auto video_out = v.reshape({1, latent_t, lat_h / 2, lat_w / 2, kVideoC, 1, 2, 2})
                         .permute({0, 4, 1, 5, 2, 6, 3, 7}).reshape({1, kVideoC, latent_t, lat_h, lat_w});
    auto audio_out = a.reshape({2, audio_t, kAudioC}).permute({2, 0, 1}).unsqueeze(0);
    return {(-video_out).to(video_x.scalar_type()), (-audio_out).to(audio_x.scalar_type())};
}

// MiniMaxH3Model.forward: the sampler carries audio as (sigma_v / sigma_a) * x_audio.
std::vector<Tensor> forward(const Weights& w, const Tensor& video_x, const Tensor& audio_x, const Tensor& timestep,
                            const Tensor& context, const Tensor& token_tags, double shift_v, double shift_a,
                            double audio_scale) {
    auto audio_src = audio_x;
    Tensor carry, sigma_a;
    auto x_audio = audio_x;
    if (audio_scale != 1.0) {
        auto sigma_v = (timestep.flatten()[0] / 1000.0).to(at::kFloat).clamp_min(1e-6);
        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a);
        carry = (sigma_a / sigma_v).to(audio_src.scalar_type());
        x_audio = audio_src * carry;
    }
    auto out = forward_impl(w, video_x, x_audio, timestep, context, token_tags, shift_v, shift_a);
    if (audio_scale != 1.0) {
        out[1] = (1.0 - audio_scale) * (audio_src * carry) + (1.0 + (audio_scale - 1.0) * sigma_a).to(out[1].scalar_type()) * out[1];
    }
    return out;
}

Tensor block_op(const Weights& w, int64_t index, const Tensor& x, const Tensor& t_emb, std::vector<int64_t> segments,
                const Tensor& table) {
    return block(w, index, x.clone(), t_emb, segments_from(segments), table);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(dsh_h3, m) {
    m.def("dit_forward(Dict(str, Tensor) weights, Tensor video_x, Tensor audio_x, Tensor timestep, Tensor context, "
          "Tensor token_tags, float shift_video, float shift_audio, float audio_scale) -> Tensor[]", &forward);
    m.def("dit_block(Dict(str, Tensor) weights, int index, Tensor x, Tensor t_emb, int[] segments, Tensor rope_table) -> Tensor",
          &block_op);
}
