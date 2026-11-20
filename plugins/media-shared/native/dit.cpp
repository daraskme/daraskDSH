// MiniMax H3 audio-video DiT forward for text-to-video (batch 1, no keyframes/references/masks).
// Written from ComfyUI's comfy/ldm/minimax/model.py (GPL-3.0); every op keeps ComfyUI's dtype and
// order so the BF16 results can be compared bit for bit. Attention is dense SDPA, or the SLA block-sparse
// path (comfy_extras/nodes_sparse_attention.py "sla", keep 10%, exact_kv_and_rows; comfy-kitchen
// sol_attn_chunked) when per-block statistics are passed in.
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace dsh_kitchen {  // kitchen/kitchen_ops.cpp (vendored comfy-kitchen kernel, bit-identical to ComfyUI)
void rms_rope_split_half_(at::Tensor q, at::Tensor k, const at::Tensor& freqs, const at::Tensor& q_scale,
                          const at::Tensor& k_scale, double epsilon, int64_t rot_dim);
std::vector<int64_t> plan(int64_t batch, int64_t seq_len, int64_t num_heads, int64_t token_aug);
at::Tensor int8_linear(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& weight_scale,
                       const std::optional<at::Tensor>& bias, int64_t act);
void producer_begin(at::Tensor workspace, int64_t batch, int64_t seq_len, int64_t num_heads, int64_t token_aug);
void producer_chunk(at::Tensor workspace, const at::Tensor& qkv, const at::Tensor& fab, const at::Tensor& qw,
                    const at::Tensor& kw, const at::Tensor& kmean, const at::Tensor& vscale,
                    const std::optional<at::Tensor>& block_len, double rope_eps, int64_t rot_dim, int64_t t0,
                    int64_t m, int64_t batch, int64_t seq_len, int64_t num_heads, int64_t token_aug);
void attn_core(at::Tensor workspace, at::Tensor out, const at::Tensor& vscale, at::Tensor kmean_next,
               at::Tensor vamax, const std::optional<at::Tensor>& block_len, bool tail, int64_t batch,
               int64_t seq_len, int64_t num_heads, double tau, double scale,
               const std::optional<at::Tensor>& threshold, int64_t sink_start, int64_t sink_end,
               int64_t sink_q_start, int64_t sink_q_end, int64_t token_aug);
}

namespace {

using at::Tensor;
using Weights = c10::Dict<std::string, Tensor>;

constexpr int64_t kHidden = 5376, kHeads = 56, kHeadDim = 128, kRotDim = 96, kVideoC = 24, kAudioC = 32;
constexpr double kEps = 1e-5;
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr double kVisualCondT = 0.999;  // VISUAL_COND_TIMESTEP: condition rows (keyframes, references)
constexpr double kAudioCondT = 1.0;     // AUDIO_COND_TIMESTEP: audio keyframe rows
constexpr int kFramePerToken[5] = {1, 4, 4, 4, 4};
constexpr int kTagVideo = 0, kTagText = 1, kTagAudio = 2;

struct Segment {
    int64_t start, stop, row;  // row: mod-row index (timestep index * 3 + modality tag) or final-layer row
};

Tensor f32(const Weights& w, const std::string& name) { return w.at(name).to(at::kFloat); }

// A block projection: BF16 (H3) or int8 convrot W8A8 (P-H3, comfy-kitchen's int8_linear); swiglu feeds fc2.
// A run-time LoRA (merge_loras' "<name>.lora_a" / "lora_b", int8 weights only) adds (z @ a^T) @ b^T, z the GEMM input.
Tensor proj(const Weights& w, const std::string& name, const Tensor& x, bool swiglu = false) {
    if (w.contains(name + ".weight_scale")) {
        auto y = dsh_kitchen::int8_linear(x, w.at(name + ".weight"), w.at(name + ".weight_scale"), std::nullopt, swiglu ? 2 : 0);
        if (w.contains(name + ".lora_a")) {
            auto z = x;
            if (swiglu) {
                auto gate_up = x.chunk(2, -1);
                z = at::silu(gate_up[0]) * gate_up[1];
            }
            y.addmm_(at::mm(z, w.at(name + ".lora_a").t()), w.at(name + ".lora_b").t());
        }
        return y;
    }
    if (!swiglu) return at::linear(x, w.at(name + ".weight"));
    auto gate_up = x.chunk(2, -1);
    return at::linear(at::silu(gate_up[0]) * gate_up[1], w.at(name + ".weight"));
}

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
    auto qkv = proj(w, p + ".attn.qkv_proj", h).split(kHeads * kHeadDim, -1);
    // fused per-head RMSNorm + partial split-half RoPE, in place on the qkv buffer (as ComfyUI)
    auto q = qkv[0].view({1, s, kHeads, kHeadDim}), k = qkv[1].view({1, s, kHeads, kHeadDim});
    dsh_kitchen::rms_rope_split_half_(q, k, table, w.at(p + ".attn.q_norm.weight"), w.at(p + ".attn.k_norm.weight"), kEps,
                                      kRotDim);
    q = q[0];
    k = k[0];
    auto v = qkv[2].view({s, kHeads, kHeadDim});
    auto out = at::scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                v.transpose(0, 1).unsqueeze(0));
    return proj(w, p + ".attn.out_proj", out.transpose(1, 2).reshape({s, kHeads * kHeadDim}));
}


// ---- SLA (sol_attn_chunked with topk_ratio 0.1, token_aug 0, B = 1) ------------------------------

constexpr int64_t kBlock = 64, kChunk = 4096;
constexpr double kKeep = 0.1, kVscaleMargin = 1.1, kLog2e = 1.4426950408889634;
// sol_attn_plan's offsets, in its key order
enum PlanKey { Tp, NTB, NPAD, NQ, qiP, qs, kiP, ksb, vTi, vsc, kciP, kcs, vcT, thr, cen8, cens, idx, cnt, oPart, mPart,
               lPart, statsV, qmean };

struct Sparse {
    Tensor fab;                    // [S, 96, 2] FP32 rope coefficients (_packed_rope_fab)
    int64_t sink0, sink1, sink_q0, sink_q1;
    std::vector<Tensor> stats;     // per block: kmean, vscale ([56, 128] FP32), updated in place across steps
    bool first;                    // the first step: bootstrap the statistics
};

Tensor packed_rope_fab(const Tensor& table) {
    auto f = table.reshape({-1, kRotDim / 2, 2, 2});
    auto fab = at::empty({f.size(0), kRotDim, 2}, f.options().dtype(at::kFloat));
    const int64_t half = kRotDim / 2;
    fab.narrow(1, 0, half).select(2, 0).copy_(f.select(2, 0).select(2, 0));
    fab.narrow(1, 0, half).select(2, 1).copy_(f.select(2, 0).select(2, 1));
    fab.narrow(1, half, half).select(2, 0).copy_(f.select(2, 1).select(2, 1));
    fab.narrow(1, half, half).select(2, 1).copy_(f.select(2, 1).select(2, 0));
    return fab;
}

Tensor ws_view(const Tensor& ws, int64_t offset, int64_t bytes, at::ScalarType dtype) {
    return ws.narrow(0, offset, bytes).view(dtype);
}

Tensor vscale_of(const Tensor& vamax) { return (vamax / 127.0 * kVscaleMargin).clamp_min(1e-8); }

Tensor block_lengths(int64_t t, int64_t n, const at::Device& device) {
    auto lengths = at::full({n}, double(kBlock), at::TensorOptions().dtype(at::kFloat).device(device));
    const int64_t tail = t - (n - 1) * kBlock;
    if (tail < kBlock) lengths[n - 1] = double(tail);
    return lengths;
}

// _topk_threshold_from_workspace + _topk_from_pooled (H = 56 heads fit one slice at these lengths)
Tensor topk_threshold(const Tensor& ws, const std::vector<int64_t>& p, const Tensor& lengths, int64_t s0, int64_t s1) {
    const int64_t ntb = p[NTB], npad = p[NPAD], d = kHeadDim, h = kHeads;
    auto c8 = ws_view(ws, p[cen8], h * ntb * d, at::kChar).view({h, ntb, d}).to(at::kFloat);
    auto csc = ws_view(ws, p[cens], h * ntb * 4, at::kFloat).view({h, ntb});
    auto ksums = ws_view(ws, p.back() /* scratch */, h * npad * d * 4, at::kFloat).view({h, npad, d}).narrow(1, 0, ntb);
    auto kc = ksums / lengths.view({1, -1, 1});
    kc = kc - kc.mean(1, true);
    // cen8 is stored perm_d'd: gather at perm_d to recover channel order
    auto dd = at::arange(128, at::kLong);
    auto kcq = dd.__rshift__(5), rem = dd.bitwise_and(31);
    auto hh = rem.__rshift__(4), r2 = rem.bitwise_and(15);
    auto perm = (kcq * 32 + 8 * r2.__rshift__(2) + 4 * hh + r2.bitwise_and(3)).to(ws.device());
    c8 = c8.index_select(2, perm);
    const int64_t n = ntb;
    auto ksc = (kc.abs().amax(-1, true) / 127.0).clamp_min(1e-12);
    auto k8 = at::round(kc / ksc).clamp_(-127, 127);
    const int64_t sinks = std::max<int64_t>(0, std::min(s1, n) - std::min(s0, n));
    const int64_t kk = std::max<int64_t>(0, std::min<int64_t>(n - sinks - 1, std::max<int64_t>(1, (int64_t)std::nearbyint(kKeep * (n - sinks)))));
    if (kk == 0) return at::full({h, n}, INFINITY, c8.options());
    auto s = at::bmm(c8, k8.transpose(1, 2));
    s.mul_((csc * (std::pow(double(d), -0.5) * kLog2e)).unsqueeze(-1)).mul_(ksc.squeeze(-1).unsqueeze(-2));
    if (s1 > s0) s.narrow(-1, s0, s1 - s0).fill_(-INFINITY);
    auto kth = std::get<0>(s.topk(kk, -1, true, false)).amin(-1);
    return (kth - kth.abs() * 1e-5).contiguous();
}

Tensor sparse_attention(const Weights& w, const std::string& pre, int64_t index, const Tensor& h, const Sparse& sp) {
    const int64_t t = h.size(0);
    const auto& qkv_w = w.at(pre + ".attn.qkv_proj.weight");
    auto qw = w.at(pre + ".attn.q_norm.weight").to(at::kBFloat16).contiguous();
    auto kw = w.at(pre + ".attn.k_norm.weight").to(at::kBFloat16).contiguous();
    auto p = dsh_kitchen::plan(1, t, kHeads, 0);
    auto ws = at::empty({p.back() /* total */}, h.options().dtype(at::kByte));
    p.pop_back();  // scratch is now last
    auto lengths = block_lengths(t, p[NTB], h.device());
    auto produce = [&](const Tensor& km, const Tensor& vs) {
        dsh_kitchen::producer_begin(ws, 1, t, kHeads, 0);
        for (int64_t i = 0; i < t; i += kChunk) {
            const int64_t m = std::min(kChunk, t - i);
            auto qkv = at::linear(h.narrow(0, i, m), qkv_w).contiguous();
            dsh_kitchen::producer_chunk(ws, qkv, sp.fab, qw, kw, km, vs, std::nullopt, kEps, kRotDim, i, m, 1, t, kHeads, 0);
        }
    };
    Tensor kmean = sp.stats.at(2 * index), vscale = sp.stats.at(2 * index + 1);
    const auto f32 = h.options().dtype(at::kFloat);
    if (sp.first) {
        produce(at::zeros({kHeads, kHeadDim}, f32), at::ones({kHeads, kHeadDim}, f32));
        auto ksums = ws_view(ws, p.back(), kHeads * p[NPAD] * kHeadDim * 4, at::kFloat).view({kHeads, p[NPAD], kHeadDim})
                         .narrow(1, 0, p[NTB]);
        kmean = ksums.sum(1) / lengths.sum();
        vscale = vscale_of(ws_view(ws, p[statsV], kHeads * kHeadDim * 4, at::kFloat).view({kHeads, kHeadDim}));
    }
    kmean = kmean.contiguous();
    vscale = vscale.clamp_min(1e-8).contiguous();
    produce(kmean, vscale);
    auto out = at::empty({1, t, kHeads, kHeadDim}, h.options().dtype(at::kBFloat16));
    auto kmean_next = at::empty({kHeads, kHeadDim}, f32);
    auto vamax = at::empty({kHeads, kHeadDim}, f32);
    auto threshold = topk_threshold(ws, p, lengths, sp.sink0, sp.sink1);
    dsh_kitchen::attn_core(ws, out, vscale, kmean_next, vamax, std::nullopt, true, 1, t, kHeads, 1.3,
                           std::pow(double(kHeadDim), -0.5), threshold, sp.sink0, sp.sink1, sp.sink_q0, sp.sink_q1, 0);
    sp.stats[2 * index].copy_(kmean_next);
    sp.stats[2 * index + 1].copy_(vscale_of(vamax));
    return at::linear(out.view({t, kHeads * kHeadDim}), w.at(pre + ".attn.out_proj.weight"));
}

Tensor mlp(const Weights& w, const std::string& p, const Tensor& h) {
    return proj(w, p + ".mlp.fc2", proj(w, p + ".mlp.fc1", h), true);
}

Tensor block(const Weights& w, int64_t index, Tensor x, const Tensor& t_emb, const std::vector<Segment>& segments,
             const Tensor& table, const Sparse* sparse = nullptr) {
    const auto p = "blocks." + std::to_string(index);
    auto mod = adaln(w, p + ".adaln_proj", t_emb, 6, 3);  // shift/scale/gate msa, shift/scale/gate mlp
    auto h = at::rms_norm(x, {kHidden}, w.at(p + ".norm1.weight"), kEps);
    mod_scale_shift(h, mod[0], mod[1], segments);
    mod_gate(x, mod[2], sparse ? sparse_attention(w, p, index, h, *sparse) : attention(w, p, h, table), segments);
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
                                 const Tensor& context, const Tensor& token_tags, double shift_v, double shift_a,
                                 const std::vector<Tensor>& stats, bool first, const std::optional<Tensor>& cond_rows,
                                 const std::optional<Tensor>& positions, const std::optional<Tensor>& cond_audio_rows,
                                 int64_t cond_audio_at) {
    TORCH_CHECK(video_x.dim() == 5 && video_x.size(0) == 1, "H3: video latent must be [1, 24, T, H, W]");
    const auto device = video_x.device();
    const auto dtype = context.scalar_type();
    const int64_t latent_t = video_x.size(2), lat_h = video_x.size(3), lat_w = video_x.size(4);
    const int64_t audio_t = audio_x.size(-1), text_len = context.size(1);
    TORCH_CHECK(lat_h % 2 == 0 && lat_w % 2 == 0, "H3: latent height/width must be even");
    auto layout = make_layout(text_len, latent_t, lat_h, lat_w, audio_t);
    // keyframe / reference rows (PackedLayout's cond and ref_img segments) pack between the text and the targets;
    // their positions come from the caller (sampler.packed_positions)
    // audio keyframe rows (cond_audio) sit after the first cond_audio_at condition rows (the video keyframes; references
    // follow them), as PackedLayout orders keyframes before references
    const int64_t n_cond = cond_rows ? cond_rows->size(0) : 0;
    const int64_t n_cond_a = cond_audio_rows ? cond_audio_rows->size(0) : 0;
    const int64_t k_cond = cond_audio_at < 0 ? n_cond : cond_audio_at;
    TORCH_CHECK(k_cond <= n_cond, "H3: cond_audio_at is past the condition rows");
    if (positions) {
        TORCH_CHECK(positions->size(0) == layout.seq + n_cond + n_cond_a, "H3: positions do not match the packed sequence");
        layout.positions = positions->to(at::kCPU).to(at::kDouble);
        layout.seq += n_cond + n_cond_a;
    } else {
        TORCH_CHECK(n_cond + n_cond_a == 0, "H3: condition rows need their positions");
    }

    auto sigma_v = (timestep.flatten()[0] / 1000.0).to(at::kFloat).clamp_min(1e-6);
    const double t_v = (1.0 - sigma_v).item<float>();
    const double t_a = (1.0 - time_shift_sigma(sigma_v, shift_v, shift_a)).item<float>();
    const double t_c = std::max(t_v, kVisualCondT);  // condition rows sit near the clean end
    std::set<double> unique_set{t_v, t_a};
    const double t_ca = std::max(t_a, kAudioCondT);
    if (n_cond) unique_set.insert(t_c);
    if (n_cond_a) unique_set.insert(t_ca);
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
    const int64_t c1 = text_len + k_cond, c2 = c1 + n_cond_a, a0 = text_len + n_cond + n_cond_a, v0 = a0 + layout.audio;
    if (k_cond) segments.push_back({text_len, c1, row_of(t_c) * 3 + kTagVideo});
    if (n_cond_a) segments.push_back({c1, c2, row_of(t_ca) * 3 + kTagAudio});
    if (n_cond > k_cond) segments.push_back({c2, a0, row_of(t_c) * 3 + kTagVideo});
    segments.push_back({a0, v0, row_of(t_a) * 3 + kTagAudio});
    segments.push_back({v0, layout.seq, row_of(t_v) * 3 + kTagVideo});

    // embed and pack [text | audio | video]
    auto video_rows = video_x.to(at::kFloat).reshape({1, kVideoC, latent_t, 1, lat_h / 2, 2, lat_w / 2, 2})
                          .permute({0, 2, 4, 6, 1, 3, 5, 7}).reshape({-1, kVideoC * 4});
    // one GEMM over [condition rows | target rows], as ComfyUI packs them
    if (n_cond) video_rows = at::cat({cond_rows->to(video_rows.device()).to(at::kFloat), video_rows});
    auto video_embed = at::linear(video_rows, f32(w, "video_patch_proj.weight"), f32(w, "video_patch_proj.bias")).to(dtype);
    auto audio_rows = audio_x.to(at::kFloat)[0].permute({1, 2, 0}).reshape({-1, kAudioC});
    if (n_cond_a) audio_rows = at::cat({cond_audio_rows->to(audio_rows.device()).to(at::kFloat), audio_rows});
    auto audio_embed = at::linear(audio_rows, f32(w, "audio_patch_proj.weight"), f32(w, "audio_patch_proj.bias")).to(dtype);
    auto h = at::cat({context[0], video_embed.narrow(0, 0, k_cond), audio_embed.narrow(0, 0, n_cond_a),
                      video_embed.narrow(0, k_cond, n_cond - k_cond), audio_embed.narrow(0, n_cond_a, layout.audio),
                      video_embed.narrow(0, n_cond, layout.video)});

    auto t_emb = time_embedding(w, unique_t, device);
    auto table = rope_table(layout, f32(w, "rope.inv_freq").to(device), dtype);
    std::optional<Sparse> sparse;
    if (!stats.empty()) {  // sinks: every block before the video stays exact; audio query blocks run dense
        const int64_t sink_end = (v0 + kBlock - 1) / kBlock;
        sparse = Sparse{packed_rope_fab(table), 0, sink_end, a0 / kBlock, sink_end, stats, first};
    }
    for (int64_t i = 0; w.contains("blocks." + std::to_string(i) + ".norm1.weight"); ++i) {
        h = block(w, i, h, t_emb, segments, table, sparse ? &*sparse : nullptr);
    }

    // final layer: rows index t_emb directly (one modality)
    auto mod = adaln(w, "final_layer.adaln_proj", t_emb, 2, 1);  // shift, scale
    auto head = [&](int64_t a, int64_t b, double t, const std::string& name) {
        auto normed = at::rms_norm(h.narrow(0, a, b - a), {kHidden}, w.at("final_layer.norm.weight"), kEps);
        auto x = (normed * (1.0 + mod[1][row_of(t)]) + mod[0][row_of(t)]).to(at::kFloat);
        return at::linear(x, f32(w, name + ".weight"), f32(w, name + ".bias"));
    };
    auto v = head(v0, layout.seq, t_v, "final_layer.video_out");
    auto a = head(a0, v0, t_a, "final_layer.audio_out");  // target audio only
    auto video_out = v.reshape({1, latent_t, lat_h / 2, lat_w / 2, kVideoC, 1, 2, 2})
                         .permute({0, 4, 1, 5, 2, 6, 3, 7}).reshape({1, kVideoC, latent_t, lat_h, lat_w});
    auto audio_out = a.reshape({2, audio_t, kAudioC}).permute({2, 0, 1}).unsqueeze(0);
    return {(-video_out).to(video_x.scalar_type()), (-audio_out).to(audio_x.scalar_type())};
}

// MiniMaxH3Model.forward: the sampler carries audio as (sigma_v / sigma_a) * x_audio.
std::vector<Tensor> forward(const Weights& w, const Tensor& video_x, const Tensor& audio_x, const Tensor& timestep,
                            const Tensor& context, const Tensor& token_tags, double shift_v, double shift_a,
                            double audio_scale, std::vector<Tensor> stats, bool first, const std::optional<Tensor>& cond_rows,
                            const std::optional<Tensor>& positions, const std::optional<Tensor>& cond_audio_rows,
                            int64_t cond_audio_at) {
    auto audio_src = audio_x;
    Tensor carry, sigma_a;
    auto x_audio = audio_x;
    if (audio_scale != 1.0) {
        auto sigma_v = (timestep.flatten()[0] / 1000.0).to(at::kFloat).clamp_min(1e-6);
        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a);
        carry = (sigma_a / sigma_v).to(audio_src.scalar_type());
        x_audio = audio_src * carry;
    }
    auto out = forward_impl(w, video_x, x_audio, timestep, context, token_tags, shift_v, shift_a, stats, first, cond_rows, positions, cond_audio_rows, cond_audio_at);
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
          "Tensor token_tags, float shift_video, float shift_audio, float audio_scale, Tensor(a!)[] sla_stats, "
          "bool sla_first, Tensor? cond_rows=None, Tensor? positions=None, Tensor? cond_audio_rows=None, int cond_audio_at=-1) -> Tensor[]", &forward);
    m.def("dit_block(Dict(str, Tensor) weights, int index, Tensor x, Tensor t_emb, int[] segments, Tensor rope_table) -> Tensor",
          &block_op);
}
