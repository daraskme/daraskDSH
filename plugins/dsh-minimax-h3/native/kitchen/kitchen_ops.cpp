// LibTorch bindings for the vendored comfy-kitchen v0.2.41 kernels (Apache-2.0): Sol-Attn and the
// fused per-head RMSNorm + split-half RoPE. Kernels live in libdsh_kitchen.so (nvcc, plain C ABI,
// see ../build_kitchen.py); every launch uses the caller's current LibTorch CUDA stream.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <optional>
#include <vector>

extern "C" {
int sol_attn_plan(int batch, int seq_len, int num_heads, int n_tok, int64_t* out, int cap);
void sol_producer_begin(void* workspace, int batch, int seq_len, int num_heads, int n_tok, cudaStream_t stream);
void sol_producer_chunk(void* workspace, const void* qkv, const void* fab, const void* qw, const void* kw,
                        const void* kmean, const void* vscale, const void* blen, float rope_eps, int rot_dim,
                        int t0, int M, int batch, int seq_len, int num_heads, int n_tok, cudaStream_t stream);
void launch_sol_attn_core(void* workspace, void* out, const void* vscale, void* kmean_next, void* vamax_out,
                          const void* blen, int tail, int batch, int seq_len, int num_heads, float tau, float scale,
                          const void* ext_threshold, int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                          int n_tok, cudaStream_t stream);
void launch_rms_rope_kernel(const void* q, const void* k, const void* freqs, const void* q_scale, const void* k_scale,
                            void* q_out, void* k_out, int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim,
                            int64_t rot_dim, int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2,
                            int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0, int64_t k_s1,
                            int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1, int64_t qo_s2, int64_t qo_s3,
                            int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3, int64_t f_s0, int64_t f_s1,
                            int64_t f_s2, int64_t f_s3, int64_t f_s4, int64_t f_s5, int64_t qs_stride,
                            int64_t ks_stride, float epsilon, int input_dtype_code, int freqs_dtype_code,
                            int scale_dtype_code, bool has_k, bool split_half, cudaStream_t stream);
void launch_apply_rope_kernel(const void* q, const void* k, const void* freqs, void* q_out, void* k_out, int64_t batch,
                              int64_t dim1, int64_t dim2, int64_t head_dim, int64_t freqs_batch, int64_t freqs_dim1,
                              int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0,
                              int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1, int64_t qo_s2,
                              int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3, int64_t f_s0,
                              int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4, int64_t f_s5,
                              int input_dtype_code, int freqs_dtype_code, bool has_k, bool split_half, cudaStream_t stream);
}

namespace dsh_kitchen {

cudaStream_t stream_of(const at::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "dsh_kitchen: tensors must be on CUDA");
    return c10::cuda::getCurrentCUDAStream(t.get_device()).stream();
}

const void* ptr(const std::optional<at::Tensor>& t) {
    return t.has_value() && t->defined() ? t->const_data_ptr() : nullptr;
}

std::vector<int64_t> plan(int64_t batch, int64_t seq_len, int64_t num_heads, int64_t token_aug) {
    std::vector<int64_t> values(48);
    const int n = sol_attn_plan((int)batch, (int)seq_len, (int)num_heads, (int)token_aug, values.data(), 48);
    TORCH_CHECK(n > 0 && n <= 48, "dsh_kitchen: sol_attn_plan failed");
    values.resize(n);
    return values;
}

void producer_begin(at::Tensor workspace, int64_t batch, int64_t seq_len, int64_t num_heads, int64_t token_aug) {
    sol_producer_begin(workspace.data_ptr(), (int)batch, (int)seq_len, (int)num_heads, (int)token_aug,
                       stream_of(workspace));
}

void producer_chunk(at::Tensor workspace, const at::Tensor& qkv, const at::Tensor& fab, const at::Tensor& qw,
                    const at::Tensor& kw, const at::Tensor& kmean, const at::Tensor& vscale,
                    const std::optional<at::Tensor>& block_len, double rope_eps, int64_t rot_dim, int64_t t0,
                    int64_t m, int64_t batch, int64_t seq_len, int64_t num_heads, int64_t token_aug) {
    TORCH_CHECK(qkv.is_contiguous() && fab.is_contiguous(), "dsh_kitchen: qkv and fab must be contiguous");
    sol_producer_chunk(workspace.data_ptr(), qkv.const_data_ptr(), fab.const_data_ptr(), qw.const_data_ptr(),
                       kw.const_data_ptr(), kmean.const_data_ptr(), vscale.const_data_ptr(), ptr(block_len),
                       (float)rope_eps, (int)rot_dim, (int)t0, (int)m, (int)batch, (int)seq_len, (int)num_heads,
                       (int)token_aug, stream_of(workspace));
}

void attn_core(at::Tensor workspace, at::Tensor out, const at::Tensor& vscale, at::Tensor kmean_next,
               at::Tensor vamax, const std::optional<at::Tensor>& block_len, bool tail, int64_t batch,
               int64_t seq_len, int64_t num_heads, double tau, double scale,
               const std::optional<at::Tensor>& threshold, int64_t sink_start, int64_t sink_end,
               int64_t sink_q_start, int64_t sink_q_end, int64_t token_aug) {
    launch_sol_attn_core(workspace.data_ptr(), out.data_ptr(), vscale.const_data_ptr(), kmean_next.data_ptr(),
                         vamax.data_ptr(), ptr(block_len), tail ? 1 : 0, (int)batch, (int)seq_len, (int)num_heads,
                         (float)tau, (float)scale, ptr(threshold), (int)sink_start, (int)sink_end,
                         (int)sink_q_start, (int)sink_q_end, (int)token_aug, stream_of(workspace));
}

int dtype_code(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kFloat: return 0;
        case at::kHalf: return 1;
        case at::kBFloat16: return 2;
        default: TORCH_CHECK(false, "dsh_kitchen: unsupported dtype ", t.scalar_type());
    }
}

// In-place per-head RMSNorm (over head_dim) + split-half RoPE of the first rot_dim dims of q and k
// ([B, d1, d2, head_dim] views, any strides); freqs [*, *, *, rot_dim/2, 2, 2] broadcasting to q.
void rms_rope_split_half_(at::Tensor q, at::Tensor k, const at::Tensor& freqs, const at::Tensor& q_scale,
                          const at::Tensor& k_scale, double epsilon, int64_t rot_dim) {
    TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && freqs.dim() == 6, "dsh_kitchen.rms_rope: bad shapes");
    const auto rot = rot_dim > 0 ? rot_dim : q.size(3);
    TORCH_CHECK(freqs.size(3) == rot / 2 && q_scale.dim() == 1 && q_scale.size(0) == q.size(3),
                "dsh_kitchen.rms_rope: freqs/scale do not match q");
    launch_rms_rope_kernel(q.data_ptr(), k.data_ptr(), freqs.const_data_ptr(), q_scale.const_data_ptr(),
                           k_scale.const_data_ptr(), q.data_ptr(), k.data_ptr(), q.size(0), q.size(1), q.size(2),
                           q.size(3), rot, freqs.size(0), freqs.size(1), freqs.size(2), q.stride(0), q.stride(1),
                           q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), q.stride(0),
                           q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                           freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3), freqs.stride(4),
                           freqs.stride(5), q_scale.stride(0), k_scale.stride(0), (float)epsilon, dtype_code(q),
                           dtype_code(freqs), dtype_code(q_scale), true, true, stream_of(q));
}

// Split-half RoPE of one [B, d1, d2, head_dim] tensor into `out` (comfy-kitchen apply_rope_split_half1);
// freqs [*, *, *, head_dim/2, 2, 2] broadcasting to x.
void apply_rope_split_half1(const at::Tensor& x, const at::Tensor& freqs, at::Tensor out) {
    TORCH_CHECK(x.dim() == 4 && freqs.dim() == 6 && out.sizes() == x.sizes(), "dsh_kitchen.apply_rope: bad shapes");
    launch_apply_rope_kernel(x.const_data_ptr(), nullptr, freqs.const_data_ptr(), out.data_ptr(), nullptr, x.size(0),
                             x.size(1), x.size(2), x.size(3), freqs.size(0), freqs.size(1), freqs.size(2), x.stride(0),
                             x.stride(1), x.stride(2), x.stride(3), 0, 0, 0, 0, out.stride(0), out.stride(1),
                             out.stride(2), out.stride(3), 0, 0, 0, 0, freqs.stride(0), freqs.stride(1), freqs.stride(2),
                             freqs.stride(3), freqs.stride(4), freqs.stride(5), dtype_code(x), dtype_code(freqs), false,
                             true, stream_of(x));
}

}  // namespace dsh_kitchen

using namespace dsh_kitchen;

at::Tensor apply_rope_split_half1_op(const at::Tensor& x, const at::Tensor& freqs) {
    auto out = at::empty_like(x);
    apply_rope_split_half1(x, freqs, out);
    return out;
}

TORCH_LIBRARY(dsh_kitchen, m) {
    m.def("apply_rope_split_half1(Tensor x, Tensor freqs) -> Tensor", &apply_rope_split_half1_op);
    m.def("rms_rope_split_half_(Tensor(a!) q, Tensor(b!) k, Tensor freqs, Tensor q_scale, Tensor k_scale, float epsilon, "
          "int rot_dim) -> ()", &rms_rope_split_half_);
    m.def("plan(int batch, int seq_len, int num_heads, int token_aug) -> int[]", &plan);
    m.def("producer_begin(Tensor(a!) workspace, int batch, int seq_len, int num_heads, int token_aug) -> ()",
          &producer_begin);
    m.def("producer_chunk(Tensor(a!) workspace, Tensor qkv, Tensor fab, Tensor qw, Tensor kw, Tensor kmean, "
          "Tensor vscale, Tensor? block_len, float rope_eps, int rot_dim, int t0, int m, int batch, int seq_len, "
          "int num_heads, int token_aug) -> ()", &producer_chunk);
    m.def("attn_core(Tensor(a!) workspace, Tensor(b!) out, Tensor vscale, Tensor(c!) kmean_next, Tensor(d!) vamax, "
          "Tensor? block_len, bool tail, int batch, int seq_len, int num_heads, float tau, float scale, "
          "Tensor? threshold, int sink_start, int sink_end, int sink_q_start, int sink_q_end, int token_aug) -> ()",
          &attn_core);
}
