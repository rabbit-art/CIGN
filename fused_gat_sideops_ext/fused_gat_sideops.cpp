#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> attention_forward_cuda(
    torch::Tensor x,
    torch::Tensor att_src,
    torch::Tensor att_dst);

std::vector<torch::Tensor> attention_backward_cuda(
    torch::Tensor grad_src,
    torch::Tensor grad_dst,
    torch::Tensor x,
    torch::Tensor att_src,
    torch::Tensor att_dst);

std::vector<torch::Tensor> head_silu_forward_cuda(
    torch::Tensor out,
    torch::Tensor bias);

std::vector<torch::Tensor> head_silu_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor pre,
    int64_t heads);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static void validate_attention(
    const torch::Tensor& x,
    const torch::Tensor& att_src,
    const torch::Tensor& att_dst) {
  CHECK_INPUT(x);
  CHECK_INPUT(att_src);
  CHECK_INPUT(att_dst);
  TORCH_CHECK(x.dim() == 3, "x must be [N,H,C]");
  TORCH_CHECK(att_src.dim() == 3 && att_src.size(0) == 1,
              "att_src must be [1,H,C]");
  TORCH_CHECK(att_dst.sizes() == att_src.sizes(), "att_dst must match att_src");
  TORCH_CHECK(att_src.size(1) == x.size(1) && att_src.size(2) == x.size(2),
              "attention shapes do not match x");
}

std::vector<torch::Tensor> attention_forward(
    torch::Tensor x,
    torch::Tensor att_src,
    torch::Tensor att_dst) {
  validate_attention(x, att_src, att_dst);
  return attention_forward_cuda(x, att_src, att_dst);
}

std::vector<torch::Tensor> attention_backward(
    torch::Tensor grad_src,
    torch::Tensor grad_dst,
    torch::Tensor x,
    torch::Tensor att_src,
    torch::Tensor att_dst) {
  validate_attention(x, att_src, att_dst);
  CHECK_INPUT(grad_src);
  CHECK_INPUT(grad_dst);
  TORCH_CHECK(grad_src.dim() == 2 && grad_src.size(0) == x.size(0) && grad_src.size(1) == x.size(1),
              "grad_src must be [N,H]");
  TORCH_CHECK(grad_dst.sizes() == grad_src.sizes(), "grad_dst must match grad_src");
  return attention_backward_cuda(grad_src, grad_dst, x, att_src, att_dst);
}

std::vector<torch::Tensor> head_silu_forward(
    torch::Tensor out,
    torch::Tensor bias) {
  CHECK_INPUT(out);
  CHECK_INPUT(bias);
  TORCH_CHECK(out.dim() == 3, "out must be [N,H,C]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out.size(2), "bias must be [C]");
  return head_silu_forward_cuda(out, bias);
}

std::vector<torch::Tensor> head_silu_backward(
    torch::Tensor grad_y,
    torch::Tensor pre,
    int64_t heads) {
  CHECK_INPUT(grad_y);
  CHECK_INPUT(pre);
  TORCH_CHECK(grad_y.dim() == 2, "grad_y must be [N,C]");
  TORCH_CHECK(pre.sizes() == grad_y.sizes(), "pre must match grad_y");
  TORCH_CHECK(heads > 0, "heads must be positive");
  return head_silu_backward_cuda(grad_y, pre, heads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_forward", &attention_forward, "Fused attention logits forward (CUDA)");
  m.def("attention_backward", &attention_backward, "Fused attention logits backward (CUDA)");
  m.def("head_silu_forward", &head_silu_forward, "Fused head mean + bias + SiLU forward (CUDA)");
  m.def("head_silu_backward", &head_silu_backward, "Fused head mean + bias + SiLU backward (CUDA)");
}
