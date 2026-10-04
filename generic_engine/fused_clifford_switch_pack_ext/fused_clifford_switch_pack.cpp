#include <torch/extension.h>
#include <vector>

torch::Tensor cign_switch_pack_forward_cuda(
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha,
    int64_t interaction_mode,
    int64_t weighting_mode);

std::vector<torch::Tensor> cign_switch_pack_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha,
    int64_t interaction_mode,
    int64_t weighting_mode);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static void validate(
    const torch::Tensor& H,
    const torch::Tensor& C,
    const torch::Tensor& P0,
    const torch::Tensor& P1,
    const torch::Tensor& P2,
    const torch::Tensor& alpha,
    int64_t interaction_mode,
    int64_t weighting_mode) {
  CHECK_INPUT(H);
  CHECK_INPUT(C);
  CHECK_INPUT(P0);
  CHECK_INPUT(P1);
  CHECK_INPUT(P2);
  CHECK_INPUT(alpha);

  TORCH_CHECK(H.dim() == 2, "H must be [N,D]");
  TORCH_CHECK(C.sizes() == H.sizes(), "C must match H");

  const auto N = H.size(0);
  const auto D = H.size(1);

  TORCH_CHECK(
      P0.dim() == 2 && P0.size(0) == N && P0.size(1) == 2 * D,
      "P0 must be [N,2D]");
  TORCH_CHECK(P1.sizes() == P0.sizes(), "P1 must match P0");
  TORCH_CHECK(P2.sizes() == P0.sizes(), "P2 must match P0");
  TORCH_CHECK(
      alpha.dim() == 2 && alpha.size(0) == N && alpha.size(1) == 3,
      "alpha must be [N,3]");

  TORCH_CHECK(
      interaction_mode == 0 || interaction_mode == 1,
      "interaction_mode must be 0(old) or 1(new)");
  TORCH_CHECK(
      weighting_mode == 0 || weighting_mode == 1,
      "weighting_mode must be 0(old) or 1(new)");
}

torch::Tensor forward(
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha,
    int64_t interaction_mode,
    int64_t weighting_mode) {
  validate(
      H, C, P0, P1, P2, alpha,
      interaction_mode, weighting_mode);
  return cign_switch_pack_forward_cuda(
      H, C, P0, P1, P2, alpha,
      interaction_mode, weighting_mode);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_out,
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha,
    int64_t interaction_mode,
    int64_t weighting_mode) {
  validate(
      H, C, P0, P1, P2, alpha,
      interaction_mode, weighting_mode);

  CHECK_INPUT(grad_out);
  const auto N = H.size(0);
  const auto D = H.size(1);

  if (weighting_mode == 0) {
    TORCH_CHECK(
        grad_out.dim() == 2 &&
        grad_out.size(0) == N &&
        grad_out.size(1) == 6 * D,
        "old weighting grad_out must be [N,6D]");
  } else {
    TORCH_CHECK(
        grad_out.dim() == 2 &&
        grad_out.size(0) == N &&
        grad_out.size(1) == 2 * D,
        "new weighting grad_out must be [N,2D]");
  }

  return cign_switch_pack_backward_cuda(
      grad_out, H, C, P0, P1, P2, alpha,
      interaction_mode, weighting_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "CIGN switch pack forward (CUDA)");
  m.def("backward", &backward, "CIGN switch pack backward (CUDA)");
}
