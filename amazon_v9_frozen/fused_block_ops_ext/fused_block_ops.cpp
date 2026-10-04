#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> dropout_layernorm_forward_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    double dropout_p, double eps, int64_t seed);
std::vector<torch::Tensor> dropout_layernorm_backward_cuda(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor weight,
    torch::Tensor mean, torch::Tensor rstd, torch::Tensor mask,
    double dropout_p);
std::vector<torch::Tensor> add_layernorm_forward_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight, torch::Tensor bias,
    double eps);
std::vector<torch::Tensor> add_layernorm_backward_cuda(
    torch::Tensor grad_y, torch::Tensor a, torch::Tensor b,
    torch::Tensor weight, torch::Tensor mean, torch::Tensor rstd);
std::vector<torch::Tensor> dropout_gamma_residual_forward_cuda(
    torch::Tensor h, torch::Tensor g, torch::Tensor gamma,
    double dropout_p, int64_t seed);
std::vector<torch::Tensor> dropout_gamma_residual_backward_cuda(
    torch::Tensor grad_y, torch::Tensor g, torch::Tensor gamma,
    torch::Tensor mask, double dropout_p);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_U8(x) TORCH_CHECK(x.scalar_type() == at::kByte, #x " must be uint8")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static void check_2d_256(const torch::Tensor& x, const char* name) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 256, name, " must be [N,256]");
}
static void check_param_256(const torch::Tensor& x, const char* name) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() == 1 && x.numel() == 256, name, " must be [256]");
}

std::vector<torch::Tensor> dropout_layernorm_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    double dropout_p, double eps, int64_t seed) {
  check_2d_256(x, "x"); check_param_256(weight, "weight"); check_param_256(bias, "bias");
  TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must be in [0,1)");
  return dropout_layernorm_forward_cuda(x, weight, bias, dropout_p, eps, seed);
}
std::vector<torch::Tensor> dropout_layernorm_backward(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor weight,
    torch::Tensor mean, torch::Tensor rstd, torch::Tensor mask,
    double dropout_p) {
  check_2d_256(grad_y, "grad_y"); check_2d_256(x, "x"); check_param_256(weight, "weight");
  CHECK_INPUT(mean); CHECK_INPUT(rstd); CHECK_CUDA(mask); CHECK_CONTIGUOUS(mask); CHECK_U8(mask);
  TORCH_CHECK(mean.dim()==1 && mean.size(0)==x.size(0), "mean must be [N]");
  TORCH_CHECK(rstd.sizes()==mean.sizes(), "rstd must be [N]");
  TORCH_CHECK(mask.sizes()==x.sizes(), "mask must match x");
  return dropout_layernorm_backward_cuda(grad_y, x, weight, mean, rstd, mask, dropout_p);
}
std::vector<torch::Tensor> add_layernorm_forward(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight, torch::Tensor bias,
    double eps) {
  check_2d_256(a, "a"); check_2d_256(b, "b"); check_param_256(weight, "weight"); check_param_256(bias, "bias");
  TORCH_CHECK(a.sizes()==b.sizes(), "a and b must match");
  return add_layernorm_forward_cuda(a, b, weight, bias, eps);
}
std::vector<torch::Tensor> add_layernorm_backward(
    torch::Tensor grad_y, torch::Tensor a, torch::Tensor b,
    torch::Tensor weight, torch::Tensor mean, torch::Tensor rstd) {
  check_2d_256(grad_y, "grad_y"); check_2d_256(a, "a"); check_2d_256(b, "b"); check_param_256(weight, "weight");
  CHECK_INPUT(mean); CHECK_INPUT(rstd);
  TORCH_CHECK(a.sizes()==b.sizes() && a.sizes()==grad_y.sizes(), "tensor shapes must match");
  TORCH_CHECK(mean.dim()==1 && mean.size(0)==a.size(0), "mean must be [N]");
  TORCH_CHECK(rstd.sizes()==mean.sizes(), "rstd must be [N]");
  return add_layernorm_backward_cuda(grad_y, a, b, weight, mean, rstd);
}
std::vector<torch::Tensor> dropout_gamma_residual_forward(
    torch::Tensor h, torch::Tensor g, torch::Tensor gamma,
    double dropout_p, int64_t seed) {
  check_2d_256(h, "h"); check_2d_256(g, "g"); check_param_256(gamma, "gamma");
  TORCH_CHECK(h.sizes()==g.sizes(), "h and g must match");
  TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must be in [0,1)");
  return dropout_gamma_residual_forward_cuda(h, g, gamma, dropout_p, seed);
}
std::vector<torch::Tensor> dropout_gamma_residual_backward(
    torch::Tensor grad_y, torch::Tensor g, torch::Tensor gamma,
    torch::Tensor mask, double dropout_p) {
  check_2d_256(grad_y, "grad_y"); check_2d_256(g, "g"); check_param_256(gamma, "gamma");
  CHECK_CUDA(mask); CHECK_CONTIGUOUS(mask); CHECK_U8(mask);
  TORCH_CHECK(grad_y.sizes()==g.sizes() && mask.sizes()==g.sizes(), "shape mismatch");
  return dropout_gamma_residual_backward_cuda(grad_y, g, gamma, mask, dropout_p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dropout_layernorm_forward", &dropout_layernorm_forward);
  m.def("dropout_layernorm_backward", &dropout_layernorm_backward);
  m.def("add_layernorm_forward", &add_layernorm_forward);
  m.def("add_layernorm_backward", &add_layernorm_backward);
  m.def("dropout_gamma_residual_forward", &dropout_gamma_residual_forward);
  m.def("dropout_gamma_residual_backward", &dropout_gamma_residual_backward);
}
