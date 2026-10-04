#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> collapsed_gat_forward_cuda(
    torch::Tensor attn_row,
    torch::Tensor attn_col,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    double negative_slope,
    torch::Tensor in_feat,
    torch::Tensor bias,
    double attn_drop,
    int64_t seed);

std::vector<torch::Tensor> collapsed_gat_backward_cuda(
    double negative_slope,
    double attn_drop,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor col_ptr,
    torch::Tensor row_ind,
    torch::Tensor permute,
    torch::Tensor edge_max,
    torch::Tensor edge_sum,
    torch::Tensor edge_mask,
    torch::Tensor in_feat,
    torch::Tensor attn_row,
    torch::Tensor attn_col,
    torch::Tensor pre,
    torch::Tensor grad_y);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")
#define CHECK_UINT8(x) TORCH_CHECK(x.scalar_type() == at::kByte, #x " must be uint8")

static void check_float_cuda(const torch::Tensor& x) {
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x);
}
static void check_int_cuda(const torch::Tensor& x) {
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT32(x);
}

std::vector<torch::Tensor> forward(
    torch::Tensor attn_row,
    torch::Tensor attn_col,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    double negative_slope,
    torch::Tensor in_feat,
    torch::Tensor bias,
    double attn_drop,
    int64_t seed) {
  check_float_cuda(attn_row);
  check_float_cuda(attn_col);
  check_int_cuda(row_ptr);
  check_int_cuda(col_ind);
  check_float_cuda(in_feat);
  check_float_cuda(bias);
  TORCH_CHECK(attn_row.dim() == 2, "attn_row must be [N,H]");
  TORCH_CHECK(attn_col.sizes() == attn_row.sizes(), "attn_col must match attn_row");
  TORCH_CHECK(in_feat.dim() == 3, "in_feat must be [N,H,C]");
  TORCH_CHECK(in_feat.size(0) == attn_row.size(0) && in_feat.size(1) == attn_row.size(1),
              "in_feat N/H mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == in_feat.size(2), "bias must be [C]");
  TORCH_CHECK(in_feat.size(1) == 6 && in_feat.size(2) == 256,
              "A3-v6 v1 requires H=6,C=256");
  TORCH_CHECK(attn_drop >= 0.0 && attn_drop < 1.0, "attn_drop must be in [0,1)");
  return collapsed_gat_forward_cuda(attn_row, attn_col, row_ptr, col_ind,
                                    negative_slope, in_feat, bias, attn_drop, seed);
}

std::vector<torch::Tensor> backward(
    double negative_slope,
    double attn_drop,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor col_ptr,
    torch::Tensor row_ind,
    torch::Tensor permute,
    torch::Tensor edge_max,
    torch::Tensor edge_sum,
    torch::Tensor edge_mask,
    torch::Tensor in_feat,
    torch::Tensor attn_row,
    torch::Tensor attn_col,
    torch::Tensor pre,
    torch::Tensor grad_y) {
  check_int_cuda(row_ptr); check_int_cuda(col_ind); check_int_cuda(col_ptr);
  check_int_cuda(row_ind); check_int_cuda(permute);
  check_float_cuda(edge_max); check_float_cuda(edge_sum);
  CHECK_CUDA(edge_mask); CHECK_CONTIGUOUS(edge_mask); CHECK_UINT8(edge_mask);
  check_float_cuda(in_feat); check_float_cuda(attn_row); check_float_cuda(attn_col);
  check_float_cuda(pre); check_float_cuda(grad_y);
  TORCH_CHECK(in_feat.size(1) == 6 && in_feat.size(2) == 256,
              "A3-v6 v1 requires H=6,C=256");
  TORCH_CHECK(pre.sizes() == grad_y.sizes(), "pre and grad_y must match");
  return collapsed_gat_backward_cuda(negative_slope, attn_drop, row_ptr, col_ind,
                                     col_ptr, row_ind, permute, edge_max, edge_sum,
                                     edge_mask, in_feat, attn_row, attn_col, pre, grad_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "A3-v6 collapsed-head GAT forward (CUDA)");
  m.def("backward", &backward, "A3-v6 collapsed-head GAT backward (CUDA)");
}
