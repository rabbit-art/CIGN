#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace {

__device__ __forceinline__ float silu_f(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return x * s;
}

__device__ __forceinline__ float silu_grad_f(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

__global__ void fused_pack_forward_kernel(
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    const float* __restrict__ alpha,
    float* __restrict__ out,
    int64_t N,
    int64_t D) {
  const int64_t total = N * 3 * D;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t d = linear % D;
    const int64_t tmp = linear / D;
    const int64_t k = tmp % 3;
    const int64_t n = tmp / 3;

    const float* P = (k == 0 ? P0 : (k == 1 ? P1 : P2));
    const int64_t pbase = n * (2 * D);
    const float TH = P[pbase + d];
    const float TC = P[pbase + D + d];
    const float h = H[n * D + d];
    const float c = C[n * D + d];
    const float z = h * TC;
    const float W = z - TH * c;
    const float Dv = silu_f(z);
    const float a = alpha[n * 3 + k];

    const int64_t obase = n * (6 * D) + k * (2 * D);
    out[obase + d] = a * W;
    out[obase + D + d] = a * Dv;
  }
}

__global__ void fused_pack_backward_main_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    const float* __restrict__ alpha,
    float* __restrict__ grad_H,
    float* __restrict__ grad_C,
    float* __restrict__ grad_P0,
    float* __restrict__ grad_P1,
    float* __restrict__ grad_P2,
    int64_t N,
    int64_t D) {
  const int64_t total = N * D;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t d = linear % D;
    const int64_t n = linear / D;
    const float h = H[n * D + d];
    const float c = C[n * D + d];
    float gh = 0.0f;
    float gc = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      const float* P = (k == 0 ? P0 : (k == 1 ? P1 : P2));
      float* gP = (k == 0 ? grad_P0 : (k == 1 ? grad_P1 : grad_P2));
      const int64_t pbase = n * (2 * D);
      const float TH = P[pbase + d];
      const float TC = P[pbase + D + d];
      const float z = h * TC;
      const float a = alpha[n * 3 + k];
      const int64_t obase = n * (6 * D) + k * (2 * D);
      const float gW = grad_out[obase + d] * a;
      const float gD = grad_out[obase + D + d] * a;
      const float gz = gW + gD * silu_grad_f(z);

      gh += gz * TC;
      gc += -gW * TH;
      gP[pbase + d] = -gW * c;
      gP[pbase + D + d] = gz * h;
    }

    grad_H[n * D + d] = gh;
    grad_C[n * D + d] = gc;
  }
}

__global__ void fused_pack_backward_alpha_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    float* __restrict__ grad_alpha,
    int64_t N,
    int64_t D) {
  // One warp handles one (node, hop). Eight warps per 256-thread block.
  // For D=256 each lane processes exactly 8 channels, avoiding ~73k tiny
  // blocks in the earlier reduction design.
  const int warp_id_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t nk = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id_in_block;
  const int64_t total_nk = N * 3;
  if (nk >= total_nk) return;

  const int64_t n = nk / 3;
  const int k = static_cast<int>(nk % 3);
  const float* P = (k == 0 ? P0 : (k == 1 ? P1 : P2));
  const int64_t pbase = n * (2 * D);
  const int64_t obase = n * (6 * D) + k * (2 * D);

  float local = 0.0f;
  for (int64_t d = lane; d < D; d += 32) {
    const float TH = P[pbase + d];
    const float TC = P[pbase + D + d];
    const float h = H[n * D + d];
    const float c = C[n * D + d];
    const float z = h * TC;
    const float W = z - TH * c;
    const float Dv = silu_f(z);
    local += grad_out[obase + d] * W + grad_out[obase + D + d] * Dv;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, offset);
  }
  if (lane == 0) grad_alpha[n * 3 + k] = local;
}

} // namespace

torch::Tensor fused_clifford_pack_forward_cuda(
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha) {
  c10::cuda::CUDAGuard device_guard(H.device());
  const auto N = H.size(0);
  const auto D = H.size(1);
  auto out = torch::empty({N, 6 * D}, H.options());

  constexpr int threads = 256;
  const int64_t total = N * 3 * D;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_pack_forward_kernel<<<blocks, threads, 0, stream>>>(
      H.data_ptr<float>(), C.data_ptr<float>(),
      P0.data_ptr<float>(), P1.data_ptr<float>(), P2.data_ptr<float>(),
      alpha.data_ptr<float>(), out.data_ptr<float>(), N, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> fused_clifford_pack_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha) {
  c10::cuda::CUDAGuard device_guard(H.device());
  const auto N = H.size(0);
  const auto D = H.size(1);
  auto grad_H = torch::empty_like(H);
  auto grad_C = torch::empty_like(C);
  auto grad_P0 = torch::empty_like(P0);
  auto grad_P1 = torch::empty_like(P1);
  auto grad_P2 = torch::empty_like(P2);
  auto grad_alpha = torch::empty_like(alpha);

  constexpr int threads = 256;
  const int64_t total = N * D;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fused_pack_backward_main_kernel<<<blocks, threads, 0, stream>>>(
      grad_out.data_ptr<float>(), H.data_ptr<float>(), C.data_ptr<float>(),
      P0.data_ptr<float>(), P1.data_ptr<float>(), P2.data_ptr<float>(), alpha.data_ptr<float>(),
      grad_H.data_ptr<float>(), grad_C.data_ptr<float>(),
      grad_P0.data_ptr<float>(), grad_P1.data_ptr<float>(), grad_P2.data_ptr<float>(), N, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int warps_per_block = threads / 32;
  const int alpha_blocks = static_cast<int>((N * 3 + warps_per_block - 1) / warps_per_block);
  fused_pack_backward_alpha_kernel<<<alpha_blocks, threads, 0, stream>>>(
      grad_out.data_ptr<float>(), H.data_ptr<float>(), C.data_ptr<float>(),
      P0.data_ptr<float>(), P1.data_ptr<float>(), P2.data_ptr<float>(),
      grad_alpha.data_ptr<float>(), N, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_H, grad_C, grad_P0, grad_P1, grad_P2, grad_alpha};
}
