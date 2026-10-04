#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace {

__device__ __forceinline__ float silu_f(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return x * s;
}

__device__ __forceinline__ float silu_grad_f(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

__device__ __forceinline__ void compute_dw(
    float bhr,
    float brh,
    int interaction_mode,
    float& Dv,
    float& Wv,
    float& dD_dbhr,
    float& dD_dbrh,
    float& dW_dbhr,
    float& dW_dbrh) {

  if (interaction_mode == 0) {
    // OLD
    Dv = silu_f(bhr);
    Wv = bhr - brh;

    dD_dbhr = silu_grad_f(bhr);
    dD_dbrh = 0.0f;
    dW_dbhr = 1.0f;
    dW_dbrh = -1.0f;
  } else {
    // NEW
    const float sym = 0.5f * (bhr + brh);
    const float anti = 0.5f * (bhr - brh);

    Dv = silu_f(sym);
    Wv = tanhf(anti);

    const float dD_dsym = silu_grad_f(sym);
    const float dW_danti = 1.0f - Wv * Wv;

    dD_dbhr = 0.5f * dD_dsym;
    dD_dbrh = 0.5f * dD_dsym;
    dW_dbhr = 0.5f * dW_danti;
    dW_dbrh = -0.5f * dW_danti;
  }
}


// ============================================================
// OLD weighting forward: Cat_k(alpha_k * [W_k,D_k])
// output [N,6D]
// ============================================================
__global__ void forward_old_weight_kernel(
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    const float* __restrict__ A,
    float* __restrict__ O,
    int64_t N,
    int64_t D,
    int interaction_mode) {

  const int64_t total = N * 3 * D;

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {

    const int64_t d = idx % D;
    const int64_t q = idx / D;
    const int64_t k = q % 3;
    const int64_t n = q / 3;

    const float* P = (
        k == 0 ? P0 : (k == 1 ? P1 : P2)
    );

    const int64_t hidx = n * D + d;
    const int64_t pbase = n * (2 * D);

    const float h = H[hidx];
    const float c = C[hidx];
    const float th = P[pbase + d];
    const float tc = P[pbase + D + d];

    const float bhr = h * tc;
    const float brh = c * th;

    float Dv, Wv, a1, a2, a3, a4;
    compute_dw(
        bhr, brh, interaction_mode,
        Dv, Wv, a1, a2, a3, a4
    );

    const float alpha = A[n * 3 + k];
    const int64_t obase = n * (6 * D) + k * (2 * D);

    O[obase + d] = alpha * Wv;
    O[obase + D + d] = alpha * Dv;
  }
}


// ============================================================
// NEW weighting forward:
// Dagg=sum alpha*D, Wagg=sum alpha*W, output [Dagg,Wagg]
// output [N,2D]
// ============================================================
__global__ void forward_new_weight_kernel(
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    const float* __restrict__ A,
    float* __restrict__ O,
    int64_t N,
    int64_t D,
    int interaction_mode) {

  const int64_t total = N * D;

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {

    const int64_t d = idx % D;
    const int64_t n = idx / D;

    const float h = H[idx];
    const float c = C[idx];

    float Dagg = 0.0f;
    float Wagg = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      const float* P = (
          k == 0 ? P0 : (k == 1 ? P1 : P2)
      );
      const int64_t pbase = n * (2 * D);
      const float th = P[pbase + d];
      const float tc = P[pbase + D + d];

      const float bhr = h * tc;
      const float brh = c * th;

      float Dv, Wv, a1, a2, a3, a4;
      compute_dw(
          bhr, brh, interaction_mode,
          Dv, Wv, a1, a2, a3, a4
      );

      const float alpha = A[n * 3 + k];
      Dagg += alpha * Dv;
      Wagg += alpha * Wv;
    }

    // Teacher-modified order: Cat(Dagg, Wagg).
    O[n * (2 * D) + d] = Dagg;
    O[n * (2 * D) + D + d] = Wagg;
  }
}


// ============================================================
// Backward main: one thread per (node,channel), loop over 3 hops.
// No atomicAdd needed for H/C or P because each thread owns one channel.
// ============================================================
__global__ void backward_main_kernel(
    const float* __restrict__ GO,
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    const float* __restrict__ A,
    float* __restrict__ GH,
    float* __restrict__ GC,
    float* __restrict__ GP0,
    float* __restrict__ GP1,
    float* __restrict__ GP2,
    int64_t N,
    int64_t D,
    int interaction_mode,
    int weighting_mode) {

  const int64_t total = N * D;

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {

    const int64_t d = idx % D;
    const int64_t n = idx / D;

    const float h = H[idx];
    const float c = C[idx];

    float gh = 0.0f;
    float gc = 0.0f;

    // For new weighting the same two gradients feed all hops.
    const float goD_agg = (
        weighting_mode == 1
        ? GO[n * (2 * D) + d]
        : 0.0f
    );
    const float goW_agg = (
        weighting_mode == 1
        ? GO[n * (2 * D) + D + d]
        : 0.0f
    );

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      const float* P = (
          k == 0 ? P0 : (k == 1 ? P1 : P2)
      );
      float* GP = (
          k == 0 ? GP0 : (k == 1 ? GP1 : GP2)
      );

      const int64_t pbase = n * (2 * D);
      const float th = P[pbase + d];
      const float tc = P[pbase + D + d];

      const float bhr = h * tc;
      const float brh = c * th;

      float Dv, Wv;
      float dD_dbhr, dD_dbrh;
      float dW_dbhr, dW_dbrh;

      compute_dw(
          bhr, brh, interaction_mode,
          Dv, Wv,
          dD_dbhr, dD_dbrh,
          dW_dbhr, dW_dbrh
      );

      const float alpha = A[n * 3 + k];

      float gD;
      float gW;

      if (weighting_mode == 0) {
        const int64_t obase =
            n * (6 * D) + k * (2 * D);
        gW = GO[obase + d] * alpha;
        gD = GO[obase + D + d] * alpha;
      } else {
        gD = goD_agg * alpha;
        gW = goW_agg * alpha;
      }

      const float gbhr =
          gD * dD_dbhr + gW * dW_dbhr;
      const float gbrh =
          gD * dD_dbrh + gW * dW_dbrh;

      // bhr = h * tc
      // brh = c * th
      gh += gbhr * tc;
      gc += gbrh * th;

      GP[pbase + d] = gbrh * c;       // grad TH
      GP[pbase + D + d] = gbhr * h;   // grad TC
    }

    GH[idx] = gh;
    GC[idx] = gc;
  }
}


// ============================================================
// Alpha gradient: one warp per (node,hop), reduce over D.
// ============================================================
__global__ void backward_alpha_kernel(
    const float* __restrict__ GO,
    const float* __restrict__ H,
    const float* __restrict__ C,
    const float* __restrict__ P0,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    float* __restrict__ GA,
    int64_t N,
    int64_t D,
    int interaction_mode,
    int weighting_mode) {

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;

  const int64_t nk =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  const int64_t total_nk = N * 3;

  if (nk >= total_nk) {
    return;
  }

  const int64_t n = nk / 3;
  const int k = static_cast<int>(nk % 3);

  const float* P = (
      k == 0 ? P0 : (k == 1 ? P1 : P2)
  );
  const int64_t pbase = n * (2 * D);

  float local = 0.0f;

  for (int64_t d = lane; d < D; d += 32) {
    const float h = H[n * D + d];
    const float c = C[n * D + d];
    const float th = P[pbase + d];
    const float tc = P[pbase + D + d];

    const float bhr = h * tc;
    const float brh = c * th;

    float Dv, Wv, a1, a2, a3, a4;
    compute_dw(
        bhr, brh, interaction_mode,
        Dv, Wv, a1, a2, a3, a4
    );

    if (weighting_mode == 0) {
      const int64_t obase =
          n * (6 * D) + k * (2 * D);
      local +=
          GO[obase + d] * Wv +
          GO[obase + D + d] * Dv;
    } else {
      const float goD = GO[n * (2 * D) + d];
      const float goW = GO[n * (2 * D) + D + d];
      local += goD * Dv + goW * Wv;
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, offset);
  }

  if (lane == 0) {
    GA[n * 3 + k] = local;
  }
}

} // namespace


torch::Tensor cign_switch_pack_forward_cuda(
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha,
    int64_t interaction_mode,
    int64_t weighting_mode) {

  c10::cuda::CUDAGuard guard(H.device());

  const int64_t N = H.size(0);
  const int64_t D = H.size(1);

  auto out = (
      weighting_mode == 0
      ? torch::empty({N, 6 * D}, H.options())
      : torch::empty({N, 2 * D}, H.options())
  );

  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (weighting_mode == 0) {
    const int64_t total = N * 3 * D;
    const int blocks = static_cast<int>(
        std::min<int64_t>(
            (total + threads - 1) / threads,
            65535
        )
    );

    forward_old_weight_kernel<<<blocks, threads, 0, stream>>>(
        H.data_ptr<float>(),
        C.data_ptr<float>(),
        P0.data_ptr<float>(),
        P1.data_ptr<float>(),
        P2.data_ptr<float>(),
        alpha.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D,
        static_cast<int>(interaction_mode)
    );
  } else {
    const int64_t total = N * D;
    const int blocks = static_cast<int>(
        std::min<int64_t>(
            (total + threads - 1) / threads,
            65535
        )
    );

    forward_new_weight_kernel<<<blocks, threads, 0, stream>>>(
        H.data_ptr<float>(),
        C.data_ptr<float>(),
        P0.data_ptr<float>(),
        P1.data_ptr<float>(),
        P2.data_ptr<float>(),
        alpha.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D,
        static_cast<int>(interaction_mode)
    );
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


std::vector<torch::Tensor> cign_switch_pack_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor H,
    torch::Tensor C,
    torch::Tensor P0,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor alpha,
    int64_t interaction_mode,
    int64_t weighting_mode) {

  c10::cuda::CUDAGuard guard(H.device());

  const int64_t N = H.size(0);
  const int64_t D = H.size(1);

  auto GH = torch::empty_like(H);
  auto GC = torch::empty_like(C);
  auto GP0 = torch::empty_like(P0);
  auto GP1 = torch::empty_like(P1);
  auto GP2 = torch::empty_like(P2);
  auto GA = torch::empty_like(alpha);

  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t total = N * D;
  const int blocks = static_cast<int>(
      std::min<int64_t>(
          (total + threads - 1) / threads,
          65535
      )
  );

  backward_main_kernel<<<blocks, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      H.data_ptr<float>(),
      C.data_ptr<float>(),
      P0.data_ptr<float>(),
      P1.data_ptr<float>(),
      P2.data_ptr<float>(),
      alpha.data_ptr<float>(),
      GH.data_ptr<float>(),
      GC.data_ptr<float>(),
      GP0.data_ptr<float>(),
      GP1.data_ptr<float>(),
      GP2.data_ptr<float>(),
      N,
      D,
      static_cast<int>(interaction_mode),
      static_cast<int>(weighting_mode)
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int warps_per_block = threads / 32;
  const int alpha_blocks = static_cast<int>(
      (N * 3 + warps_per_block - 1) / warps_per_block
  );

  backward_alpha_kernel<<<alpha_blocks, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      H.data_ptr<float>(),
      C.data_ptr<float>(),
      P0.data_ptr<float>(),
      P1.data_ptr<float>(),
      P2.data_ptr<float>(),
      GA.data_ptr<float>(),
      N,
      D,
      static_cast<int>(interaction_mode),
      static_cast<int>(weighting_mode)
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {GH, GC, GP0, GP1, GP2, GA};
}
