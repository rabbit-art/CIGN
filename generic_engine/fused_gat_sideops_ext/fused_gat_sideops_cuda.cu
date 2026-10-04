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

__device__ __forceinline__ float silu_grad_from_pre(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

// One warp handles one (node, head), directly reducing C channels for both
// source and destination logits. This is retained from A3-v2 because its
// forward path was already fast in the profiler.
__global__ void attention_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ att_src,
    const float* __restrict__ att_dst,
    float* __restrict__ alpha_src,
    float* __restrict__ alpha_dst,
    int64_t N,
    int64_t H,
    int64_t C) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  constexpr int warps_per_block = 8;
  const int64_t nh = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (nh >= N * H) return;
  const int64_t n = nh / H;
  const int64_t h = nh - n * H;
  const int64_t xbase = (n * H + h) * C;
  const int64_t abase = h * C;

  float ssrc = 0.0f;
  float sdst = 0.0f;
  for (int64_t c = lane; c < C; c += 32) {
    const float xv = x[xbase + c];
    ssrc += xv * att_src[abase + c];
    sdst += xv * att_dst[abase + c];
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    ssrc += __shfl_down_sync(0xffffffff, ssrc, offset);
    sdst += __shfl_down_sync(0xffffffff, sdst, offset);
  }
  if (lane == 0) {
    alpha_src[nh] = ssrc;
    alpha_dst[nh] = sdst;
  }
}

// A3-v3: grad_x uses the same warp-per-(node,head) layout as forward.
// The previous flat kernel performed integer div/mod for every element.
// Here one warp computes all channels for one (n,h), with coalesced writes.
__global__ void attention_backward_x_warp_kernel(
    const float* __restrict__ grad_src,
    const float* __restrict__ grad_dst,
    const float* __restrict__ att_src,
    const float* __restrict__ att_dst,
    float* __restrict__ grad_x,
    int64_t NH,
    int64_t H,
    int64_t C) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  constexpr int warps_per_block = 8;
  const int64_t nh = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (nh >= NH) return;
  const int64_t h = nh % H;
  const float gs = grad_src[nh];
  const float gd = grad_dst[nh];
  const int64_t xbase = nh * C;
  const int64_t abase = h * C;
  for (int64_t c = lane; c < C; c += 32) {
    grad_x[xbase + c] = gs * att_src[abase + c] + gd * att_dst[abase + c];
  }
}

// A3-v3 two-stage attention-parameter reduction.
//
// A3-v2 used one block for each (head,channel) and walked nodes with a stride
// of H*C. Threads in a warp therefore read x locations ~6 KB apart for the
// Amazon configuration (H=6,C=256), which profiler data showed was very slow.
//
// Stage 1 instead uses one block for each (node-chunk, head). Threads map to
// channels, so for each node the block reads a contiguous x[h,:] row. The
// per-node grad_src/grad_dst scalars are cached in shared memory. Each block
// writes one float2 partial (src,dst) per channel.
constexpr int ATT_THREADS = 256;
constexpr int ATT_CHUNK_N = 256;

__global__ void attention_backward_att_stage1_kernel(
    const float* __restrict__ grad_src,
    const float* __restrict__ grad_dst,
    const float* __restrict__ x,
    float2* __restrict__ partial,
    int64_t N,
    int64_t H,
    int64_t C,
    int64_t num_chunks) {
  const int64_t block_linear = blockIdx.x;
  const int64_t h = block_linear % H;
  const int64_t chunk = block_linear / H;
  if (chunk >= num_chunks) return;

  const int64_t n0 = chunk * ATT_CHUNK_N;
  const int64_t remaining = N - n0;
  const int count = static_cast<int>(remaining < ATT_CHUNK_N ? remaining : ATT_CHUNK_N);

  __shared__ float s_grad_src[ATT_CHUNK_N];
  __shared__ float s_grad_dst[ATT_CHUNK_N];
  const int tid = threadIdx.x;
  if (tid < count) {
    const int64_t gh = (n0 + tid) * H + h;
    s_grad_src[tid] = grad_src[gh];
    s_grad_dst[tid] = grad_dst[gh];
  }
  __syncthreads();

  for (int64_t c = tid; c < C; c += blockDim.x) {
    float ss = 0.0f;
    float sd = 0.0f;
    for (int local_n = 0; local_n < count; ++local_n) {
      const int64_t n = n0 + local_n;
      const float xv = x[(n * H + h) * C + c];
      ss = fmaf(s_grad_src[local_n], xv, ss);
      sd = fmaf(s_grad_dst[local_n], xv, sd);
    }
    const int64_t pidx = (chunk * H + h) * C + c;
    partial[pidx] = make_float2(ss, sd);
  }
}

// Stage 2 is small: one block per head, one thread (or a thread-stride) per
// channel. It reduces only num_chunks partials (96 on Amazon-ratings).
__global__ void attention_backward_att_stage2_kernel(
    const float2* __restrict__ partial,
    float* __restrict__ grad_att_src,
    float* __restrict__ grad_att_dst,
    int64_t H,
    int64_t C,
    int64_t num_chunks) {
  const int64_t h = blockIdx.x;
  if (h >= H) return;
  for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
    float ss = 0.0f;
    float sd = 0.0f;
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      const float2 v = partial[(chunk * H + h) * C + c];
      ss += v.x;
      sd += v.y;
    }
    const int64_t hc = h * C + c;
    grad_att_src[hc] = ss;
    grad_att_dst[hc] = sd;
  }
}

// Fuses concat=False head mean, bias addition and the immediately following
// SiLU from GraphCliffordBlock. Retained from A3-v2.
__global__ void head_silu_forward_kernel(
    const float* __restrict__ out,
    const float* __restrict__ bias,
    float* __restrict__ y,
    float* __restrict__ pre,
    int64_t N,
    int64_t H,
    int64_t C) {
  const int64_t total = N * C;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = idx % C;
    const int64_t n = idx / C;
    float s = 0.0f;
    const int64_t base = n * H * C + c;
    #pragma unroll 4
    for (int64_t h = 0; h < H; ++h) {
      s += out[base + h * C];
    }
    const float p = s / static_cast<float>(H) + bias[c];
    pre[idx] = p;
    y[idx] = silu_f(p);
  }
}

__global__ void head_silu_backward_out_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ pre,
    float* __restrict__ grad_out,
    int64_t N,
    int64_t H,
    int64_t C) {
  const int64_t total = N * C;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = idx % C;
    const int64_t n = idx / C;
    const float gp = grad_y[idx] * silu_grad_from_pre(pre[idx]);
    const float g = gp / static_cast<float>(H);
    const int64_t base = n * H * C + c;
    for (int64_t h = 0; h < H; ++h) {
      grad_out[base + h * C] = g;
    }
  }
}

__global__ void head_silu_backward_bias_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ pre,
    float* __restrict__ grad_bias,
    int64_t N,
    int64_t C) {
  const int64_t c = blockIdx.x;
  float sum = 0.0f;
  for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
    const int64_t idx = n * C + c;
    sum += grad_y[idx] * silu_grad_from_pre(pre[idx]);
  }
  __shared__ float buf[256];
  buf[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) grad_bias[c] = buf[0];
}

} // namespace

std::vector<torch::Tensor> attention_forward_cuda(
    torch::Tensor x,
    torch::Tensor att_src,
    torch::Tensor att_dst) {
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t N = x.size(0), H = x.size(1), C = x.size(2);
  auto alpha_src = torch::empty({N, H}, x.options());
  auto alpha_dst = torch::empty({N, H}, x.options());
  constexpr int threads = 256;
  constexpr int warps_per_block = 8;
  const int blocks = static_cast<int>((N * H + warps_per_block - 1) / warps_per_block);
  auto stream = at::cuda::getCurrentCUDAStream();
  attention_forward_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), att_src.data_ptr<float>(), att_dst.data_ptr<float>(),
      alpha_src.data_ptr<float>(), alpha_dst.data_ptr<float>(), N, H, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {alpha_src, alpha_dst};
}

std::vector<torch::Tensor> attention_backward_cuda(
    torch::Tensor grad_src,
    torch::Tensor grad_dst,
    torch::Tensor x,
    torch::Tensor att_src,
    torch::Tensor att_dst) {
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t N = x.size(0), H = x.size(1), C = x.size(2);
  auto grad_x = torch::empty_like(x);
  auto grad_att_src = torch::empty_like(att_src);
  auto grad_att_dst = torch::empty_like(att_dst);
  const int64_t num_chunks = (N + ATT_CHUNK_N - 1) / ATT_CHUNK_N;
  // [chunk, head, channel, {src,dst}] -> float2 view in CUDA.
  auto partial = torch::empty({num_chunks, H, C, 2}, x.options());

  constexpr int threads = ATT_THREADS;
  constexpr int warps_per_block = 8;
  const int blocks_x = static_cast<int>((N * H + warps_per_block - 1) / warps_per_block);
  auto stream = at::cuda::getCurrentCUDAStream();

  attention_backward_x_warp_kernel<<<blocks_x, threads, 0, stream>>>(
      grad_src.data_ptr<float>(), grad_dst.data_ptr<float>(),
      att_src.data_ptr<float>(), att_dst.data_ptr<float>(), grad_x.data_ptr<float>(),
      N * H, H, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  attention_backward_att_stage1_kernel<<<static_cast<int>(num_chunks * H), threads, 0, stream>>>(
      grad_src.data_ptr<float>(), grad_dst.data_ptr<float>(), x.data_ptr<float>(),
      reinterpret_cast<float2*>(partial.data_ptr<float>()), N, H, C, num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  attention_backward_att_stage2_kernel<<<static_cast<int>(H), threads, 0, stream>>>(
      reinterpret_cast<const float2*>(partial.data_ptr<float>()),
      grad_att_src.data_ptr<float>(), grad_att_dst.data_ptr<float>(), H, C, num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_att_src, grad_att_dst};
}

std::vector<torch::Tensor> head_silu_forward_cuda(
    torch::Tensor out,
    torch::Tensor bias) {
  c10::cuda::CUDAGuard guard(out.device());
  const int64_t N = out.size(0), H = out.size(1), C = out.size(2);
  auto y = torch::empty({N, C}, out.options());
  auto pre = torch::empty({N, C}, out.options());
  constexpr int threads = 256;
  const int64_t total = N * C;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  head_silu_forward_kernel<<<blocks, threads, 0, stream>>>(
      out.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), pre.data_ptr<float>(), N, H, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, pre};
}

std::vector<torch::Tensor> head_silu_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor pre,
    int64_t heads) {
  c10::cuda::CUDAGuard guard(grad_y.device());
  const int64_t N = grad_y.size(0), C = grad_y.size(1), H = heads;
  auto grad_out = torch::empty({N, H, C}, grad_y.options());
  auto grad_bias = torch::empty({C}, grad_y.options());
  constexpr int threads = 256;
  const int64_t total = N * C;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  head_silu_backward_out_kernel<<<blocks, threads, 0, stream>>>(
      grad_y.data_ptr<float>(), pre.data_ptr<float>(), grad_out.data_ptr<float>(), N, H, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  head_silu_backward_bias_kernel<<<static_cast<int>(C), threads, 0, stream>>>(
      grad_y.data_ptr<float>(), pre.data_ptr<float>(), grad_bias.data_ptr<float>(), N, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_out, grad_bias};
}
