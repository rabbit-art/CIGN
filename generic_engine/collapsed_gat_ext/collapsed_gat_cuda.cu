#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ float leaky_relu_f(float x, float slope) {
  return x > 0.0f ? x : x * slope;
}

__device__ __forceinline__ float silu_f(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return x * s;
}

__device__ __forceinline__ float silu_grad_from_pre(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__global__ void make_dropout_mask_kernel(
    uint8_t* __restrict__ mask,
    int64_t total,
    float p,
    uint64_t seed) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const uint64_t r = splitmix64(seed ^ (static_cast<uint64_t>(idx) * 0xD2B74407B1CE6E93ull));
    const uint32_t hi = static_cast<uint32_t>(r >> 32);
    const float u = (static_cast<float>(hi) + 0.5f) * 2.3283064365386963e-10f;
    mask[idx] = static_cast<uint8_t>(u > p);
  }
}

// A3-v6 core forward, specialized for concat=False, H=6, C=256.
// One 256-thread block owns one destination row. Heads are processed inside
// the same block, so the [N,H,C] message-passing output is never materialized.
__global__ void collapsed_forward_kernel(
    int m,
    int h,
    int f,
    float attn_drop,
    const float* __restrict__ attn_row,
    const float* __restrict__ attn_col,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ in_feat,
    const float* __restrict__ bias,
    float negative_slope,
    float* __restrict__ edge_max,
    float* __restrict__ edge_sum,
    const uint8_t* __restrict__ edge_mask,
    float* __restrict__ pre,
    float* __restrict__ out) {
  const int rid = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int lb = row_ptr[rid];
  const int hb = row_ptr[rid + 1];
  const int degree = hb - lb;
  const int loops = (degree + 31) >> 5;

  __shared__ float w_sh[32];
  __shared__ int cid_sh[32];
  __shared__ float max_sh;
  __shared__ float sum_sh;

  float acc_heads = 0.0f;

  for (int hid = 0; hid < h; ++hid) {
    if (tid < 32) {
      const float row_val = attn_row[rid * h + hid];
      float weight_max = -1.0e38f;
      for (int j = 0; j < loops; ++j) {
        const int pid = lb + (j << 5) + lane;
        float w = -1.0e38f;
        if (pid < hb) {
          const int cid = col_ind[pid];
          w = leaky_relu_f(row_val + attn_col[cid * h + hid], negative_slope);
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
          const float other = __shfl_xor_sync(0xffffffff, w, stride, 32);
          w = fmaxf(w, other);
        }
        weight_max = fmaxf(weight_max, w);
      }
      if (lane == 0) {
        max_sh = weight_max;
        edge_max[rid * h + hid] = weight_max;
      }
    }
    __syncthreads();

    if (tid < 32) {
      const float row_val = attn_row[rid * h + hid];
      float exp_all = 0.0f;
      for (int j = 0; j < loops; ++j) {
        const int pid = lb + (j << 5) + lane;
        float e = 0.0f;
        if (pid < hb) {
          const int cid = col_ind[pid];
          const float w = leaky_relu_f(row_val + attn_col[cid * h + hid], negative_slope);
          e = expf(w - max_sh);
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
          e += __shfl_xor_sync(0xffffffff, e, stride, 32);
        }
        exp_all += e;
      }
      if (lane == 0) {
        sum_sh = exp_all;
        edge_sum[rid * h + hid] = exp_all;
      }
    }
    __syncthreads();

    for (int j = 0; j < loops; ++j) {
      if (tid < 32) {
        const int pid = lb + (j << 5) + lane;
        float w = 0.0f;
        int cid = 0;
        if (pid < hb) {
          cid = col_ind[pid];
          const bool keep = (attn_drop <= 0.0f) || (edge_mask[pid * h + hid] != 0);
          if (keep && sum_sh > 0.0f) {
            const float raw = leaky_relu_f(
                attn_row[rid * h + hid] + attn_col[cid * h + hid], negative_slope);
            w = expf(raw - max_sh) / sum_sh;
            if (attn_drop > 0.0f) w /= (1.0f - attn_drop);
          }
        }
        w_sh[lane] = w;
        cid_sh[lane] = cid;
      }
      __syncthreads();

      if (tid < f) {
        const int rem = hb - (lb + (j << 5));
        const int valid = rem < 32 ? rem : 32;
        float local = 0.0f;
        #pragma unroll 4
        for (int kk = 0; kk < valid; ++kk) {
          const int cid = cid_sh[kk];
          local = fmaf(w_sh[kk], in_feat[(cid * h + hid) * f + tid], local);
        }
        acc_heads += local;
      }
      __syncthreads();
    }
  }

  if (tid < f) {
    const int idx = rid * f + tid;
    const float p = acc_heads / static_cast<float>(h) + bias[tid];
    pre[idx] = p;
    out[idx] = silu_f(p);
  }
}

__global__ void grad_pre_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ pre,
    float* __restrict__ grad_pre,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    grad_pre[idx] = grad_y[idx] * silu_grad_from_pre(pre[idx]);
  }
}

__global__ void grad_bias_kernel(
    const float* __restrict__ grad_pre,
    float* __restrict__ grad_bias,
    int m,
    int f) {
  const int fid = blockIdx.x;
  float sum = 0.0f;
  for (int rid = threadIdx.x; rid < m; rid += blockDim.x) {
    sum += grad_pre[rid * f + fid];
  }
  __shared__ float buf[256];
  buf[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) grad_bias[fid] = buf[0];
}

// Equivalent to dgNN mhspmm_backward_kernel, except grad_pre is [N,C].
// The concat=False head-mean derivative 1/H is folded into this kernel, so the
// old [N,H,C] expanded grad-output tensor is never created.
__global__ void mhspmm_backward_collapsed_kernel(
    int m,
    int h,
    int f,
    float negative_slope,
    float attn_drop,
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_ind,
    const int* __restrict__ permute,
    const float* __restrict__ edge_max,
    const float* __restrict__ edge_sum,
    const uint8_t* __restrict__ edge_mask,
    const float* __restrict__ attn_row,
    const float* __restrict__ attn_col,
    const float* __restrict__ grad_pre,
    float* __restrict__ grad_feat) {
  const int cid = blockIdx.x;
  const int hid = blockIdx.y;
  const int lane = threadIdx.x;
  const int fy = threadIdx.y;
  const int fid = fy * 32 + lane;
  const int lb = col_ptr[cid];
  const int hb = col_ptr[cid + 1];
  const int loops = (hb - lb + 31) >> 5;

  __shared__ float w_sh[32];
  __shared__ int rid_sh[32];
  float acc = 0.0f;

  for (int j = 0; j < loops; ++j) {
    if (fy == 0) {
      const int pid = lb + (j << 5) + lane;
      float w = 0.0f;
      int rid = 0;
      if (pid < hb) {
        rid = row_ind[pid];
        const int csr_e = permute[pid];
        const bool keep = (attn_drop <= 0.0f) || (edge_mask[csr_e * h + hid] != 0);
        if (keep) {
          const float raw = leaky_relu_f(
              attn_row[rid * h + hid] + attn_col[cid * h + hid], negative_slope);
          const float denom = edge_sum[rid * h + hid];
          if (denom > 0.0f) {
            w = expf(raw - edge_max[rid * h + hid]) / denom;
            if (attn_drop > 0.0f) w /= (1.0f - attn_drop);
          }
        }
      }
      w_sh[lane] = w;
      rid_sh[lane] = rid;
    }
    __syncthreads();

    if (fid < f) {
      const int rem = hb - (lb + (j << 5));
        const int valid = rem < 32 ? rem : 32;
      for (int kk = 0; kk < valid; ++kk) {
        acc = fmaf(w_sh[kk], grad_pre[rid_sh[kk] * f + fid] / static_cast<float>(h), acc);
      }
    }
    __syncthreads();
  }

  if (fid < f) {
    grad_feat[(cid * h + hid) * f + fid] = acc;
  }
}

__device__ __forceinline__ int find_row_binary(
    const int* __restrict__ row_ptr,
    int m,
    int edge) {
  int lo = 0;
  int hi = m;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (row_ptr[mid] <= edge) lo = mid;
    else hi = mid;
  }
  return lo;
}

// Same SDDMM as dgNN backward, but the head-mean grad is read directly from
// [N,C] and scaled by 1/H. Each block processes 16 CSR edges per head.
__global__ void mhsddmm_collapsed_kernel(
    int m,
    int nnz,
    int h,
    int f,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ grad_pre,
    const float* __restrict__ feature,
    float* __restrict__ grad_edge) {
  const int lane = threadIdx.x;
  const int group = threadIdx.y; // 0..3, four edges per warp
  const int hid = blockIdx.y;
  const int base = blockIdx.x * 16 + group * 4;

  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int rows[4] = {0, 0, 0, 0};
  int cols[4] = {0, 0, 0, 0};
  bool valid[4] = {false, false, false, false};

  if (lane == 0) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int e = base + k;
      if (e < nnz) {
        valid[k] = true;
        rows[k] = find_row_binary(row_ptr, m, e);
        cols[k] = col_ind[e];
      }
    }
  }
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    rows[k] = __shfl_sync(0xffffffff, rows[k], 0);
    cols[k] = __shfl_sync(0xffffffff, cols[k], 0);
    const int v = __shfl_sync(0xffffffff, valid[k] ? 1 : 0, 0);
    valid[k] = (v != 0);
  }

  const float inv_h = 1.0f / static_cast<float>(h);
  for (int c = lane; c < f; c += 32) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (valid[k]) {
        const float g = grad_pre[rows[k] * f + c] * inv_h;
        const float x = feature[(cols[k] * h + hid) * f + c];
        sums[k] = fmaf(g, x, sums[k]);
      }
    }
  }
  #pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      sums[k] += __shfl_xor_sync(0xffffffff, sums[k], stride, 32);
    }
  }
  if (lane == 0) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int e = base + k;
      if (e < nnz) grad_edge[e * h + hid] = sums[k];
    }
  }
}

// Adapted from dgNN's fused_backward_kernel. edge_mask is uint8 here.
__global__ void attention_backward_kernel(
    int m,
    int h,
    float attn_drop,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    float negative_slope,
    const float* __restrict__ edge_max,
    const float* __restrict__ edge_sum,
    const uint8_t* __restrict__ edge_mask,
    const float* __restrict__ attn_row,
    const float* __restrict__ attn_col,
    const float* __restrict__ grad_edge,
    float* __restrict__ grad_attn_row,
    float* __restrict__ grad_attn_col) {
  const int rid = blockIdx.x;
  const int hid = threadIdx.y;
  const int lane = threadIdx.x;
  const int lb = row_ptr[rid];
  const int hb = row_ptr[rid + 1];
  const int loops = (hb - lb + 31) >> 5;
  const float row_val = attn_row[rid * h + hid];

  float weight_sum = 0.0f;
  for (int j = 0; j < loops; ++j) {
    const int pid = lb + (j << 5) + lane;
    float term = 0.0f;
    if (pid < hb) {
      const int cid = col_ind[pid];
      const float val = leaky_relu_f(row_val + attn_col[cid * h + hid], negative_slope);
      const float denom = edge_sum[rid * h + hid];
      const float soft = denom > 0.0f ? expf(val - edge_max[rid * h + hid]) / denom : 0.0f;
      const bool keep = (attn_drop <= 0.0f) || (edge_mask[pid * h + hid] != 0);
      float ge = keep ? grad_edge[pid * h + hid] : 0.0f;
      if (attn_drop > 0.0f) ge /= (1.0f - attn_drop);
      term = soft * ge;
    }
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      term += __shfl_xor_sync(0xffffffff, term, stride, 32);
    }
    weight_sum += term;
  }

  float grad_row = 0.0f;
  for (int j = 0; j < loops; ++j) {
    const int pid = lb + (j << 5) + lane;
    float gout = 0.0f;
    if (pid < hb) {
      const int cid = col_ind[pid];
      const float raw0 = row_val + attn_col[cid * h + hid];
      const float val = leaky_relu_f(raw0, negative_slope);
      const float denom = edge_sum[rid * h + hid];
      const float soft = denom > 0.0f ? expf(val - edge_max[rid * h + hid]) / denom : 0.0f;
      const bool keep = (attn_drop <= 0.0f) || (edge_mask[pid * h + hid] != 0);
      float ge = keep ? grad_edge[pid * h + hid] : 0.0f;
      if (attn_drop > 0.0f) ge /= (1.0f - attn_drop);
      gout = soft * (ge - weight_sum);
      if (raw0 < 0.0f) gout *= negative_slope;
      atomicAdd(&grad_attn_col[cid * h + hid], gout);
    }
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      gout += __shfl_xor_sync(0xffffffff, gout, stride, 32);
    }
    grad_row += gout;
  }
  if (lane == 0) grad_attn_row[rid * h + hid] = grad_row;
}

} // namespace

std::vector<torch::Tensor> collapsed_gat_forward_cuda(
    torch::Tensor attn_row,
    torch::Tensor attn_col,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    double negative_slope,
    torch::Tensor in_feat,
    torch::Tensor bias,
    double attn_drop,
    int64_t seed) {
  c10::cuda::CUDAGuard guard(in_feat.device());
  const int m = static_cast<int>(row_ptr.size(0) - 1);
  const int nnz = static_cast<int>(col_ind.size(0));
  const int h = static_cast<int>(in_feat.size(1));
  const int f = static_cast<int>(in_feat.size(2));
  auto float_opts = in_feat.options();
  auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(in_feat.device());
  auto out = torch::empty({m, f}, float_opts);
  auto pre = torch::empty({m, f}, float_opts);
  auto edge_max = torch::empty({m, h}, float_opts);
  auto edge_sum = torch::empty({m, h}, float_opts);
  auto edge_mask = torch::empty({nnz, h}, byte_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const float p = static_cast<float>(attn_drop);
  if (p > 0.0f) {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(nnz) * h;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    make_dropout_mask_kernel<<<blocks, threads, 0, stream>>>(
        edge_mask.data_ptr<uint8_t>(), total, p, static_cast<uint64_t>(seed));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  collapsed_forward_kernel<<<m, 256, 0, stream>>>(
      m, h, f, p,
      attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
      row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(),
      in_feat.data_ptr<float>(), bias.data_ptr<float>(),
      static_cast<float>(negative_slope),
      edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(),
      edge_mask.data_ptr<uint8_t>(), pre.data_ptr<float>(), out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, pre, edge_max, edge_sum, edge_mask};
}

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
    torch::Tensor grad_y) {
  c10::cuda::CUDAGuard guard(in_feat.device());
  const int m = static_cast<int>(row_ptr.size(0) - 1);
  const int nnz = static_cast<int>(col_ind.size(0));
  const int h = static_cast<int>(in_feat.size(1));
  const int f = static_cast<int>(in_feat.size(2));
  const float p = static_cast<float>(attn_drop);
  const float slope = static_cast<float>(negative_slope);
  auto opts = in_feat.options();

  auto grad_pre = torch::empty({m, f}, opts);
  auto grad_feat = torch::empty_like(in_feat);
  auto grad_edge = torch::empty({nnz, h}, opts);
  auto grad_attn_row = torch::empty_like(attn_row);
  auto grad_attn_col = torch::zeros_like(attn_col);
  auto grad_bias = torch::empty({f}, opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int64_t total = static_cast<int64_t>(m) * f;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  grad_pre_kernel<<<blocks, threads, 0, stream>>>(
      grad_y.data_ptr<float>(), pre.data_ptr<float>(), grad_pre.data_ptr<float>(), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  grad_bias_kernel<<<f, threads, 0, stream>>>(
      grad_pre.data_ptr<float>(), grad_bias.data_ptr<float>(), m, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  mhspmm_backward_collapsed_kernel<<<dim3(m, h, 1), dim3(32, 8, 1), 0, stream>>>(
      m, h, f, slope, p,
      col_ptr.data_ptr<int>(), row_ind.data_ptr<int>(), permute.data_ptr<int>(),
      edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(), edge_mask.data_ptr<uint8_t>(),
      attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
      grad_pre.data_ptr<float>(), grad_feat.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int edge_blocks = (nnz + 15) / 16;
  mhsddmm_collapsed_kernel<<<dim3(edge_blocks, h, 1), dim3(32, 4, 1), 0, stream>>>(
      m, nnz, h, f, row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(),
      grad_pre.data_ptr<float>(), in_feat.data_ptr<float>(), grad_edge.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  attention_backward_kernel<<<dim3(m, 1, 1), dim3(32, h, 1), 0, stream>>>(
      m, h, p, row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(), slope,
      edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(), edge_mask.data_ptr<uint8_t>(),
      attn_row.data_ptr<float>(), attn_col.data_ptr<float>(), grad_edge.data_ptr<float>(),
      grad_attn_row.data_ptr<float>(), grad_attn_col.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_feat, grad_attn_row, grad_attn_col, grad_bias};
}
