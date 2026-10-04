#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int D = 256;
constexpr int THREADS = 256;
constexpr int PARAM_CHUNK_N = 128;

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__device__ __forceinline__ bool dropout_keep(uint64_t seed, uint64_t idx, float p) {
  if (p <= 0.0f) return true;
  const uint64_t r = splitmix64(seed ^ (idx * 0xD2B74407B1CE6E93ull));
  // 53-bit uniform in [0,1).
  const double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
  return u >= static_cast<double>(p);
}

__device__ __forceinline__ float block_sum_256(float v, float* warp_sums) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
  if (lane==0) warp_sums[warp]=v;
  __syncthreads();
  float out = (threadIdx.x < 8) ? warp_sums[lane] : 0.0f;
  if (warp==0) {
    #pragma unroll
    for (int off=16; off>0; off>>=1) out += __shfl_down_sync(0xffffffff, out, off);
  }
  if (threadIdx.x==0) warp_sums[0]=out;
  __syncthreads();
  return warp_sums[0];
}

__global__ void dropout_ln_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    uint8_t* __restrict__ mask,
    int64_t N, float p, float eps, uint64_t seed) {
  const int64_t n = blockIdx.x;
  const int c = threadIdx.x;
  if (n >= N || c >= D) return;
  const int64_t idx = n * D + c;
  const bool keep = dropout_keep(seed, static_cast<uint64_t>(idx), p);
  mask[idx] = keep ? 1 : 0;
  const float scale = p > 0.0f ? 1.0f / (1.0f-p) : 1.0f;
  const float q = keep ? x[idx] * scale : 0.0f;
  __shared__ float red[8];
  const float s = block_sum_256(q, red);
  const float mu = s / static_cast<float>(D);
  const float centered = q - mu;
  const float varsum = block_sum_256(centered * centered, red);
  const float var = varsum / static_cast<float>(D);
  const float rs = rsqrtf(var + eps);
  if (c==0) { mean[n]=mu; rstd[n]=rs; }
  y[idx] = (q-mu)*rs*w[c] + b[c];
}

__global__ void add_ln_forward_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b0,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int64_t N, float eps) {
  const int64_t n=blockIdx.x; const int c=threadIdx.x;
  if (n>=N || c>=D) return;
  const int64_t idx=n*D+c;
  const float z=a[idx]+b0[idx];
  __shared__ float red[8];
  const float s=block_sum_256(z, red);
  const float mu=s/static_cast<float>(D);
  const float centered=z-mu;
  const float varsum=block_sum_256(centered*centered, red);
  const float var=varsum/static_cast<float>(D);
  const float rs=rsqrtf(var+eps);
  if(c==0){mean[n]=mu;rstd[n]=rs;}
  y[idx]=(z-mu)*rs*w[c]+b[c];
}

__global__ void dropout_ln_dx_kernel(
    const float* __restrict__ gy,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const uint8_t* __restrict__ mask,
    float* __restrict__ dx,
    int64_t N, float p) {
  const int64_t n=blockIdx.x; const int c=threadIdx.x;
  if(n>=N || c>=D) return;
  const int64_t idx=n*D+c;
  const float scale=p>0.0f ? 1.0f/(1.0f-p) : 1.0f;
  const float q=mask[idx] ? x[idx]*scale : 0.0f;
  const float xhat=(q-mean[n])*rstd[n];
  const float dnorm=gy[idx]*w[c];
  __shared__ float red[8];
  const float sum1=block_sum_256(dnorm, red);
  const float sum2=block_sum_256(dnorm*xhat, red);
  const float dq=(rstd[n]/static_cast<float>(D))*(static_cast<float>(D)*dnorm-sum1-xhat*sum2);
  dx[idx]=mask[idx] ? dq*scale : 0.0f;
}

__global__ void add_ln_dz_kernel(
    const float* __restrict__ gy,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ w,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dz,
    int64_t N) {
  const int64_t n=blockIdx.x; const int c=threadIdx.x;
  if(n>=N || c>=D) return;
  const int64_t idx=n*D+c;
  const float z=a[idx]+b[idx];
  const float xhat=(z-mean[n])*rstd[n];
  const float dnorm=gy[idx]*w[c];
  __shared__ float red[8];
  const float sum1=block_sum_256(dnorm, red);
  const float sum2=block_sum_256(dnorm*xhat, red);
  dz[idx]=(rstd[n]/static_cast<float>(D))*(static_cast<float>(D)*dnorm-sum1-xhat*sum2);
}

__global__ void ln_param_stage1_dropout_kernel(
    const float* __restrict__ gy,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const uint8_t* __restrict__ mask,
    float* __restrict__ partial_w,
    float* __restrict__ partial_b,
    int64_t N, int64_t chunks, float p) {
  const int chunk=blockIdx.x; const int c=threadIdx.x;
  if(chunk>=chunks || c>=D) return;
  const int64_t n0=static_cast<int64_t>(chunk)*PARAM_CHUNK_N;
  const int64_t n1=(n0+PARAM_CHUNK_N < N) ? (n0+PARAM_CHUNK_N) : N;
  const float scale=p>0.0f ? 1.0f/(1.0f-p) : 1.0f;
  float sw=0.0f,sb=0.0f;
  for(int64_t n=n0;n<n1;++n){
    const int64_t idx=n*D+c;
    const float q=mask[idx]?x[idx]*scale:0.0f;
    const float xhat=(q-mean[n])*rstd[n];
    const float g=gy[idx];
    sw += g*xhat; sb += g;
  }
  partial_w[static_cast<int64_t>(chunk)*D+c]=sw;
  partial_b[static_cast<int64_t>(chunk)*D+c]=sb;
}

__global__ void ln_param_stage1_add_kernel(
    const float* __restrict__ gy,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ partial_w,
    float* __restrict__ partial_b,
    int64_t N, int64_t chunks) {
  const int chunk=blockIdx.x; const int c=threadIdx.x;
  if(chunk>=chunks || c>=D) return;
  const int64_t n0=static_cast<int64_t>(chunk)*PARAM_CHUNK_N;
  const int64_t n1=(n0+PARAM_CHUNK_N < N) ? (n0+PARAM_CHUNK_N) : N;
  float sw=0.0f,sb=0.0f;
  for(int64_t n=n0;n<n1;++n){
    const int64_t idx=n*D+c;
    const float z=a[idx]+b[idx];
    const float xhat=(z-mean[n])*rstd[n];
    const float g=gy[idx];
    sw += g*xhat; sb += g;
  }
  partial_w[static_cast<int64_t>(chunk)*D+c]=sw;
  partial_b[static_cast<int64_t>(chunk)*D+c]=sb;
}

__global__ void param_stage2_kernel(
    const float* __restrict__ pw,
    const float* __restrict__ pb,
    float* __restrict__ gw,
    float* __restrict__ gb,
    int64_t chunks) {
  const int c=threadIdx.x;
  if(c>=D) return;
  float sw=0.0f,sb=0.0f;
  for(int64_t k=0;k<chunks;++k){
    sw += pw[k*D+c]; sb += pb[k*D+c];
  }
  gw[c]=sw; gb[c]=sb;
}

__global__ void gamma_residual_forward_kernel(
    const float* __restrict__ h,
    const float* __restrict__ g,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    uint8_t* __restrict__ mask,
    int64_t N, float p, uint64_t seed) {
  const int64_t n=blockIdx.x; const int c=threadIdx.x;
  if(n>=N || c>=D) return;
  const int64_t idx=n*D+c;
  const bool keep=dropout_keep(seed,static_cast<uint64_t>(idx),p);
  mask[idx]=keep?1:0;
  const float scale=p>0.0f?1.0f/(1.0f-p):1.0f;
  const float gd=keep?g[idx]*scale:0.0f;
  y[idx]=h[idx]+gamma[c]*gd;
}

__global__ void gamma_backward_stage1_kernel(
    const float* __restrict__ gy,
    const float* __restrict__ g,
    const float* __restrict__ gamma,
    const uint8_t* __restrict__ mask,
    float* __restrict__ grad_g,
    float* __restrict__ partial_gamma,
    int64_t N,int64_t chunks,float p) {
  const int chunk=blockIdx.x; const int c=threadIdx.x;
  if(chunk>=chunks || c>=D) return;
  const int64_t n0=static_cast<int64_t>(chunk)*PARAM_CHUNK_N;
  const int64_t n1=(n0+PARAM_CHUNK_N < N) ? (n0+PARAM_CHUNK_N) : N;
  const float scale=p>0.0f?1.0f/(1.0f-p):1.0f;
  float sg=0.0f;
  for(int64_t n=n0;n<n1;++n){
    const int64_t idx=n*D+c;
    const float m=mask[idx]?scale:0.0f;
    const float go=gy[idx];
    grad_g[idx]=go*gamma[c]*m;
    sg += go*g[idx]*m;
  }
  partial_gamma[static_cast<int64_t>(chunk)*D+c]=sg;
}

__global__ void gamma_stage2_kernel(const float* __restrict__ partial,float* __restrict__ gg,int64_t chunks){
  const int c=threadIdx.x; if(c>=D)return;
  float s=0.0f; for(int64_t k=0;k<chunks;++k)s+=partial[k*D+c]; gg[c]=s;
}

} // namespace

std::vector<torch::Tensor> dropout_layernorm_forward_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    double dropout_p, double eps, int64_t seed) {
  c10::cuda::CUDAGuard guard(x.device());
  auto y=torch::empty_like(x);
  auto mean=torch::empty({x.size(0)},x.options());
  auto rstd=torch::empty({x.size(0)},x.options());
  auto mask=torch::empty(x.sizes(),x.options().dtype(torch::kUInt8));
  auto stream=at::cuda::getCurrentCUDAStream();
  dropout_ln_forward_kernel<<<x.size(0),THREADS,0,stream>>>(
      x.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(),y.data_ptr<float>(),
      mean.data_ptr<float>(),rstd.data_ptr<float>(),mask.data_ptr<uint8_t>(),x.size(0),
      static_cast<float>(dropout_p),static_cast<float>(eps),static_cast<uint64_t>(seed));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,mean,rstd,mask};
}

std::vector<torch::Tensor> dropout_layernorm_backward_cuda(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor weight,
    torch::Tensor mean, torch::Tensor rstd, torch::Tensor mask,
    double dropout_p) {
  c10::cuda::CUDAGuard guard(x.device());
  auto dx=torch::empty_like(x); const int64_t chunks=(x.size(0)+PARAM_CHUNK_N-1)/PARAM_CHUNK_N;
  auto pw=torch::empty({chunks,D},x.options()); auto pb=torch::empty({chunks,D},x.options());
  auto gw=torch::empty({D},x.options()); auto gb=torch::empty({D},x.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  dropout_ln_dx_kernel<<<x.size(0),THREADS,0,stream>>>(grad_y.data_ptr<float>(),x.data_ptr<float>(),weight.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),mask.data_ptr<uint8_t>(),dx.data_ptr<float>(),x.size(0),static_cast<float>(dropout_p));
  ln_param_stage1_dropout_kernel<<<chunks,THREADS,0,stream>>>(grad_y.data_ptr<float>(),x.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),mask.data_ptr<uint8_t>(),pw.data_ptr<float>(),pb.data_ptr<float>(),x.size(0),chunks,static_cast<float>(dropout_p));
  param_stage2_kernel<<<1,THREADS,0,stream>>>(pw.data_ptr<float>(),pb.data_ptr<float>(),gw.data_ptr<float>(),gb.data_ptr<float>(),chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx,gw,gb};
}

std::vector<torch::Tensor> add_layernorm_forward_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight, torch::Tensor bias,
    double eps) {
  c10::cuda::CUDAGuard guard(a.device()); auto y=torch::empty_like(a); auto mean=torch::empty({a.size(0)},a.options()); auto rstd=torch::empty({a.size(0)},a.options()); auto stream=at::cuda::getCurrentCUDAStream();
  add_ln_forward_kernel<<<a.size(0),THREADS,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(),y.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),a.size(0),static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {y,mean,rstd};
}

std::vector<torch::Tensor> add_layernorm_backward_cuda(
    torch::Tensor grad_y, torch::Tensor a, torch::Tensor b,
    torch::Tensor weight, torch::Tensor mean, torch::Tensor rstd) {
  c10::cuda::CUDAGuard guard(a.device()); auto dz=torch::empty_like(a); const int64_t chunks=(a.size(0)+PARAM_CHUNK_N-1)/PARAM_CHUNK_N;
  auto pw=torch::empty({chunks,D},a.options()); auto pb=torch::empty({chunks,D},a.options()); auto gw=torch::empty({D},a.options()); auto gb=torch::empty({D},a.options()); auto stream=at::cuda::getCurrentCUDAStream();
  add_ln_dz_kernel<<<a.size(0),THREADS,0,stream>>>(grad_y.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),weight.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),dz.data_ptr<float>(),a.size(0));
  ln_param_stage1_add_kernel<<<chunks,THREADS,0,stream>>>(grad_y.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),pw.data_ptr<float>(),pb.data_ptr<float>(),a.size(0),chunks);
  param_stage2_kernel<<<1,THREADS,0,stream>>>(pw.data_ptr<float>(),pb.data_ptr<float>(),gw.data_ptr<float>(),gb.data_ptr<float>(),chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {dz,gw,gb};
}

std::vector<torch::Tensor> dropout_gamma_residual_forward_cuda(
    torch::Tensor h, torch::Tensor g, torch::Tensor gamma,
    double dropout_p, int64_t seed) {
  c10::cuda::CUDAGuard guard(h.device()); auto y=torch::empty_like(h); auto mask=torch::empty(h.sizes(),h.options().dtype(torch::kUInt8)); auto stream=at::cuda::getCurrentCUDAStream();
  gamma_residual_forward_kernel<<<h.size(0),THREADS,0,stream>>>(h.data_ptr<float>(),g.data_ptr<float>(),gamma.data_ptr<float>(),y.data_ptr<float>(),mask.data_ptr<uint8_t>(),h.size(0),static_cast<float>(dropout_p),static_cast<uint64_t>(seed));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {y,mask};
}

std::vector<torch::Tensor> dropout_gamma_residual_backward_cuda(
    torch::Tensor grad_y, torch::Tensor g, torch::Tensor gamma,
    torch::Tensor mask, double dropout_p) {
  c10::cuda::CUDAGuard guard(g.device()); const int64_t chunks=(g.size(0)+PARAM_CHUNK_N-1)/PARAM_CHUNK_N; auto grad_g=torch::empty_like(g); auto partial=torch::empty({chunks,D},g.options()); auto grad_gamma=torch::empty({D},g.options()); auto stream=at::cuda::getCurrentCUDAStream();
  gamma_backward_stage1_kernel<<<chunks,THREADS,0,stream>>>(grad_y.data_ptr<float>(),g.data_ptr<float>(),gamma.data_ptr<float>(),mask.data_ptr<uint8_t>(),grad_g.data_ptr<float>(),partial.data_ptr<float>(),g.size(0),chunks,static_cast<float>(dropout_p));
  gamma_stage2_kernel<<<1,THREADS,0,stream>>>(partial.data_ptr<float>(),grad_gamma.data_ptr<float>(),chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {grad_g,grad_gamma};
}
