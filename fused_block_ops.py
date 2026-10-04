from __future__ import annotations

import itertools
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load, load_inline

_EXT=None
_EXT_SPECIAL=None
_EXT_ERROR=None
_CALL_COUNTER=itertools.count()
_CPP_SOURCE='#include <torch/extension.h>\n#include <vector>\n\nstd::vector<torch::Tensor> dropout_layernorm_forward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,double,double,int64_t);\nstd::vector<torch::Tensor> dropout_layernorm_backward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double);\nstd::vector<torch::Tensor> add_layernorm_forward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double);\nstd::vector<torch::Tensor> add_layernorm_backward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\nstd::vector<torch::Tensor> dropout_gamma_residual_forward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,double,int64_t);\nstd::vector<torch::Tensor> dropout_gamma_residual_backward_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double);\n\n#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")\n#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")\n#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")\n#define CHECK_U8(x) TORCH_CHECK(x.scalar_type() == at::kByte, #x " must be uint8")\n#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)\n\nstatic int64_t check_2d(const torch::Tensor& x,const char* name){CHECK_INPUT(x);TORCH_CHECK(x.dim()==2,name," must be [N,D]");TORCH_CHECK(x.size(1)>=1 && x.size(1)<=256,name," requires 1<=D<=256");return x.size(1);}\nstatic void check_param(const torch::Tensor& x,const char* name,int64_t D){CHECK_INPUT(x);TORCH_CHECK(x.dim()==1 && x.numel()==D,name," must have D elements");}\n\nstd::vector<torch::Tensor> dropout_layernorm_forward(torch::Tensor x,torch::Tensor w,torch::Tensor b,double p,double eps,int64_t seed){auto D=check_2d(x,"x");check_param(w,"weight",D);check_param(b,"bias",D);TORCH_CHECK(p>=0.0 && p<1.0,"dropout_p must be in [0,1)");return dropout_layernorm_forward_cuda(x,w,b,p,eps,seed);}\nstd::vector<torch::Tensor> dropout_layernorm_backward(torch::Tensor gy,torch::Tensor x,torch::Tensor w,torch::Tensor mean,torch::Tensor rstd,torch::Tensor mask,double p){auto D=check_2d(x,"x");TORCH_CHECK(check_2d(gy,"grad_y")==D,"grad_y D mismatch");check_param(w,"weight",D);CHECK_INPUT(mean);CHECK_INPUT(rstd);CHECK_CUDA(mask);CHECK_CONTIGUOUS(mask);CHECK_U8(mask);TORCH_CHECK(gy.sizes()==x.sizes(),"grad_y/x shape mismatch");TORCH_CHECK(mean.dim()==1 && mean.size(0)==x.size(0),"mean must be [N]");TORCH_CHECK(rstd.sizes()==mean.sizes(),"rstd must be [N]");TORCH_CHECK(mask.sizes()==x.sizes(),"mask shape mismatch");return dropout_layernorm_backward_cuda(gy,x,w,mean,rstd,mask,p);}\nstd::vector<torch::Tensor> add_layernorm_forward(torch::Tensor a,torch::Tensor b,torch::Tensor w,torch::Tensor bias,double eps){auto D=check_2d(a,"a");TORCH_CHECK(check_2d(b,"b")==D,"b D mismatch");check_param(w,"weight",D);check_param(bias,"bias",D);TORCH_CHECK(a.sizes()==b.sizes(),"a/b shape mismatch");return add_layernorm_forward_cuda(a,b,w,bias,eps);}\nstd::vector<torch::Tensor> add_layernorm_backward(torch::Tensor gy,torch::Tensor a,torch::Tensor b,torch::Tensor w,torch::Tensor mean,torch::Tensor rstd){auto D=check_2d(a,"a");TORCH_CHECK(check_2d(b,"b")==D && check_2d(gy,"grad_y")==D,"D mismatch");check_param(w,"weight",D);CHECK_INPUT(mean);CHECK_INPUT(rstd);TORCH_CHECK(a.sizes()==b.sizes() && a.sizes()==gy.sizes(),"shape mismatch");TORCH_CHECK(mean.dim()==1 && mean.size(0)==a.size(0),"mean must be [N]");TORCH_CHECK(rstd.sizes()==mean.sizes(),"rstd must be [N]");return add_layernorm_backward_cuda(gy,a,b,w,mean,rstd);}\nstd::vector<torch::Tensor> dropout_gamma_residual_forward(torch::Tensor h,torch::Tensor g,torch::Tensor gamma,double p,int64_t seed){auto D=check_2d(h,"h");TORCH_CHECK(check_2d(g,"g")==D,"g D mismatch");check_param(gamma,"gamma",D);TORCH_CHECK(h.sizes()==g.sizes(),"h/g shape mismatch");TORCH_CHECK(p>=0.0 && p<1.0,"dropout_p must be in [0,1)");return dropout_gamma_residual_forward_cuda(h,g,gamma,p,seed);}\nstd::vector<torch::Tensor> dropout_gamma_residual_backward(torch::Tensor gy,torch::Tensor g,torch::Tensor gamma,torch::Tensor mask,double p){auto D=check_2d(g,"g");TORCH_CHECK(check_2d(gy,"grad_y")==D,"grad_y D mismatch");check_param(gamma,"gamma",D);CHECK_CUDA(mask);CHECK_CONTIGUOUS(mask);CHECK_U8(mask);TORCH_CHECK(gy.sizes()==g.sizes() && mask.sizes()==g.sizes(),"shape mismatch");return dropout_gamma_residual_backward_cuda(gy,g,gamma,mask,p);}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("dropout_layernorm_forward",&dropout_layernorm_forward);m.def("dropout_layernorm_backward",&dropout_layernorm_backward);m.def("add_layernorm_forward",&add_layernorm_forward);m.def("add_layernorm_backward",&add_layernorm_backward);m.def("dropout_gamma_residual_forward",&dropout_gamma_residual_forward);m.def("dropout_gamma_residual_backward",&dropout_gamma_residual_backward);}\n'
_CUDA_SOURCE='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <vector>\n\nnamespace {\nconstexpr int PARAM_CHUNK_N=128;\n__device__ __forceinline__ uint64_t splitmix64(uint64_t x){x+=0x9E3779B97F4A7C15ull;x=(x^(x>>30))*0xBF58476D1CE4E5B9ull;x=(x^(x>>27))*0x94D049BB133111EBull;return x^(x>>31);}\n__device__ __forceinline__ bool dropout_keep(uint64_t seed,uint64_t idx,float p){if(p<=0.0f)return true;const uint64_t r=splitmix64(seed^(idx*0xD2B74407B1CE6E93ull));const double u=static_cast<double>(r>>11)*(1.0/9007199254740992.0);return u>=static_cast<double>(p);}\n__device__ __forceinline__ float block_sum(float v,float* warp_sums){const int lane=threadIdx.x&31;const int warp=threadIdx.x>>5;for(int off=16;off>0;off>>=1)v+=__shfl_down_sync(0xffffffff,v,off);if(lane==0)warp_sums[warp]=v;__syncthreads();const int nw=blockDim.x>>5;float out=(warp==0 && lane<nw)?warp_sums[lane]:0.0f;if(warp==0){for(int off=16;off>0;off>>=1)out+=__shfl_down_sync(0xffffffff,out,off);}if(threadIdx.x==0)warp_sums[0]=out;__syncthreads();return warp_sums[0];}\n__host__ inline int launch_threads(int D){return D<=32?32:D<=64?64:D<=128?128:256;}\n\n__global__ void dropout_ln_forward_kernel(const float* x,const float* w,const float* b,float* y,float* mean,float* rstd,uint8_t* mask,int64_t N,int D,float p,float eps,uint64_t seed){const int64_t n=blockIdx.x;const int c=threadIdx.x;if(n>=N)return;const bool valid=c<D;float q=0.0f;int64_t idx=0;if(valid){idx=n*static_cast<int64_t>(D)+c;const bool keep=dropout_keep(seed,static_cast<uint64_t>(idx),p);mask[idx]=keep?1:0;const float scale=p>0.0f?1.0f/(1.0f-p):1.0f;q=keep?x[idx]*scale:0.0f;}__shared__ float red[8];const float s=block_sum(q,red);const float mu=s/static_cast<float>(D);const float centered=valid?(q-mu):0.0f;const float varsum=block_sum(centered*centered,red);const float rs=rsqrtf(varsum/static_cast<float>(D)+eps);if(c==0){mean[n]=mu;rstd[n]=rs;}if(valid)y[idx]=(q-mu)*rs*w[c]+b[c];}\n__global__ void add_ln_forward_kernel(const float* a,const float* b0,const float* w,const float* b,float* y,float* mean,float* rstd,int64_t N,int D,float eps){const int64_t n=blockIdx.x;const int c=threadIdx.x;if(n>=N)return;const bool valid=c<D;float z=0.0f;int64_t idx=0;if(valid){idx=n*static_cast<int64_t>(D)+c;z=a[idx]+b0[idx];}__shared__ float red[8];const float s=block_sum(z,red);const float mu=s/static_cast<float>(D);const float centered=valid?(z-mu):0.0f;const float varsum=block_sum(centered*centered,red);const float rs=rsqrtf(varsum/static_cast<float>(D)+eps);if(c==0){mean[n]=mu;rstd[n]=rs;}if(valid)y[idx]=(z-mu)*rs*w[c]+b[c];}\n__global__ void dropout_ln_dx_kernel(const float* gy,const float* x,const float* w,const float* mean,const float* rstd,const uint8_t* mask,float* dx,int64_t N,int D,float p){const int64_t n=blockIdx.x;const int c=threadIdx.x;if(n>=N)return;const bool valid=c<D;float xhat=0.0f,dnorm=0.0f;int64_t idx=0;float scale=p>0.0f?1.0f/(1.0f-p):1.0f;if(valid){idx=n*static_cast<int64_t>(D)+c;const float q=mask[idx]?x[idx]*scale:0.0f;xhat=(q-mean[n])*rstd[n];dnorm=gy[idx]*w[c];}__shared__ float red[8];const float sum1=block_sum(dnorm,red);const float sum2=block_sum(dnorm*xhat,red);if(valid){const float dq=(rstd[n]/static_cast<float>(D))*(static_cast<float>(D)*dnorm-sum1-xhat*sum2);dx[idx]=mask[idx]?dq*scale:0.0f;}}\n__global__ void add_ln_dz_kernel(const float* gy,const float* a,const float* b,const float* w,const float* mean,const float* rstd,float* dz,int64_t N,int D){const int64_t n=blockIdx.x;const int c=threadIdx.x;if(n>=N)return;const bool valid=c<D;float xhat=0.0f,dnorm=0.0f;int64_t idx=0;if(valid){idx=n*static_cast<int64_t>(D)+c;const float z=a[idx]+b[idx];xhat=(z-mean[n])*rstd[n];dnorm=gy[idx]*w[c];}__shared__ float red[8];const float sum1=block_sum(dnorm,red);const float sum2=block_sum(dnorm*xhat,red);if(valid)dz[idx]=(rstd[n]/static_cast<float>(D))*(static_cast<float>(D)*dnorm-sum1-xhat*sum2);}\n__global__ void ln_param_stage1_dropout_kernel(const float* gy,const float* x,const float* mean,const float* rstd,const uint8_t* mask,float* pw,float* pb,int64_t N,int64_t chunks,int D,float p){const int chunk=blockIdx.x;const int c=threadIdx.x;if(chunk>=chunks||c>=D)return;const int64_t n0=static_cast<int64_t>(chunk)*PARAM_CHUNK_N;const int64_t n1=(n0+PARAM_CHUNK_N<N)?(n0+PARAM_CHUNK_N):N;const float scale=p>0.0f?1.0f/(1.0f-p):1.0f;float sw=0,sb=0;for(int64_t n=n0;n<n1;++n){const int64_t idx=n*D+c;const float q=mask[idx]?x[idx]*scale:0.0f;const float xhat=(q-mean[n])*rstd[n];const float g=gy[idx];sw+=g*xhat;sb+=g;}pw[static_cast<int64_t>(chunk)*D+c]=sw;pb[static_cast<int64_t>(chunk)*D+c]=sb;}\n__global__ void ln_param_stage1_add_kernel(const float* gy,const float* a,const float* b,const float* mean,const float* rstd,float* pw,float* pb,int64_t N,int64_t chunks,int D){const int chunk=blockIdx.x;const int c=threadIdx.x;if(chunk>=chunks||c>=D)return;const int64_t n0=static_cast<int64_t>(chunk)*PARAM_CHUNK_N;const int64_t n1=(n0+PARAM_CHUNK_N<N)?(n0+PARAM_CHUNK_N):N;float sw=0,sb=0;for(int64_t n=n0;n<n1;++n){const int64_t idx=n*D+c;const float z=a[idx]+b[idx];const float xhat=(z-mean[n])*rstd[n];const float g=gy[idx];sw+=g*xhat;sb+=g;}pw[static_cast<int64_t>(chunk)*D+c]=sw;pb[static_cast<int64_t>(chunk)*D+c]=sb;}\n__global__ void param_stage2_kernel(const float* pw,const float* pb,float* gw,float* gb,int64_t chunks,int D){const int c=threadIdx.x;if(c>=D)return;float sw=0,sb=0;for(int64_t k=0;k<chunks;++k){sw+=pw[k*D+c];sb+=pb[k*D+c];}gw[c]=sw;gb[c]=sb;}\n__global__ void gamma_residual_forward_kernel(const float* h,const float* g,const float* gamma,float* y,uint8_t* mask,int64_t N,int D,float p,uint64_t seed){const int64_t n=blockIdx.x;const int c=threadIdx.x;if(n>=N||c>=D)return;const int64_t idx=n*static_cast<int64_t>(D)+c;const bool keep=dropout_keep(seed,static_cast<uint64_t>(idx),p);mask[idx]=keep?1:0;const float scale=p>0.0f?1.0f/(1.0f-p):1.0f;const float gd=keep?g[idx]*scale:0.0f;y[idx]=h[idx]+gamma[c]*gd;}\n__global__ void gamma_backward_stage1_kernel(const float* gy,const float* g,const float* gamma,const uint8_t* mask,float* grad_g,float* partial,int64_t N,int64_t chunks,int D,float p){const int chunk=blockIdx.x;const int c=threadIdx.x;if(chunk>=chunks||c>=D)return;const int64_t n0=static_cast<int64_t>(chunk)*PARAM_CHUNK_N;const int64_t n1=(n0+PARAM_CHUNK_N<N)?(n0+PARAM_CHUNK_N):N;const float scale=p>0.0f?1.0f/(1.0f-p):1.0f;float sg=0;for(int64_t n=n0;n<n1;++n){const int64_t idx=n*D+c;const float m=mask[idx]?scale:0.0f;const float go=gy[idx];grad_g[idx]=go*gamma[c]*m;sg+=go*g[idx]*m;}partial[static_cast<int64_t>(chunk)*D+c]=sg;}\n__global__ void gamma_stage2_kernel(const float* partial,float* gg,int64_t chunks,int D){const int c=threadIdx.x;if(c>=D)return;float s=0;for(int64_t k=0;k<chunks;++k)s+=partial[k*D+c];gg[c]=s;}\n}\n\nstd::vector<torch::Tensor> dropout_layernorm_forward_cuda(torch::Tensor x,torch::Tensor w,torch::Tensor b,double p,double eps,int64_t seed){c10::cuda::CUDAGuard guard(x.device());const int D=x.size(1),T=launch_threads(D);auto y=torch::empty_like(x);auto mean=torch::empty({x.size(0)},x.options());auto rstd=torch::empty({x.size(0)},x.options());auto mask=torch::empty(x.sizes(),x.options().dtype(torch::kUInt8));auto stream=at::cuda::getCurrentCUDAStream();dropout_ln_forward_kernel<<<x.size(0),T,0,stream>>>(x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),mask.data_ptr<uint8_t>(),x.size(0),D,(float)p,(float)eps,(uint64_t)seed);C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,mean,rstd,mask};}\nstd::vector<torch::Tensor> dropout_layernorm_backward_cuda(torch::Tensor gy,torch::Tensor x,torch::Tensor w,torch::Tensor mean,torch::Tensor rstd,torch::Tensor mask,double p){c10::cuda::CUDAGuard guard(x.device());const int D=x.size(1),T=launch_threads(D);const int64_t chunks=(x.size(0)+PARAM_CHUNK_N-1)/PARAM_CHUNK_N;auto dx=torch::empty_like(x);auto pw=torch::empty({chunks,D},x.options());auto pb=torch::empty({chunks,D},x.options());auto gw=torch::empty({D},x.options());auto gb=torch::empty({D},x.options());auto stream=at::cuda::getCurrentCUDAStream();dropout_ln_dx_kernel<<<x.size(0),T,0,stream>>>(gy.data_ptr<float>(),x.data_ptr<float>(),w.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),mask.data_ptr<uint8_t>(),dx.data_ptr<float>(),x.size(0),D,(float)p);ln_param_stage1_dropout_kernel<<<chunks,T,0,stream>>>(gy.data_ptr<float>(),x.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),mask.data_ptr<uint8_t>(),pw.data_ptr<float>(),pb.data_ptr<float>(),x.size(0),chunks,D,(float)p);param_stage2_kernel<<<1,T,0,stream>>>(pw.data_ptr<float>(),pb.data_ptr<float>(),gw.data_ptr<float>(),gb.data_ptr<float>(),chunks,D);C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,gw,gb};}\nstd::vector<torch::Tensor> add_layernorm_forward_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor w,torch::Tensor bias,double eps){c10::cuda::CUDAGuard guard(a.device());const int D=a.size(1),T=launch_threads(D);auto y=torch::empty_like(a);auto mean=torch::empty({a.size(0)},a.options());auto rstd=torch::empty({a.size(0)},a.options());auto stream=at::cuda::getCurrentCUDAStream();add_ln_forward_kernel<<<a.size(0),T,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),w.data_ptr<float>(),bias.data_ptr<float>(),y.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),a.size(0),D,(float)eps);C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,mean,rstd};}\nstd::vector<torch::Tensor> add_layernorm_backward_cuda(torch::Tensor gy,torch::Tensor a,torch::Tensor b,torch::Tensor w,torch::Tensor mean,torch::Tensor rstd){c10::cuda::CUDAGuard guard(a.device());const int D=a.size(1),T=launch_threads(D);const int64_t chunks=(a.size(0)+PARAM_CHUNK_N-1)/PARAM_CHUNK_N;auto dz=torch::empty_like(a);auto pw=torch::empty({chunks,D},a.options());auto pb=torch::empty({chunks,D},a.options());auto gw=torch::empty({D},a.options());auto gb=torch::empty({D},a.options());auto stream=at::cuda::getCurrentCUDAStream();add_ln_dz_kernel<<<a.size(0),T,0,stream>>>(gy.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),w.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),dz.data_ptr<float>(),a.size(0),D);ln_param_stage1_add_kernel<<<chunks,T,0,stream>>>(gy.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),pw.data_ptr<float>(),pb.data_ptr<float>(),a.size(0),chunks,D);param_stage2_kernel<<<1,T,0,stream>>>(pw.data_ptr<float>(),pb.data_ptr<float>(),gw.data_ptr<float>(),gb.data_ptr<float>(),chunks,D);C10_CUDA_KERNEL_LAUNCH_CHECK();return {dz,gw,gb};}\nstd::vector<torch::Tensor> dropout_gamma_residual_forward_cuda(torch::Tensor h,torch::Tensor g,torch::Tensor gamma,double p,int64_t seed){c10::cuda::CUDAGuard guard(h.device());const int D=h.size(1),T=launch_threads(D);auto y=torch::empty_like(h);auto mask=torch::empty(h.sizes(),h.options().dtype(torch::kUInt8));auto stream=at::cuda::getCurrentCUDAStream();gamma_residual_forward_kernel<<<h.size(0),T,0,stream>>>(h.data_ptr<float>(),g.data_ptr<float>(),gamma.data_ptr<float>(),y.data_ptr<float>(),mask.data_ptr<uint8_t>(),h.size(0),D,(float)p,(uint64_t)seed);C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,mask};}\nstd::vector<torch::Tensor> dropout_gamma_residual_backward_cuda(torch::Tensor gy,torch::Tensor g,torch::Tensor gamma,torch::Tensor mask,double p){c10::cuda::CUDAGuard guard(g.device());const int D=g.size(1),T=launch_threads(D);const int64_t chunks=(g.size(0)+PARAM_CHUNK_N-1)/PARAM_CHUNK_N;auto grad_g=torch::empty_like(g);auto partial=torch::empty({chunks,D},g.options());auto gg=torch::empty({D},g.options());auto stream=at::cuda::getCurrentCUDAStream();gamma_backward_stage1_kernel<<<chunks,T,0,stream>>>(gy.data_ptr<float>(),g.data_ptr<float>(),gamma.data_ptr<float>(),mask.data_ptr<uint8_t>(),grad_g.data_ptr<float>(),partial.data_ptr<float>(),g.size(0),chunks,D,(float)p);gamma_stage2_kernel<<<1,T,0,stream>>>(partial.data_ptr<float>(),gg.data_ptr<float>(),chunks,D);C10_CUDA_KERNEL_LAUNCH_CHECK();return {grad_g,gg};}\n'

def _extension_name():
    return "graph_clifford_fused_block_ops_general_v2"

def ensure_fused_block_ops_loaded(verbose: bool=False):
    """Build/load generalized V8 block-fusion kernels for 1<=D<=256."""
    global _EXT,_EXT_SPECIAL,_EXT_ERROR
    if _EXT is not None and _EXT_SPECIAL is not None:return _EXT
    if _EXT_ERROR is not None:raise RuntimeError("Generalized block-op extension previously failed to load") from _EXT_ERROR
    if not torch.cuda.is_available():raise RuntimeError("Generalized fused block ops require CUDA")
    root=Path(__file__).resolve().parent
    build_dir=root/".fused_block_general_build"
    special_build_dir=root/".fused_block_ops_build"
    build_dir.mkdir(parents=True,exist_ok=True);special_build_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST","8.6")
    try:
        # Exact V8-v1 kernel remains the D=256 fast path; generalized CUDA is
        # dispatched for smaller hidden widths. This avoids slowing the proven V9 case.
        src_dir=root/"fused_block_ops_ext"
        _EXT_SPECIAL=load(name="graph_clifford_fused_block_ops_a3v8_v1",
                          sources=[str(src_dir/"fused_block_ops.cpp"),str(src_dir/"fused_block_ops_cuda.cu")],
                          extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],
                          build_directory=str(special_build_dir),verbose=verbose)
        _EXT=load_inline(name=_extension_name(),cpp_sources=_CPP_SOURCE,cuda_sources=_CUDA_SOURCE,functions=None,
                         extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],with_cuda=True,
                         build_directory=str(build_dir),verbose=verbose)
        return _EXT
    except Exception as exc:
        _EXT_ERROR=exc;raise

def _next_seed(tag:int,dropout_p:float)->int:
    if float(dropout_p)<=0.0:return 0
    base=int(torch.initial_seed())&((1<<63)-1)
    call=next(_CALL_COUNTER)+1
    return (base+0x9E3779B97F4A7C15*call+0xD2B74407B1CE6E93*int(tag))&((1<<63)-1)

class _DropoutLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,dropout_p,eps,seed):
        ensure_fused_block_ops_loaded(False)
        global _EXT,_EXT_SPECIAL
        ctx.use_special = int(x.size(1)) == 256
        ext = _EXT_SPECIAL if ctx.use_special else _EXT
        y,mean,rstd,mask=ext.dropout_layernorm_forward(x.contiguous(),weight.contiguous(),bias.contiguous(),float(dropout_p),float(eps),int(seed))
        ctx.save_for_backward(x,weight,mean,rstd,mask);ctx.dropout_p=float(dropout_p);return y
    @staticmethod
    def backward(ctx,grad_y):
        ensure_fused_block_ops_loaded(False);x,weight,mean,rstd,mask=ctx.saved_tensors
        global _EXT,_EXT_SPECIAL
        ext = _EXT_SPECIAL if ctx.use_special else _EXT
        grad_y=grad_y.to(dtype=torch.float32).contiguous()
        gx,gw,gb=ext.dropout_layernorm_backward(grad_y,x.contiguous(),weight.contiguous(),mean.contiguous(),rstd.contiguous(),mask.contiguous(),float(ctx.dropout_p))
        return gx,gw,gb,None,None,None

class _AddLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,weight,bias,eps):
        ensure_fused_block_ops_loaded(False)
        global _EXT,_EXT_SPECIAL
        ctx.use_special = int(a.size(1)) == 256
        ext = _EXT_SPECIAL if ctx.use_special else _EXT
        y,mean,rstd=ext.add_layernorm_forward(a.contiguous(),b.contiguous(),weight.contiguous(),bias.contiguous(),float(eps));ctx.save_for_backward(a,b,weight,mean,rstd);return y
    @staticmethod
    def backward(ctx,grad_y):
        ensure_fused_block_ops_loaded(False);a,b,weight,mean,rstd=ctx.saved_tensors
        global _EXT,_EXT_SPECIAL
        ext = _EXT_SPECIAL if ctx.use_special else _EXT
        grad_y=grad_y.to(dtype=torch.float32).contiguous()
        gz,gw,gb=ext.add_layernorm_backward(grad_y,a.contiguous(),b.contiguous(),weight.contiguous(),mean.contiguous(),rstd.contiguous());return gz,gz,gw,gb,None

class _DropoutGammaResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,g,gamma,dropout_p,seed):
        ensure_fused_block_ops_loaded(False)
        global _EXT,_EXT_SPECIAL
        ctx.use_special = int(h.size(1)) == 256
        ext = _EXT_SPECIAL if ctx.use_special else _EXT
        y,mask=ext.dropout_gamma_residual_forward(h.contiguous(),g.contiguous(),gamma.contiguous(),float(dropout_p),int(seed));ctx.save_for_backward(g,gamma,mask);ctx.dropout_p=float(dropout_p);return y
    @staticmethod
    def backward(ctx,grad_y):
        ensure_fused_block_ops_loaded(False);g,gamma,mask=ctx.saved_tensors
        global _EXT,_EXT_SPECIAL
        ext = _EXT_SPECIAL if ctx.use_special else _EXT
        grad_y=grad_y.to(dtype=torch.float32).contiguous()
        gg,ggamma=ext.dropout_gamma_residual_backward(grad_y,g.contiguous(),gamma.contiguous(),mask.contiguous(),float(ctx.dropout_p));return grad_y,gg,ggamma,None,None

def _check_2d(x,name):
    if x.device.type!="cuda" or x.dtype!=torch.float32:raise TypeError(f"{name} must be CUDA float32")
    if x.dim()!=2 or not (1<=x.size(1)<=256):raise ValueError(f"{name} must be [N,D] with 1<=D<=256, got {tuple(x.shape)}")
    return int(x.size(1))

def fused_dropout_layernorm(x,weight,bias,dropout_p,eps):
    D=_check_2d(x,"x")
    if weight.numel()!=D or bias.numel()!=D:raise ValueError("LayerNorm weight/bias must match hidden dimension")
    return _DropoutLayerNormFn.apply(x,weight,bias,float(dropout_p),float(eps),_next_seed(1,dropout_p))

def fused_add_layernorm(a,b,weight,bias,eps):
    D=_check_2d(a,"a");_check_2d(b,"b")
    if a.shape!=b.shape or weight.numel()!=D or bias.numel()!=D:raise ValueError("Add+LayerNorm shapes do not match")
    return _AddLayerNormFn.apply(a,b,weight,bias,float(eps))

def fused_dropout_gamma_residual(h,g,gamma,dropout_p):
    D=_check_2d(h,"h");_check_2d(g,"g")
    if h.shape!=g.shape:raise ValueError("h and g must have the same shape")
    if gamma.dim()!=1 or gamma.numel()!=D:raise ValueError(f"vector gamma must have shape [{D}]")
    return _DropoutGammaResidualFn.apply(h,g,gamma,float(dropout_p),_next_seed(2,dropout_p))
