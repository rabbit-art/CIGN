from __future__ import annotations

"""Specialized K=3 + generic K Clifford feature packing CUDA backends."""

import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load, load_inline

_EXT = None
_EXT_ERROR = None
_GENERIC_EXT = None
_GENERIC_EXT_ERROR = None


def _extension_name() -> str:
    return "graph_clifford_fused_pack_a3_v1"


def ensure_special_fused_clifford_pack_loaded(verbose: bool = False):
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("A3 fused Clifford extension previously failed to load") from _EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("A3 fused Clifford pack requires CUDA.")
    root = Path(__file__).resolve().parent
    src_dir = root / "fused_clifford_pack_ext"
    build_dir = root / ".fused_clifford_pack_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    try:
        _EXT = load(
            name=_extension_name(),
            sources=[str(src_dir / "fused_clifford_pack.cpp"), str(src_dir / "fused_clifford_pack_cuda.cu")],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir), verbose=verbose,
        )
        return _EXT
    except Exception as exc:
        _EXT_ERROR = exc
        raise


_GENERIC_CPP = r'''
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> generic_pack_forward_cuda(
    torch::Tensor H, torch::Tensor C, torch::Tensor P, torch::Tensor alpha);
std::vector<torch::Tensor> generic_pack_backward_cuda(
    torch::Tensor grad_out, torch::Tensor H, torch::Tensor C,
    torch::Tensor P, torch::Tensor alpha);
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
static void check_f(const torch::Tensor& x){CHECK_CUDA(x);CHECK_CONTIGUOUS(x);CHECK_FLOAT(x);}
std::vector<torch::Tensor> forward(torch::Tensor H, torch::Tensor C, torch::Tensor P, torch::Tensor alpha){
  check_f(H);check_f(C);check_f(P);check_f(alpha);
  TORCH_CHECK(H.dim()==2 && C.sizes()==H.sizes(), "H/C must be [N,D]");
  TORCH_CHECK(P.dim()==3 && P.size(0)==H.size(0) && P.size(2)==2*H.size(1), "P must be [N,K,2D]");
  TORCH_CHECK(alpha.dim()==2 && alpha.size(0)==H.size(0) && alpha.size(1)==P.size(1), "alpha must be [N,K]");
  return generic_pack_forward_cuda(H,C,P,alpha);
}
std::vector<torch::Tensor> backward(torch::Tensor grad_out, torch::Tensor H, torch::Tensor C, torch::Tensor P, torch::Tensor alpha){
  check_f(grad_out);check_f(H);check_f(C);check_f(P);check_f(alpha);
  return generic_pack_backward_cuda(grad_out,H,C,P,alpha);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
'''

_GENERIC_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

__device__ __forceinline__ float silu_f(float x){float s=1.0f/(1.0f+expf(-x));return x*s;}
__device__ __forceinline__ float silu_grad_f(float x){float s=1.0f/(1.0f+expf(-x));return s*(1.0f+x*(1.0f-s));}

__global__ void pack_forward_kernel(
    int64_t total, int64_t D, int64_t K,
    const float* H,const float* C,const float* P,const float* A,float* O){
  for(int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;idx<total;idx+=(int64_t)blockDim.x*gridDim.x){
    int64_t d=idx%D; int64_t q=idx/D; int64_t k=q%K; int64_t n=q/K;
    int64_t hidx=n*D+d; int64_t pbase=(n*K+k)*(2*D);
    float h=H[hidx], c=C[hidx], th=P[pbase+d], tc=P[pbase+D+d];
    float u=h*tc; float w=u-th*c; float dv=silu_f(u); float a=A[n*K+k];
    O[pbase+d]=a*w; O[pbase+D+d]=a*dv;
  }
}

__global__ void pack_backward_kernel(
    int64_t total, int64_t D, int64_t K,
    const float* GO,const float* H,const float* C,const float* P,const float* A,
    float* GH,float* GC,float* GP,float* GA){
  for(int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;idx<total;idx+=(int64_t)blockDim.x*gridDim.x){
    int64_t d=idx%D; int64_t q=idx/D; int64_t k=q%K; int64_t n=q/K;
    int64_t hidx=n*D+d; int64_t pbase=(n*K+k)*(2*D); int64_t aidx=n*K+k;
    float h=H[hidx], c=C[hidx], th=P[pbase+d], tc=P[pbase+D+d], a=A[aidx];
    float u=h*tc; float w=u-th*c; float dv=silu_f(u); float sd=silu_grad_f(u);
    float gwo=GO[pbase+d], gdo=GO[pbase+D+d];
    float gw=gwo*a, gd=gdo*a;
    atomicAdd(&GH[hidx], gw*tc + gd*sd*tc);
    atomicAdd(&GC[hidx], -gw*th);
    GP[pbase+d] = -gw*c;
    GP[pbase+D+d] = gw*h + gd*sd*h;
    atomicAdd(&GA[aidx], gwo*w + gdo*dv);
  }
}

std::vector<torch::Tensor> generic_pack_forward_cuda(
    torch::Tensor H, torch::Tensor C, torch::Tensor P, torch::Tensor alpha){
  c10::cuda::CUDAGuard guard(H.device());
  int64_t N=H.size(0),D=H.size(1),K=P.size(1); int64_t total=N*K*D;
  auto out=torch::empty({N,K*2*D},H.options());
  auto stream=at::cuda::getCurrentCUDAStream(); constexpr int threads=256;
  int blocks=(int)std::min<int64_t>((total+threads-1)/threads,65535);
  pack_forward_kernel<<<blocks,threads,0,stream>>>(total,D,K,H.data_ptr<float>(),C.data_ptr<float>(),P.data_ptr<float>(),alpha.data_ptr<float>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {out};
}

std::vector<torch::Tensor> generic_pack_backward_cuda(
    torch::Tensor grad_out, torch::Tensor H, torch::Tensor C, torch::Tensor P, torch::Tensor alpha){
  c10::cuda::CUDAGuard guard(H.device());
  int64_t N=H.size(0),D=H.size(1),K=P.size(1); int64_t total=N*K*D;
  auto gh=torch::zeros_like(H); auto gc=torch::zeros_like(C); auto gp=torch::empty_like(P); auto ga=torch::zeros_like(alpha);
  auto stream=at::cuda::getCurrentCUDAStream(); constexpr int threads=256;
  int blocks=(int)std::min<int64_t>((total+threads-1)/threads,65535);
  pack_backward_kernel<<<blocks,threads,0,stream>>>(total,D,K,grad_out.data_ptr<float>(),H.data_ptr<float>(),C.data_ptr<float>(),P.data_ptr<float>(),alpha.data_ptr<float>(),gh.data_ptr<float>(),gc.data_ptr<float>(),gp.data_ptr<float>(),ga.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {gh,gc,gp,ga};
}
'''


def ensure_generic_fused_clifford_pack_loaded(verbose: bool = False):
    global _GENERIC_EXT, _GENERIC_EXT_ERROR
    if _GENERIC_EXT is not None:
        return _GENERIC_EXT
    if _GENERIC_EXT_ERROR is not None:
        raise RuntimeError("Generic Clifford pack extension previously failed to load") from _GENERIC_EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("Generic Clifford pack requires CUDA")
    root = Path(__file__).resolve().parent
    build_dir = root / ".fused_clifford_pack_generic_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    try:
        _GENERIC_EXT = load_inline(
            name="graph_clifford_fused_pack_generic_k_v1",
            cpp_sources=_GENERIC_CPP,
            cuda_sources=_GENERIC_CUDA,
            functions=None,
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
            with_cuda=True, build_directory=str(build_dir), verbose=verbose,
        )
        return _GENERIC_EXT
    except Exception as exc:
        _GENERIC_EXT_ERROR = exc
        raise


def ensure_fused_clifford_pack_loaded(verbose: bool = False, hop_count: int = 3):
    if int(hop_count) == 3:
        return ensure_special_fused_clifford_pack_loaded(verbose)
    return ensure_generic_fused_clifford_pack_loaded(verbose)


class FusedCliffordPack3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, C, P0, P1, P2, alpha):
        ext = ensure_special_fused_clifford_pack_loaded(False)
        out = ext.forward(H, C, P0, P1, P2, alpha)
        ctx.save_for_backward(H, C, P0, P1, P2, alpha)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ext = ensure_special_fused_clifford_pack_loaded(False)
        H, C, P0, P1, P2, alpha = ctx.saved_tensors
        grads = ext.backward(grad_output.to(dtype=torch.float32).contiguous(), H, C, P0, P1, P2, alpha)
        return tuple(grads)


class GenericFusedCliffordPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, C, P, alpha):
        ext = ensure_generic_fused_clifford_pack_loaded(False)
        out, = ext.forward(H, C, P, alpha)
        ctx.save_for_backward(H, C, P, alpha)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ext = ensure_generic_fused_clifford_pack_loaded(False)
        H, C, P, alpha = ctx.saved_tensors
        gh, gc, gp, ga = ext.backward(
            grad_output.to(dtype=torch.float32).contiguous(), H, C, P, alpha
        )
        return gh, gc, gp, ga


def _normalize_alpha(alpha, N: int, K: int, device) -> torch.Tensor:
    if alpha is None:
        return torch.ones((N, K), device=device, dtype=torch.float32)
    if alpha.dim() == 3 and alpha.size(-1) == 1:
        alpha = alpha.squeeze(-1)
    if tuple(alpha.shape) != (N, K):
        raise ValueError(f"Expected alpha {(N,K)}, got {tuple(alpha.shape)}")
    return alpha.to(dtype=torch.float32).contiguous()


def fused_clifford_pack(
    H_norm: torch.Tensor,
    C_norm: torch.Tensor,
    propagated: Sequence[torch.Tensor],
    alpha: torch.Tensor | None,
) -> torch.Tensor:
    if H_norm.dim() != 2 or C_norm.shape != H_norm.shape:
        raise ValueError("H_norm and C_norm must both be [N,D]")
    N, D = H_norm.shape
    K = len(propagated)
    if K <= 0:
        raise ValueError("propagated cannot be empty")
    H = H_norm.to(dtype=torch.float32).contiguous()
    C = C_norm.to(dtype=torch.float32).contiguous()
    A = _normalize_alpha(alpha, int(N), int(K), H.device)
    props = [p.to(dtype=torch.float32).contiguous() for p in propagated]
    for p in props:
        if tuple(p.shape) != (N, 2 * D):
            raise ValueError(f"Every propagated tensor must be {(N,2*D)}, got {tuple(p.shape)}")

    if K == 3:
        return FusedCliffordPack3Function.apply(H, C, props[0], props[1], props[2], A)

    P = torch.stack(props, dim=1).contiguous()  # [N,K,2D]
    return GenericFusedCliffordPackFunction.apply(H, C, P, A)
