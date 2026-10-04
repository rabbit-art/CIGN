"""Validate CIGN fused forward/backward on the target CUDA environment."""
import os,sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'generic_engine'))
from fused_clifford_cign_pack import fused_clifford_switch_pack
from fused_clifford_cign_generic import fused_clifford_pack

def reference(h,c,props,a,weight):
    ws=[];ds=[]
    for p in props:
        th,tc=p.split(h.shape[1],-1)
        ds.append(torch.nn.functional.silu(.5*(h*tc+c*th)))
        ws.append(.5*(h*tc-c*th))
    d=torch.stack(ds,1);w=torch.stack(ws,1)
    if weight=='old':return (torch.cat([w,d],-1)*a[:,:,None]).flatten(1)
    return torch.cat([(d*a[:,:,None]).sum(1),(w*a[:,:,None]).sum(1)],-1)

def one_case(k,d,weight):
    torch.manual_seed(42+k+d);n=17
    raw=[torch.randn(n,d,device='cuda'),torch.randn(n,d,device='cuda')]+[torch.randn(n,2*d,device='cuda') for _ in range(k)]+[torch.randn(n,k,device='cuda')]
    a=[x.detach().clone().requires_grad_(True) for x in raw];b=[x.detach().clone().requires_grad_(True) for x in raw]
    expected=reference(a[0],a[1],a[2:-1],torch.softmax(a[-1],-1),weight)
    alpha=torch.softmax(b[-1],-1)
    if k==3:actual=fused_clifford_switch_pack(b[0],b[1],b[2:-1],alpha,'new',weight)
    else:
        actual=fused_clifford_pack(b[0],b[1],b[2:-1],alpha)
        if weight=='new':
            p=actual.reshape(n,k,2*d);actual=torch.cat([p[:,:,d:].sum(1),p[:,:,:d].sum(1)],-1)
    go=torch.randn_like(expected)
    ga=torch.autograd.grad(expected,a,go);gb=torch.autograd.grad(actual,b,go)
    errors=[float((x-y).norm()/y.norm().clamp_min(1e-12)) for x,y in zip([actual,*gb],[expected,*ga])]
    if not all(torch.isfinite(x).all() for x in [actual,*gb]) or max(errors)>5e-5:raise RuntimeError((k,d,weight,errors))
    print(f'PASS K={k} D={d} weighting={weight} max_relative_error={max(errors):.3e}',flush=True)
    return max(errors)

def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cap=torch.cuda.get_device_capability();os.environ.setdefault('TORCH_CUDA_ARCH_LIST',f'{cap[0]}.{cap[1]}')
    errors=[one_case(k,d,w) for k,d in [(1,24),(2,32),(3,24),(3,64),(3,256),(4,128),(5,24),(5,64)] for w in ['old','new']]
    os.environ.update(CIGN_ALPHA_SCORE_MODE='new', CIGN_INTERACTION_MODE='new', CIGN_ALPHA_WEIGHTING_MODE='old', CIGN_OUTER_NONLINEARITY='none')
    from model_cign import build_model
    for hops in ([1,2,3],[1,2,4,8,16]):
        torch.manual_seed(42);n=23
        edge=torch.stack([torch.arange(n,device='cuda'),torch.arange(n,device='cuda').roll(1)])
        operator=torch.sparse_coo_tensor(edge,torch.ones(n,device='cuda'),(n,n)).coalesce()
        x=torch.randn(n,8,device='cuda')
        kwargs=dict(in_dim=8,hidden_dim=24,out_dim=3,num_layers=2,dropout=0.,gat_dropout=0.,gat_heads=2,hop_scales=hops)
        a=build_model(**kwargs,use_fused_clifford_pack=False).cuda()
        b=build_model(**kwargs,use_fused_clifford_pack=True).cuda();b.load_state_dict(a.state_dict())
        ya=a(x,edge,operator);yb=b(x,edge,operator);go=torch.randn_like(ya)
        (ya*go).sum().backward();(yb*go).sum().backward()
        torch.testing.assert_close(yb,ya,rtol=2e-4,atol=2e-5)
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb
            if pa.grad is not None:torch.testing.assert_close(pb.grad,pa.grad,rtol=5e-4,atol=5e-5,msg=na)
        print('MODEL_OUTPUT_AND_GRADIENT_PASS',hops,flush=True)
    result={'status':'PASS' ,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'maximum_relative_error':max(errors)}
    print(json.dumps(result,indent=2));print('CIGN_ACCEL_VALIDATION_PASS')
if __name__=='__main__':main()
