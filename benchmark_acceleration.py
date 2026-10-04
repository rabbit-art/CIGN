"""Benchmark restored backend selection; does not run the ten-split experiment."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FLAGS=['use_fused_gat','use_fused_gat_sideops','use_collapsed_fused_gat','use_fused_block_ops','use_fused_clifford_pack','use_multihop_csr_pipeline']
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--datasets',nargs='+',default=['Amazon-ratings']);p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--gpu',default='0');p.add_argument('--output',type=Path,default=ROOT/'acceleration_benchmark');p.add_argument('--steps',type=int,default=60);p.add_argument('--warmup',type=int,default=10)
 p.add_argument('--config',type=Path,required=True,help='Custom training JSON; requires exactly one dataset')
 args=p.parse_args()
 if args.config and len(args.datasets)!=1:p.error('--config requires exactly one --datasets value')
 if args.steps<1 or args.warmup<0:p.error('invalid step counts')
 args.output.mkdir(parents=True,exist_ok=True);results={}
 for name in args.datasets:
  cfg=json.loads((args.config or ROOT/'configs'/(name+'.json')).read_text());cfg.update({k:True for k in FLAGS})
  cfg.update(data_root=str(args.data_root.resolve()),dataset_name=name,seed=42,split_seed=42,save_log=False,enable_dirichlet_recording=False,generic_backend_profile='auto',fast_backend_policy='auto',speed_probe_steps=args.steps,speed_probe_warmup=args.warmup,generic_speed_guard='refresh')
  cmd=[sys.executable,str(ROOT/'local_train.py')]
  for k,v in cfg.items():cmd+=['--'+k]+[str(x) for x in (v if isinstance(v,list) else [v])]
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1')
  log=args.output/(name+'.log');print('Benchmark',name,'->',log,flush=True)
  with log.open('w',encoding='utf-8') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   try:rc=child.wait()
   except KeyboardInterrupt:
    child.terminate();child.wait();raise
  lines=log.read_text(encoding='utf-8',errors='replace').splitlines();rows=[json.loads(x.split(':',1)[1]) for x in lines if x.startswith('SPEED_PROBE_RESULT_JSON:')]
  if rc or not rows or rows[-1].get('status')!='ok':raise RuntimeError('Benchmark failed; see '+str(log))
  results[name]=rows[-1];(args.output/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
  print(json.dumps(results[name],indent=2),flush=True)
if __name__=='__main__':main()
