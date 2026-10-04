"""Sequential, portable ten-split reproduction of the CIGN concat model."""
import argparse, json, os, re, statistics, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
USE_OLD_ALPHA_SCORE=False
USE_NEW_ALPHA_SCORE=True
USE_OLD_INTERACTION=False
USE_NEW_INTERACTION=True
USE_OLD_ALPHA_WEIGHTING=True
USE_NEW_ALPHA_WEIGHTING=False
SEEDS=[42,47,50,43,44,45,46,48,49,51]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='Amazon-ratings',choices=['Actor','Amazon-ratings','Cora','Minesweeper','PubMed','Questions','Roman-empire','Tolokers'])
    p.add_argument('--config',type=Path,required=True,help='Your external JSON training configuration; no tuned configurations are distributed')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--splits',type=int,choices=range(1,11),default=10)
    p.add_argument('--gpu',default='0')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,help='Override for quick checks; not historical reproduction')
    p.add_argument('--patience',type=int,help='Override early stopping')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8-sig'))
    c.update(dataset_name=args.dataset,data_root=str(args.data_root.resolve()),device=args.device,seed=42,split_index=0)
    for key in ['use_fused_gat','use_fused_gat_sideops','use_collapsed_fused_gat','use_fused_block_ops','use_fused_clifford_pack','use_multihop_csr_pipeline']:c[key]=True
    if args.epochs is not None:c['epochs']=args.epochs
    if args.patience is not None:c['patience']=args.patience
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',CUDA_VISIBLE_DEVICES=args.gpu,CIGN_ALPHA_SCORE_MODE='new',CIGN_INTERACTION_MODE='new',CIGN_ALPHA_WEIGHTING_MODE='old',CIGN_OUTER_NONLINEARITY='none',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CLIFFORD_RECORD_DIRICHLET='0',CLIFFORD_EDGE_DROP_RATIO='0')
    for key in list(env):
        if key.startswith('CLIFFORD_EDGE_DROP_') and key!='CLIFFORD_EDGE_DROP_RATIO':env.pop(key)
    dest=args.output.resolve()/args.dataset
    if not args.dry_run:dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in SEEDS[:args.splits]:
        c['split_seed']=seed
        cmd=[sys.executable,str(ROOT/'local_train.py')]
        for key,value in c.items():cmd+=['--'+key]+[str(v) for v in (value if isinstance(value,list) else [value])]
        if args.dry_run:
            print(json.dumps({'architecture':'CIGN','command':cmd}));continue
        log=dest/f'split_{seed}.log'
        print(f'{args.dataset}: split {seed} -> {log}',flush=True)
        with log.open('w',encoding='utf-8') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            try:code=proc.wait()
            except KeyboardInterrupt:
                proc.terminate();proc.wait();raise
        text=log.read_text(encoding='utf-8',errors='replace')
        if code!=0 or not re.search(r'^RUN_STATUS\s*:\s*OK\s*$',text,re.M):raise RuntimeError(f'Training failed; see {log}')
        for label,value in [('ALPHA_SCORE_MODE','new'),('INTERACTION_MODE','new'),('ALPHA_WEIGHT_MODE','old'),('OUTER_NONLINEARITY','none')]:
            if not re.search(r'^'+label+r'\s*:\s*'+value+r'\s*$',text,re.M):raise RuntimeError('Unexpected architecture: '+label)
        row={'split_seed':seed}
        for key,pattern in [('best_val',r'^Best validation .*?:\s*([0-9.]+)%'),('test_at_best_val',r'^Test .*?@ best val\s*:\s*([0-9.]+)%')]:
            found=re.findall(pattern,text,re.M)
            if not found:raise RuntimeError(f'Missing {key}: {log}')
            row[key]=float(found[-1])
        rows.append(row);(dest/'partial_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(row,flush=True)
    if args.dry_run:return
    result={'dataset':args.dataset,'architecture':'CIGN','n_splits':len(rows),'full_ten_splits':len(rows)==10,'training_config':c,'mean_val':statistics.mean(r['best_val'] for r in rows),'mean_test_at_best_val':statistics.mean(r['test_at_best_val'] for r in rows),'std_test_at_best_val':statistics.stdev(r['test_at_best_val'] for r in rows) if len(rows)>1 else None,'splits':rows}
    (dest/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
