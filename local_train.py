"""CIGN accelerated entry: production backend autotune, no GAT monkeypatch."""
import os, sys
from pathlib import Path
if __name__ == '__main__':
    os.environ.update(CIGN_ALPHA_SCORE_MODE='new', CIGN_INTERACTION_MODE='new', CIGN_ALPHA_WEIGHTING_MODE='old', CIGN_OUTER_NONLINEARITY='none')
    import torch
    if torch.cuda.is_available():
        caps={torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', ';'.join(f'{a}.{b}' for a,b in sorted(caps)))
    root=Path(__file__).resolve().parent
    import cign_dispatch_utils as u
    key,_=u.cign_speed_signature(sys.argv[1:])
    # Per-signature AND device cache prevents concurrent four-GPU registry lost updates.
    device=os.environ.get('CUDA_VISIBLE_DEVICES','default').replace('/','_').replace('\\','_')
    registry=root/'acceleration_cache'/device/(key+'.json')
    registry.parent.mkdir(parents=True,exist_ok=True)
    args=sys.argv[1:]
    if '--generic_speed_registry' not in args:args+=['--generic_speed_registry',str(registry)]
    print('OUTER_NONLINEARITY  : none',flush=True)
    os.execv(sys.executable,[sys.executable,str(root/'train_cign_dispatch.py'),*args])
