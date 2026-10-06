"""Paired greedy/sampled validation; does not select checkpoints or alter demo data."""
import argparse
import json
from pathlib import Path
import torch
from recoverrl import KINDS, RecoveryEnv, load_model, wilson


def compare(checkpoints, episodes, start_seed):
    torch.set_num_threads(1)
    results=[]
    for checkpoint in checkpoints:
        model=load_model(checkpoint)
        for mode in ['greedy','sampled']:
            for kind in KINDS:
                rows=[]
                for seed in range(start_seed,start_seed+episodes):
                    generator=torch.Generator().manual_seed(seed+3000000)
                    env=RecoveryEnv(); obs=env.reset(seed,kind)
                    total=0
                    while not env.done:
                        with torch.no_grad():
                            logits=model(torch.tensor(obs))[0]
                            action=int(logits.argmax()) if mode=='greedy' else int(torch.multinomial(logits.softmax(-1),1,generator=generator))
                        obs,r,_,_=env.step(action); total+=r
                    rows.append({'seed':seed,'success':env.success,'steps':env.steps,'injected':env.injected,'return':total})
                successes=sum(r['success'] for r in rows)
                row=dict(checkpoint=checkpoint,mode=mode,kind=kind,success_rate=successes/episodes,
                         ci95=wilson(successes,episodes),mean_steps=sum(r['steps'] for r in rows)/episodes,
                         mean_success_steps=sum(r['steps'] for r in rows if r['success'])/successes if successes else None,
                         episodes=rows)
                results.append(row)
            group=results[-4:]
            print(f'{checkpoint} {mode}: {sum(r["success_rate"] for r in group)/4:.2%}',flush=True)
    return dict(start_seed=start_seed,episodes_per_condition=episodes,results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints',nargs='+'); p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--start-seed',type=int,default=1000000); p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.episodes<1: p.error('--episodes must be positive')
    result=compare(a.checkpoints,a.episodes,a.start_seed)
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
