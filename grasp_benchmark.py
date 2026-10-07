"""Frozen unseen-risk evaluation and training for grasp/deadline decisions."""
import argparse
import json
from pathlib import Path

from benchmark import canonical,digest
from grasp_env import GraspEnv,VERSION,strategy
from recoverrl import wilson

ROOT=Path(__file__).resolve().parent
DEFAULT_SUITE=ROOT/'benchmarks'/'grasp-v1'
STRATEGIES=('always-fast','always-secure','switch-after-failure','secure-when-feasible')


def build(split):
    risks={'train':[.05,.3,.6,.9],'dev':[.15,.45,.75,.95],'test':[.2,.5,.8,.98]}[split]
    slacks=[0,2,4,7] if split=='test' else [0,1,3,6]
    n=60 if split=='train' else 50
    base={'train':6000000,'dev':7000000,'test':8000000}[split]
    rows=[]
    for r,risk in enumerate(risks):
        for s,slack in enumerate(slacks):
            for i in range(n):
                rows.append(dict(id=f'{split}-{r}-{s}-{i:03}',seed=base+(r*4+s)*1000+i,
                                 risk=risk,slack=slack,slice=f'risk={risk:.2f}/slack={slack}'))
    return dict(version=VERSION,environment=VERSION,split=split,scenarios=rows,
                protocol='Fixed enumeration before evaluation; object risk is not in policy observations.')


def freeze(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    hashes={}
    for split in ('train','dev','test'):
        raw=canonical(build(split)); (directory/f'{split}.json').write_bytes(raw)
        hashes[f'{split}.json']=digest(raw)
    (directory/'LOCK.json').write_bytes(canonical(dict(version=VERSION,files=hashes)))
    print(json.dumps(hashes,indent=2))


def load(directory,split):
    if split not in ('train','dev','test'): raise ValueError('Unknown split')
    directory=Path(directory); lock=json.loads((directory/'LOCK.json').read_text())
    raw=(directory/f'{split}.json').read_bytes()
    if lock['version']!=VERSION or digest(raw)!=lock['files'][f'{split}.json']:
        raise ValueError('Frozen grasp manifest mismatch')
    data=json.loads(raw)
    if data['version']!=VERSION or data['split']!=split: raise ValueError('Invalid split metadata')
    seeds=set(); ids=set()
    for row in data['scenarios']:
        if row['seed'] in seeds or row['id'] in ids: raise ValueError('Duplicate scenario')
        seeds.add(row['seed']); ids.add(row['id']); GraspEnv().reset(row['seed'],row['risk'],row['slack'])
    return data,digest(raw)


class GraspSampler:
    def __init__(self,directory):
        self.data,self.sha256=load(directory,'train')
        self.rows=self.data['scenarios']; self.seen=set()
        self.gamma=1.  # Time is explicitly charged; do not discount macro actions equally.

    def make_env(self): return GraspEnv()

    def reset(self,env,seed):
        row=self.rows[int(digest(str(seed).encode())[:16],16)%len(self.rows)]
        self.seen.add(row['id'])
        return env.reset(row['seed'],row['risk'],row['slack'])


def episode(row,policy,mode='greedy',record=False):
    env=GraspEnv(); obs=env.reset(row['seed'],row['risk'],row['slack'])
    if isinstance(policy,str): act=lambda o:strategy(o,policy)
    else:
        import torch
        rng=torch.Generator().manual_seed(row['seed']+9000000)
        def act(o):
            with torch.no_grad():
                logits=policy(torch.tensor(o))[0]
                return int(logits.argmax()) if mode=='greedy' else int(torch.multinomial(logits.softmax(-1),1,generator=rng))
    frames=[env.frame()] if record else []
    actions=[]; total=0
    while not env.done:
        a=act(obs); actions.append(a); obs,reward,_,_=env.step(a); total+=reward
        if record: frames.append(env.frame())
    return dict(id=row['id'],seed=row['seed'],slice=row['slice'],risk=row['risk'],slack=row['slack'],
                success=env.success,elapsed=env.elapsed,deadline=env.deadline,attempts=env.attempts,
                reward=total,actions=actions,failures=env.failures,frames=frames)


def summary(rows):
    n=len(rows); wins=sum(r['success'] for r in rows)
    return dict(episodes=n,success_rate=wins/n,ci95=wilson(wins,n),mean_time=sum(r['elapsed'] for r in rows)/n,
                time_per_success=sum(r['elapsed'] for r in rows if r['success'])/wins if wins else None,
                mean_reward=sum(r['reward'] for r in rows)/n,
                secure_fraction=sum(a==1 for r in rows for a in r['actions'])/sum(len(r['actions']) for r in rows))


def evaluate(args):
    if args.split=='test' and not args.final_test: raise ValueError('Final test requires --final-test after model selection')
    path=Path(args.out)
    if path.exists(): raise FileExistsError('Choose a new output path')
    data,sha=load(args.suite,args.split)
    policies={s:s for s in STRATEGIES}; hashes={}
    if args.checkpoint:
        import torch
        from recoverrl import load_model
        torch.set_num_threads(1)
        for ckpt in args.checkpoint:
            model=load_model(ckpt)
            if model.body[0].in_features!=6 or model.actor.out_features!=2:
                raise ValueError('Grasp benchmark requires a 6-input, 2-action checkpoint')
            policies[ckpt]=model; hashes[ckpt]=digest(Path(ckpt).read_bytes())
    results=[]; replays=[]
    for name,policy in policies.items():
        for mode in (('greedy',) if isinstance(policy,str) else ('greedy','sampled')):
            rows=[episode(row,policy,mode) for row in data['scenarios']]
            slices={s:summary([r for r in rows if r['slice']==s]) for s in dict.fromkeys(r['slice'] for r in rows)}
            total=summary(rows); results.append(dict(policy=name,mode=mode,summary=total,slices=slices,episodes=rows))
            print(f'{name} {mode}: {total["success_rate"]:.2%}, time/success {total["time_per_success"]:.2f}',flush=True)
        # Fixed first seed per slice, selected before any policy outcomes are known.
        seen=set()
        for row in data['scenarios']:
            if row['slice'] not in seen:
                replays.append(dict(policy=name,**episode(row,policy,record=True))); seen.add(row['slice'])
    payload=dict(version=VERSION,split=args.split,manifest_sha256=sha,checkpoint_sha256=hashes,
                 source_sha256={f:digest((ROOT/f).read_bytes()) for f in ('grasp_env.py','grasp_benchmark.py','ppo.py')},
                 results=results,replays=replays)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(canonical(payload))


def train(args):
    import torch
    from recoverrl import make_model,save_model
    from ppo import train_ppo
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    out=Path(args.out)
    if out.exists(): raise FileExistsError('Choose a new run directory')
    sampler=GraspSampler(args.suite)
    model=make_model(separate_value=True,input_dim=6,action_dim=2)
    xs,ys=[],[]
    # Teacher has the same observations as the learner; never uses hidden risk.
    for row in sampler.rows:
        env=GraspEnv(); obs=env.reset(row['seed'],row['risk'],row['slack'])
        while not env.done:
            a=strategy(obs,'switch-after-failure'); xs.append(obs);ys.append(a)
            obs,_,_,_=env.step(a)
    x=torch.tensor(xs); y=torch.tensor(ys)
    opt=torch.optim.Adam(list(model.body.parameters())+list(model.actor.parameters()),lr=3e-4)
    for _ in range(args.bc_epochs):
        for idx in torch.randperm(len(x)).split(128):
            loss=torch.nn.functional.cross_entropy(model(x[idx])[0],y[idx])
            opt.zero_grad();loss.backward();opt.step()
    out.mkdir(parents=True); save_model(model,out/'supervised.pt')
    (out/'demonstrations.json').write_bytes(canonical(dict(teacher='switch-after-failure',seed=args.seed,
        episodes=len(sampler.rows),transitions=len(xs),epochs=args.bc_epochs,manifest_sha256=sampler.sha256)))
    print(f'Supervised warm start: {len(xs)} demonstration transitions',flush=True)
    train_ppo(out/'supervised.pt',out,args.steps,args.seed,sampler_override=sampler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--suite',default=str(DEFAULT_SUITE))
    v=sub.add_parser('verify');v.add_argument('--suite',default=str(DEFAULT_SUITE))
    t=sub.add_parser('train');t.add_argument('--suite',default=str(DEFAULT_SUITE));t.add_argument('--out',required=True)
    t.add_argument('--seed',type=int,default=7);t.add_argument('--steps',type=int,default=100000);t.add_argument('--bc-epochs',type=int,default=30)
    e=sub.add_parser('evaluate');e.add_argument('--suite',default=str(DEFAULT_SUITE));e.add_argument('--out',required=True)
    e.add_argument('--checkpoint',action='append',default=[]);e.add_argument('--split',choices=['dev','test'],default='dev');e.add_argument('--final-test',action='store_true')
    x=sub.add_parser('export');x.add_argument('--input',required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.suite)
    elif a.command=='verify':
        sets=[]
        for s in ('train','dev','test'):
            data,sha=load(a.suite,s);sets.append({r['seed'] for r in data['scenarios']});print(s,len(data['scenarios']),sha)
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Split overlap')
    elif a.command=='train':
        if not 0<=a.seed<=8 or a.steps<1 or a.bc_epochs<1:p.error('Invalid training parameters')
        train(a)
    elif a.command=='evaluate':evaluate(a)
    else:
        data=json.loads(Path(a.input).read_text())
        if data['version']!=VERSION or data['split']!='dev':raise ValueError('Viewer requires a v1 grasp development report')
        (ROOT/'grasp-data.js').write_text('window.GRASP_DATA = '+json.dumps(data)+';\n')
        print('Open /grasp.html on the local server.')
