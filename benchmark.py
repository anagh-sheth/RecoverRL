"""Frozen RecoverRL v2 manifests, integrity checks and matched policy evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from failure_env import ENV_VERSION, FailureEnv, scripted_recovery_bound
from recoverrl import ACTIONS, FixedSequence, heuristic, wilson

SUITE_VERSION='recoverrl-composition-v1'
ROOT=Path(__file__).resolve().parent
DEFAULT_SUITE=ROOT/'benchmarks'/'composition-v1'


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def profiles(split):
    if split=='train': return [('clean',()),('miss',('miss',)),('drop',('drop',)),('shift',('shift',))]
    pairs=[('miss+drop',('miss','drop')),('miss+shift',('miss','shift')),('drop+shift',('drop','shift'))]
    if split=='dev': return profiles('train')+pairs
    return [('clean',())]+[(f'repeated-{x}',(x,)) for x in ('miss','drop','shift')]+pairs+[('miss+drop+shift',('miss','drop','shift'))]


def build_manifest(split):
    count=200 if split=='train' else 100
    base={'train':3000000,'dev':4000000,'test':5000000}[split]
    scenarios=[]
    for p,(name,families) in enumerate(profiles(split)):
        for i in range(count):
            config={'horizon':100}
            for j,family in enumerate(families):
                # Training/dev singles share distributions. Dev pairs compose these.
                # Final test adds repeated budgets and larger displacements.
                config[family]={'count':(2+i%3) if split=='test' else 1,
                                'probability':(.55,.8,1.)[(i+j)%3],
                                'after':1+i%2,'cooldown':2,'radius':(2+i%3) if split=='test' else 1+i%2}
            scenarios.append(dict(id=f'{split}-{name}-{i:03}',slice=name,seed=base+p*10000+i,
                                  required_types=list(families),config=config))
    return dict(version=SUITE_VERSION,environment=ENV_VERSION,split=split,
                selection='Fixed arithmetic enumeration; no learned-policy filtering.',scenarios=scenarios)


def freeze(directory):
    directory=Path(directory)
    # Never modify even a partly existing freeze directory.
    directory.mkdir(parents=True,exist_ok=False)
    hashes={}
    for split in ('train','dev','test'):
        data=canonical(build_manifest(split)); filename=split+'.json'
        (directory/filename).write_bytes(data); hashes[filename]=digest(data)
    lock=dict(version=SUITE_VERSION,environment=ENV_VERSION,files=hashes)
    (directory/'LOCK.json').write_bytes(canonical(lock))
    print(json.dumps(lock,indent=2))


def load_manifest(directory, split):
    directory=Path(directory)
    if split not in ('train','dev','test'): raise ValueError('Unknown split')
    lock=json.loads((directory/'LOCK.json').read_text())
    if lock['version']!=SUITE_VERSION or lock['environment']!=ENV_VERSION:
        raise ValueError('Incompatible suite/environment version')
    raw=(directory/(split+'.json')).read_bytes()
    if digest(raw)!=lock['files'][split+'.json']:
        raise ValueError('Frozen manifest hash mismatch; create a new suite version instead of editing')
    data=json.loads(raw)
    if data['split']!=split or data['environment']!=ENV_VERSION or data['version']!=SUITE_VERSION:
        raise ValueError('Invalid manifest metadata')
    ids=set(); seeds=set()
    for row in data['scenarios']:
        if row['id'] in ids or row['seed'] in seeds: raise ValueError('Duplicate scenario ID/seed')
        ids.add(row['id']); seeds.add(row['seed'])
        env=FailureEnv(); env.reset(row['seed'],row['config'])
        if scripted_recovery_bound(row['config'])>env.horizon:
            raise ValueError('Scenario exceeds the conservative scripted recoverability budget')
        enabled=sorted(k for k,v in env.faults.items() if v.count)
        if sorted(row['required_types'])!=enabled: raise ValueError('Required types do not match config')
    return data,lock['files'][split+'.json']


class ManifestSampler:
    """Training-only adapter: every sampled task comes from the locked train manifest."""
    def __init__(self,directory):
        self.data,self.sha256=load_manifest(directory,'train')
        self.rows=self.data['scenarios']
        if any(len(row['required_types'])>1 for row in self.rows):
            raise ValueError('Composition training must contain single failure families only')
        self.seen=set()

    def reset(self,env,seed):
        # Stable hash avoids sequential reset seeds accidentally aligning profiles.
        idx=int(digest(str(seed).encode())[:16],16)%len(self.rows)
        row=self.rows[idx]; self.seen.add(row['id'])
        return env.reset(row['seed'],row['config'])


def rollout(row, policy, mode='greedy', record=False):
    env=FailureEnv(); obs=env.reset(row['seed'],row['config'])
    if isinstance(policy,str):
        if policy=='scripted': act=heuristic
        elif policy=='fixed': act=FixedSequence(env)
        else: raise ValueError('Unknown baseline')
    else:
        import torch
        generator=torch.Generator().manual_seed(row['seed']+10000000)
        def act(obs):
            with torch.no_grad():
                logits=policy(torch.tensor(obs))[0]
                return int(logits.argmax()) if mode=='greedy' else int(torch.multinomial(logits.softmax(-1),1,generator=generator))
    frames=[env.frame()] if record else []
    reward=0
    while not env.done:
        action=act(obs); obs,r,_,_=env.step(action); reward+=r
        if record: frames.append(env.frame(ACTIONS[action]))
    types=sorted(k for k,v in env.counts.items() if v)
    return dict(id=row['id'],slice=row['slice'],seed=row['seed'],success=env.success,steps=env.steps,
                return_value=reward,requested_types=row['required_types'],realized_types=types,
                all_requested_triggered=set(row['required_types']).issubset(types),
                counts=env.counts,opportunities=env.opportunities,failure_log=env.failure_log,frames=frames)


def summarize(rows):
    n=len(rows); wins=sum(r['success'] for r in rows)
    failures=[r for r in rows if r['realized_types']]
    all_triggered=[r for r in rows if r['requested_types'] and r['all_requested_triggered']]
    combinations=Counter('+'.join(r['realized_types']) or 'none' for r in rows)
    return dict(episodes=n,success_rate=wins/n,ci95=wilson(wins,n),
        mean_steps=sum(r['steps'] for r in rows)/n,
        mean_success_steps=sum(r['steps'] for r in rows if r['success'])/wins if wins else None,
        fault_exposed_episodes=len(failures),
        recovery_rate=sum(r['success'] for r in failures)/len(failures) if failures else None,
        all_requested_triggered_episodes=len(all_triggered),
        all_requested_recovery_rate=sum(r['success'] for r in all_triggered)/len(all_triggered) if all_triggered else None,
        realized_combinations=dict(combinations))


def evaluate(args):
    if args.split=='test' and not args.final_test:
        raise ValueError('Final test is reserved. Use --final-test only after model selection is complete.')
    out=Path(args.out)
    if out.exists(): raise FileExistsError('Use a new output file to preserve earlier results')
    data,sha=load_manifest(args.suite,args.split)
    policies={'Fixed sequence':'fixed','Scripted recovery':'scripted'}
    checkpoint_hashes={}
    if args.checkpoint:
        import torch
        from recoverrl import load_model
        torch.set_num_threads(1)
        for checkpoint in args.checkpoint:
            if checkpoint in policies: raise ValueError('Duplicate policy')
            policies[checkpoint]=load_model(checkpoint)
            checkpoint_hashes[checkpoint]=digest(Path(checkpoint).read_bytes())
    results=[]; replays=[]
    for name,policy in policies.items():
        modes=('greedy',) if isinstance(policy,str) else ('greedy','sampled')
        for mode in modes:
            rows=[rollout(row,policy,mode) for row in data['scenarios']]
            slices={label:summarize([r for r in rows if r['slice']==label]) for label,_ in profiles(args.split)}
            total=summarize(rows)
            results.append(dict(policy=name,mode=mode,summary=total,slices=slices,episodes=rows))
            print(f'{name} [{mode}] {total["success_rate"]:.2%} / {len(rows)} episodes',flush=True)
        # First enumerated seed in every slice; never select favorable replays.
        for label,_ in profiles(args.split):
            row=next(r for r in data['scenarios'] if r['slice']==label)
            replays.append(dict(policy=name,mode='greedy',scenario=row,**rollout(row,policy,record=True)))
    payload=dict(suite=SUITE_VERSION,environment=ENV_VERSION,split=args.split,manifest_sha256=sha,
                 source_sha256={file:digest((ROOT/file).read_bytes()) for file in ('failure_env.py','recoverrl.py','benchmark.py')},
                 checkpoint_sha256=checkpoint_hashes,results=results,replays=replays,
                 notes='Configured combinations may not all trigger. Compare overall success and exposure counts; conditional rates can have selection bias.')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(canonical(payload))


def export_demo(path):
    data=json.loads(Path(path).read_text())
    if data['suite']!=SUITE_VERSION or data['environment']!=ENV_VERSION:
        raise ValueError('Unsupported report')
    (ROOT/'benchmark-data.js').write_text('window.BENCHMARK_DATA = '+json.dumps(data)+';\n')
    print('Open benchmark.html (or /benchmark.html on the local server).')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--suite',default=str(DEFAULT_SUITE))
    v=sub.add_parser('verify'); v.add_argument('--suite',default=str(DEFAULT_SUITE))
    e=sub.add_parser('evaluate'); e.add_argument('--suite',default=str(DEFAULT_SUITE))
    e.add_argument('--split',choices=['dev','test'],default='dev'); e.add_argument('--final-test',action='store_true')
    e.add_argument('--checkpoint',action='append',default=[]); e.add_argument('--out',required=True)
    x=sub.add_parser('export'); x.add_argument('--input',required=True)
    a=p.parse_args()
    if a.command=='freeze': freeze(a.suite)
    elif a.command=='verify':
        splits={}
        for split in ('train','dev','test'):
            manifest,sha=load_manifest(a.suite,split); splits[split]={r['seed'] for r in manifest['scenarios']}
            print(split,len(manifest['scenarios']),sha)
        if any(splits[a]&splits[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
            raise ValueError('Overlapping split seeds')
    elif a.command=='export': export_demo(a.input)
    else: evaluate(a)
