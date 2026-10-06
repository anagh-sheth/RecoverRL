"""Isolate how value-only updates affect policy logits in each architecture."""
import argparse
import json
from pathlib import Path
import torch
from recoverrl import load_model, RecoveryEnv, KINDS, heuristic
from ppo import independent_value_model


def diagnose(checkpoint):
    torch.set_num_threads(1)
    source=load_model(checkpoint)
    if hasattr(source,'value_body'):
        raise ValueError('Pass the original shared-network supervised checkpoint')
    xs,targets=[],[]
    for seed in range(100):
        env=RecoveryEnv(); obs=env.reset(seed,KINDS[seed%4]); trajectory=[]
        while not env.done:
            action=heuristic(obs); nxt,r,_,_=env.step(action)
            trajectory.append((obs,r)); obs=nxt
        total=0
        for obs,r in reversed(trajectory):
            total=r+.99*total; xs.append(obs); targets.append(total)
    x,y=torch.tensor(xs),torch.tensor(targets)
    with torch.no_grad(): before=source(x)[0].detach()
    report={}
    for name,model in [('shared',load_model(checkpoint)),('separate',independent_value_model(source))]:
        body=model.body if name=='shared' else model.value_body
        params=list(body.parameters())+list(model.critic.parameters())
        opt=torch.optim.Adam(params,lr=3e-4)
        for _ in range(32):
            loss=.5*(model(x)[1]-y).square().mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,.5); opt.step()
        with torch.no_grad():
            after=model(x)[0]
            report[name]=dict(greedy_action_changed_fraction=float((before.argmax(-1)!=after.argmax(-1)).float().mean()),
                mean_policy_kl=float(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=before),torch.distributions.Categorical(logits=after)).mean()),
                value_mse=float((model(x)[1]-y).square().mean()))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); result=diagnose(a.checkpoint)
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
