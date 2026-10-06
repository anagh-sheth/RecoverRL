"""PPO post-training with independent value features and bounded policy updates.

Start from an existing supervised checkpoint:
  python ppo.py --checkpoint runs/warmstart60/supervised.pt --out runs/stable7
Use --preset separated to isolate just the value-network separation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from recoverrl import GAMMA, KINDS, RecoveryEnv, load_model, make_model, save_model


def independent_value_model(source):
    """Copy all existing logits exactly; only value features gain independent weights."""
    model = make_model(separate_value=True)
    state = source.state_dict()
    if not any(k.startswith('value_body.') for k in state):
        state = dict(state)
        state.update({'value_body.'+k[5:]: v.clone() for k,v in state.items() if k.startswith('body.')})
    model.load_state_dict(state)
    return model


def gae_returns(rewards, dones, values, last, gamma=GAMMA, lam=.95):
    """Bootstrap unfinished rollouts, but never cross episode boundaries."""
    advantage = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        following = last if t == len(rewards)-1 else values[t+1]
        live = 1-dones[t]
        delta = rewards[t]+gamma*following*live-values[t]
        gae = delta+gamma*lam*live*gae
        advantage[t] = gae
    return advantage, advantage+values


class Collector:
    def __init__(self, seed, count=16, sampler=None):
        self.sampler=sampler
        if sampler:
            from failure_env import FailureEnv
            self.envs=[FailureEnv() for _ in range(count)]
        else:
            self.envs = [RecoveryEnv() for _ in range(count)]
        self.seed = 10000 + seed*100000
        self.recent = []
        self.obs = np.array([self.reset(e) for e in self.envs], dtype=np.float32)

    def reset(self, env):
        self.seed += 1
        if self.seed >= 1000000:
            raise ValueError('Training exhausted reserved seeds; cannot enter validation partition')
        return self.sampler.reset(env,self.seed) if self.sampler else env.reset(self.seed, KINDS[self.seed%4])

    def collect(self, model, horizon=128):
        observations, actions, logprobs, logits_all, rewards, dones, values = ([] for _ in range(7))
        for _ in range(horizon):
            with torch.no_grad():
                logits, value = model(torch.tensor(self.obs))
                dist = Categorical(logits=logits)
                action = dist.sample()
            observations.append(self.obs.copy())
            actions.append(action.numpy())
            logprobs.append(dist.log_prob(action).numpy())
            logits_all.append(logits.numpy())
            values.append(value.numpy())
            following, rs, ds = [], [], []
            for env,a in zip(self.envs,action.tolist()):
                o,r,d,info = env.step(a)
                rs.append(r); ds.append(d)
                if d:
                    self.recent.append(int(info['success']))
                    o=self.reset(env)
                following.append(o)
            self.recent=self.recent[-100:]
            self.obs=np.array(following,dtype=np.float32)
            rewards.append(rs); dones.append(ds)
        with torch.no_grad(): last=model(torch.tensor(self.obs))[1].numpy()
        adv,ret=gae_returns(np.array(rewards),np.array(dones),np.array(values),last)
        tensor=lambda x: torch.tensor(np.array(x),dtype=torch.float32)
        return dict(obs=tensor(observations).reshape(-1,10),
                    actions=torch.tensor(np.array(actions).flatten()),
                    logprobs=tensor(logprobs).flatten(), logits=tensor(logits_all).reshape(-1,7),
                    advantages=tensor(adv).flatten(), returns=tensor(ret).flatten())


def train_ppo(checkpoint, out, steps=100000, seed=7, preset='stable', suite=None):
    if not 0 <= seed <= 8:
        raise ValueError('Training seed must be 0–8 to preserve reserved evaluation ranges')
    if steps < 1:
        raise ValueError('Steps must be positive')
    sampler=None
    if suite:
        from benchmark import ManifestSampler
        sampler=ManifestSampler(suite)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    checkpoint, out = Path(checkpoint), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'ppo.pt').exists():
        raise FileExistsError('Output already has a PPO checkpoint; choose a new run directory')
    source=load_model(checkpoint)
    # Preserve the actual starting model alongside the post-trained checkpoint.
    if checkpoint.resolve() != (out/'supervised.pt').resolve():
        save_model(source,out/'supervised.pt')
    model=independent_value_model(source)
    actor_parameters=list(model.body.parameters())+list(model.actor.parameters())
    value_parameters=list(model.value_body.parameters())+list(model.critic.parameters())
    stable=preset=='stable'
    config=dict(preset=preset,seed=seed,requested_steps=steps,actor_lr=3e-5 if stable else 3e-4,
                value_lr=3e-4,entropy_coef=.001 if stable else .01,target_kl=.01 if stable else None,
                gamma=GAMMA,gae_lambda=.95,clip=.2,epochs=4,batch_size=256,
                num_envs=16,rollout_length=128,value_warmup_steps=0,value_warmup_episodes=256 if stable else 0,
                torch_version=torch.__version__,numpy_version=np.__version__,
                checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if sampler:
        config.update(environment=sampler.data['environment'],suite=str(suite),training_manifest_sha256=sampler.sha256)
        (out/'training_manifest.json').write_text(json.dumps(sampler.data,sort_keys=True,indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    actor_opt=torch.optim.Adam(actor_parameters,lr=config['actor_lr'])
    value_opt=torch.optim.Adam(value_parameters,lr=config['value_lr'])
    collector=Collector(seed,sampler=sampler)
    start=time.monotonic()
    if stable:
        # Fit the random critic to full discounted returns from frozen-policy data.
        # lambda=1 targets are recomputed here with complete, finite episodes.
        observations, targets=[],[]
        for s in range(500000+seed*1000,500000+seed*1000+256):
            if sampler:
                from failure_env import FailureEnv
                env=FailureEnv(); obs=sampler.reset(env,s)
            else:
                env=RecoveryEnv(); obs=env.reset(s,KINDS[s%4])
            trajectory=[]
            while not env.done:
                with torch.no_grad():
                    action=int(Categorical(logits=model(torch.tensor(obs))[0]).sample())
                nxt,reward,_,_=env.step(action); trajectory.append((obs,reward)); obs=nxt
            total=0
            for o,r in reversed(trajectory):
                total=r+GAMMA*total; observations.append(o); targets.append(total)
        config['value_warmup_steps']=len(observations)
        ox=torch.tensor(observations); target=torch.tensor(targets)
        for _ in range(20):
            for idx in torch.randperm(len(ox)).split(256):
                value=model(ox[idx])[1]
                loss=(value-target[idx]).square().mean()
                value_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(value_parameters,.5); value_opt.step()
        print(f'Critic warmup: {len(observations)} transitions; actor untouched',flush=True)
    logs=[]
    for update in range(math.ceil(steps/2048)):
        batch=collector.collect(model)
        adv=batch['advantages']; adv=(adv-adv.mean())/(adv.std()+1e-8)
        old_dist=Categorical(logits=batch['logits'])
        stopped=False; actor_updates=0
        for epoch in range(4):
            for idx in torch.randperm(len(adv)).split(256):
                logits,value=model(batch['obs'][idx]); dist=Categorical(logits=logits)
                # Value updates cannot modify actor features.
                value_loss=.5*(value-batch['returns'][idx]).square().mean()
                value_opt.zero_grad(); value_loss.backward(); torch.nn.utils.clip_grad_norm_(value_parameters,.5); value_opt.step()
                if stopped: continue
                ratio=(dist.log_prob(batch['actions'][idx])-batch['logprobs'][idx]).exp()
                surrogate=torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx])
                actor_loss=-surrogate.mean()-config['entropy_coef']*dist.entropy().mean()
                actor_opt.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(actor_parameters,.5); actor_opt.step()
                actor_updates+=1
                if stable:
                    with torch.no_grad():
                        current=Categorical(logits=model(batch['obs'])[0])
                        drift=float(kl_divergence(old_dist,current).mean())
                    if drift > config['target_kl']: stopped=True
        with torch.no_grad():
            logits,value=model(batch['obs']); current=Categorical(logits=logits)
            drift=float(kl_divergence(old_dist,current).mean())
            ratio=(current.log_prob(batch['actions'])-batch['logprobs']).exp()
            variance=batch['returns'].var(unbiased=False)
            explained=1-(batch['returns']-value).var(unbiased=False)/variance.clamp_min(1e-8)
        row=dict(steps=(update+1)*2048,training_success=sum(collector.recent)/max(1,len(collector.recent)),
                 kl=drift,clip_fraction=float(((ratio-1).abs()>.2).float().mean()),
                 entropy=float(current.entropy().mean()),value_loss=float(value_loss.detach()),
                 explained_variance=float(explained),actor_updates=actor_updates,kl_stopped=stopped)
        logs.append(row)
        if update%5==0 or update==math.ceil(steps/2048)-1: print(json.dumps(row),flush=True)
    save_model(model,out/'ppo.pt')
    initial=source.state_dict()
    config['actor_weight_delta_l2']=float(sum((v-initial[k]).square().sum() for k,v in model.state_dict().items() if k.startswith(('body.','actor.'))).sqrt())
    config.update(actual_steps=logs[-1]['steps'],total_environment_steps=logs[-1]['steps']+config['value_warmup_steps'],elapsed_seconds=time.monotonic()-start)
    if sampler: config['unique_training_scenarios_seen']=len(sampler.seen)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'training.json').write_text(json.dumps({'config':config,'history':logs},indent=2))
    return model


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=100000); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--preset',choices=['stable','separated'],default='stable')
    p.add_argument('--suite',help='Frozen composition suite directory; only its train split is sampled')
    a=p.parse_args(); train_ppo(a.checkpoint,a.out,a.steps,a.seed,a.preset,a.suite)
