"""RecoverRL Lite: seeded 2-D recovery environment, PPO, evaluation and replays."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

ACTIONS = ['inspect', 'up', 'down', 'left', 'right', 'grasp', 'release']
KINDS = ['clean', 'miss', 'drop', 'shift']
SIZE, HORIZON, GAMMA = 5, 50, .99


class RecoveryEnv:
    """Discrete grid abstraction, not a physics simulator. Observation excludes fault labels."""
    def reset(self, seed=0, kind='clean'):
        self.rng = random.Random(seed)
        self.seed, self.kind = seed, kind
        cells = self.rng.sample([(x, y) for x in range(SIZE) for y in range(SIZE)], 3)
        self.hand, self.obj, self.target = map(list, cells)
        self.seen = self.obj.copy()
        self.holding = self.success = self.done = self.injected = False
        self.age = self.steps = self.bad_grasp = 0
        self.event = 'Episode started'
        return self.obs()

    def obs(self):
        # Relative coordinates help the policy generalize across translated scenes.
        h, o, t = self.hand, self.seen, self.target
        return [h[0]/4, h[1]/4, (o[0]-h[0])/4, (o[1]-h[1])/4,
                (t[0]-h[0])/4, (t[1]-h[1])/4, float(self.holding),
                min(self.age, 10)/10, float(self.bad_grasp), self.steps/HORIZON]

    def potential(self):
        dist = lambda a, b: abs(a[0]-b[0])+abs(a[1]-b[1])
        if self.success:
            return 0.
        remaining = dist(self.hand, self.target)+1 if self.holding else dist(self.hand, self.obj)+dist(self.obj, self.target)+2
        return -.15*remaining

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a completed episode')
        if action not in range(len(ACTIONS)):
            raise ValueError('Unknown action')
        previous = self.potential()
        self.steps += 1
        self.age += 1
        self.event = ACTIONS[action]
        invalid = False
        if action == 0:
            self.seen = self.obj.copy()
            self.age = 0
            self.bad_grasp = 0
        elif action in (1, 2, 3, 4):
            dx, dy = {1:(0,-1), 2:(0,1), 3:(-1,0), 4:(1,0)}[action]
            old = self.hand.copy()
            self.hand = [max(0,min(4,old[0]+dx)), max(0,min(4,old[1]+dy))]
            invalid = old == self.hand
            if self.holding:
                self.obj = self.hand.copy()
                self.seen = self.obj.copy()
                if self.kind == 'drop' and not self.injected and not invalid:
                    self.holding = False
                    self.obj = [max(0,min(4,self.hand[0]+self.rng.choice([-1,1]))), self.hand[1]]
                    self.injected = True
                    self.bad_grasp = 1
                    self.event = 'Object slipped; holding sensor is empty'
            elif self.kind == 'shift' and not self.injected and not invalid:
                alternatives = [[x,y] for x in range(5) for y in range(5) if [x,y] != self.obj and [x,y] != self.target]
                self.obj = self.rng.choice(alternatives)
                self.injected = True
                self.event = 'Object displaced (hidden from policy)'
        elif action == 5:
            if self.holding:
                invalid = True
            elif self.hand != self.obj:
                self.bad_grasp = 1
                self.event = 'Grasp empty'
            elif self.kind == 'miss' and not self.injected:
                self.injected = True
                self.bad_grasp = 1
                self.event = 'Grasp missed'
            else:
                self.holding = True
                self.bad_grasp = 0
                self.event = 'Object grasped'
        elif action == 6:
            if not self.holding:
                invalid = True
            else:
                self.holding = False
                self.obj = self.hand.copy()
                self.seen = self.obj.copy()
                self.success = self.obj == self.target
                self.event = 'Placement verified' if self.success else 'Released outside target'
        self.done = self.success or self.steps >= HORIZON
        # Finite-horizon MDP: terminal potential is zero, including failure at horizon.
        shaping = GAMMA*(0 if self.done else self.potential())-previous
        reward = (10 if self.success else -2 if self.done else 0)-.03-.08*invalid+shaping
        return self.obs(), reward, self.done, {'success': self.success, 'injected': self.injected}

    def frame(self, action=None):
        # Ground truth is available only to the replay renderer/evaluator, not the policy.
        return dict(hand=self.hand.copy(), obj=self.obj.copy(), seen=self.seen.copy(),
                    target=self.target.copy(), holding=self.holding, success=self.success,
                    step=self.steps, event=self.event, action=action, injected=self.injected)


def heuristic(obs):
    _, _, ox, oy, tx, ty, holding, age, failed, _ = obs
    if not holding and failed:
        return 0
    dx, dy = (tx,ty) if holding else (ox,oy)
    if dx: return 4 if dx > 0 else 3
    if dy: return 2 if dy > 0 else 1
    return 6 if holding else 5


class FixedSequence:
    def __init__(self, env):
        self.actions = []
        at = env.hand.copy()
        for dest, terminal in [(env.seen,5),(env.target,6)]:
            for axis, neg, pos in [(0,3,4),(1,1,2)]:
                while at[axis] != dest[axis]:
                    self.actions.append(pos if at[axis] < dest[axis] else neg)
                    at[axis] += 1 if at[axis] < dest[axis] else -1
            self.actions.append(terminal)
    def __call__(self, obs):
        return self.actions.pop(0) if self.actions else 6


def make_model():
    import torch
    from torch import nn
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(10,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())
            self.actor, self.critic = nn.Linear(64,7), nn.Linear(64,1)
        def forward(self, x):
            z = self.body(x)
            return self.actor(z), self.critic(z).squeeze(-1)
    return Policy()


def save_model(model, path):
    import torch
    torch.save(model.state_dict(), path)


def load_model(path):
    import torch
    model = make_model()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def model_action(model, obs):
    import torch
    with torch.no_grad():
        return int(model(torch.tensor(obs, dtype=torch.float32))[0].argmax())


def train(args):
    import numpy as np
    import torch
    from torch.distributions import Categorical
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Seed ranges are disjoint from validation (1M+) and final test (2M+).
    xs, ys = [], []
    env = RecoveryEnv()
    for seed in range(1500):
        obs = env.reset(seed, KINDS[seed%4])
        while not env.done:
            action = heuristic(obs)
            xs.append(obs); ys.append(action)
            obs, _, _, _ = env.step(action)
    x, y = torch.tensor(xs), torch.tensor(ys)
    for _ in range(args.bc_epochs):
        for idx in torch.randperm(len(x)).split(256):
            loss = torch.nn.functional.cross_entropy(model(x[idx])[0], y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    save_model(model, out/'supervised.pt')
    print(f'Supervised warm start: {len(xs)} demonstration actions', flush=True)
    n, horizon = 16, 128
    envs = [RecoveryEnv() for _ in range(n)]
    next_seed = 10000 + args.seed*100000
    def reset(e):
        nonlocal next_seed
        next_seed += 1
        return e.reset(next_seed, KINDS[next_seed%4])
    obs = np.array([reset(e) for e in envs], dtype=np.float32)
    logs, total, recent = [], 0, []
    updates = math.ceil(args.steps/(n*horizon))
    for update in range(updates):
        ob, ac, lp, rw, dn, va = [], [], [], [], [], []
        for _ in range(horizon):
            with torch.no_grad():
                logits, value = model(torch.tensor(obs))
                dist = Categorical(logits=logits); action = dist.sample()
            ob.append(obs.copy()); ac.append(action.numpy()); lp.append(dist.log_prob(action).numpy()); va.append(value.numpy())
            rewards, dones, following = [], [], []
            for e,a in zip(envs, action.tolist()):
                o,r,d,info = e.step(a)
                rewards.append(r); dones.append(d)
                if d:
                    recent.append(int(info['success']))
                    o=reset(e)
                following.append(o)
            rw.append(rewards); dn.append(dones)
            obs = np.asarray(following, dtype=np.float32)
        with torch.no_grad(): last = model(torch.tensor(obs))[1].numpy()
        rewards, dones, values = map(np.asarray, (rw,dn,va))
        adv = np.zeros_like(rewards, dtype=np.float32); gae = np.zeros(n)
        for t in reversed(range(horizon)):
            nv = last if t == horizon-1 else values[t+1]
            live = 1-dones[t]
            delta = rewards[t]+GAMMA*nv*live-values[t]
            gae = delta+GAMMA*.95*live*gae; adv[t]=gae
        flatobs = torch.tensor(np.array(ob).reshape(-1,10))
        actions = torch.tensor(np.array(ac).flatten())
        oldlp = torch.tensor(np.array(lp).flatten())
        returns = torch.tensor((adv+values).flatten(), dtype=torch.float32)
        advantage = torch.tensor(adv.flatten()); advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(actions)).split(256):
                logits, value = model(flatobs[idx]); dist=Categorical(logits=logits)
                ratio=(dist.log_prob(actions[idx])-oldlp[idx]).exp()
                policy=-torch.minimum(ratio*advantage[idx], ratio.clamp(.8,1.2)*advantage[idx]).mean()
                loss=policy+.5*(value-returns[idx]).square().mean()-.01*dist.entropy().mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.5); opt.step()
        total += n*horizon
        row={'steps':total,'recent_training_success':sum(recent[-100:])/max(1,len(recent[-100:])), 'loss':float(loss.detach())}
        logs.append(row)
        if update%5 == 0 or update == updates-1: print(json.dumps(row), flush=True)
    save_model(model,out/'ppo.pt')
    (out/'training.json').write_text(json.dumps({'seed':args.seed,'requested_steps':args.steps,'actual_steps':total,'bc_epochs':args.bc_epochs,'history':logs},indent=2))


def episode(seed, kind, policy, record=False):
    env=RecoveryEnv(); obs=env.reset(seed,kind)
    if policy == 'fixed': fn=FixedSequence(env)
    elif policy == 'heuristic': fn=heuristic
    else: fn=lambda o: model_action(policy,o)
    frames=[env.frame()] if record else []
    while not env.done:
        a=fn(obs); obs,_,_,_=env.step(a)
        if record: frames.append(env.frame(ACTIONS[a]))
    return {'success':env.success,'steps':env.steps,'injected':env.injected,'frames':frames}


def wilson(k,n):
    if not n: return [0,0]
    z=1.96; p=k/n; d=1+z*z/n
    mid=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [mid-half,mid+half]


def evaluate(args):
    import torch
    torch.set_num_threads(1)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    policies={'Fixed sequence':'fixed','Scripted recovery':'heuristic'}
    for label,file in [('Supervised','supervised.pt'),('Supervised + PPO','ppo.pt')]:
        if (out/file).exists(): policies[label]=load_model(out/file)
    results=[]
    for label,policy in policies.items():
        for kind in KINDS:
            rows=[episode(args.start_seed+i,kind,policy) for i in range(args.episodes)]
            successes=sum(r['success'] for r in rows)
            injected=[r for r in rows if r['injected']]
            result={'policy':label,'kind':kind,'episodes':len(rows),'success_rate':successes/len(rows),
                    'success_ci95':wilson(successes,len(rows)),
                    'mean_success_steps':sum(r['steps'] for r in rows if r['success'])/successes if successes else None,
                    'injected_episodes':len(injected),'recovery_success':sum(r['success'] for r in injected)/len(injected) if injected else None}
            results.append(result); print(f"{label:20} {kind:6} {successes}/{len(rows)}",flush=True)
    replays=[]
    # Fixed, published demo seeds: never search for a flattering policy outcome.
    for kind in KINDS:
        for seed in range(args.start_seed,args.start_seed+5):
            replays.append({'kind':kind,'seed':seed,'policies':{label:episode(seed,kind,p,True) for label,p in policies.items()}})
    payload={'results':results,'replays':replays,'start_seed':args.start_seed,'episodes_per_slice':args.episodes}
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2))
    # External JS lets the replay also open directly from disk without fetch/CORS.
    (Path(__file__).parent/'demo-data.js').write_text('window.RECOVERRL_DATA = '+json.dumps(payload)+';')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    tr=subs.add_parser('train'); tr.add_argument('--steps',type=int,default=100000); tr.add_argument('--seed',type=int,default=7); tr.add_argument('--bc-epochs',type=int,default=60); tr.add_argument('--out',default='runs/default')
    ev=subs.add_parser('evaluate'); ev.add_argument('--episodes',type=int,default=100); ev.add_argument('--start-seed',type=int,default=1000000); ev.add_argument('--out',default='runs/default')
    args=parser.parse_args()
    if args.command=='evaluate' and args.episodes < 1: parser.error('--episodes must be positive')
    if args.command=='train' and (args.steps < 1 or args.bc_epochs < 0): parser.error('Invalid training length')
    (train if args.command=='train' else evaluate)(args)


if __name__=='__main__': main()
