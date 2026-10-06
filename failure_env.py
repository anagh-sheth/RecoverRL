"""RecoverRL v2: bounded, composable, reproducible failure injection.

This remains a discrete grid abstraction. V1 is untouched in recoverrl.py.
Fault random streams are keyed by seed, type and eligible opportunity so that
extra inspections do not arbitrarily consume the randomness for other faults.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import random

from recoverrl import ACTIONS, GAMMA, RecoveryEnv

ENV_VERSION = 'recovery-grid-v2.0'


@dataclass(frozen=True)
class Fault:
    count: int = 0
    probability: float = 1.
    after: int = 1
    cooldown: int = 1
    radius: int = 1

    def __post_init__(self):
        for name in ('count','after','cooldown','radius'):
            if type(getattr(self,name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if not 0 <= self.count <= 8: raise ValueError('count must be 0–8')
        if not 0 <= self.probability <= 1: raise ValueError('probability must be 0–1')
        if self.after < 1 or self.cooldown < 1: raise ValueError('after/cooldown must be positive')
        if not 1 <= self.radius <= 8: raise ValueError('radius must be 1–8')


def scripted_recovery_bound(config):
    """Conservative action bound for the inspect/retry heuristic on the 5×5 grid.

    Clean transport <=18; each miss adds <=2 actions, each stale-location
    shift <=radius+2, and each drop <=2*radius+2. All cells are reachable.
    This is a bound for this heuristic/geometry, not arbitrary learned policies.
    """
    faults={name:Fault(**config.get(name,{})) for name in ('miss','drop','shift')}
    return (18+2*faults['miss'].count
            +(faults['shift'].radius+2)*faults['shift'].count
            +(2*faults['drop'].radius+2)*faults['drop'].count)


class FailureEnv(RecoveryEnv):
    def reset(self, seed=0, config=None):
        config = config or {}
        if set(config)-{'miss','drop','shift','horizon'}:
            raise ValueError('Unknown failure configuration field')
        self.horizon=config.get('horizon',100)
        if type(self.horizon) is not int or not 1 <= self.horizon <= 500:
            raise ValueError('horizon must be an integer in 1–500')
        self.faults={name:Fault(**config.get(name,{})) for name in ('miss','drop','shift')}
        self.opportunities={name:0 for name in self.faults}
        self.counts={name:0 for name in self.faults}
        self.last_trigger={name:-10000 for name in self.faults}
        self.failure_log=[]
        super().reset(seed,'clean')
        return self.obs()

    def obs(self):
        observation=super().obs()
        observation[-1]=self.steps/self.horizon
        return observation

    def fault_rng(self, name, opportunity):
        key=f'{ENV_VERSION}:{self.seed}:{name}:{opportunity}'.encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8],'big'))

    def trigger(self, name):
        fault=self.faults[name]
        self.opportunities[name]+=1
        opportunity=self.opportunities[name]
        if self.counts[name]>=fault.count or opportunity<fault.after:
            return False
        if opportunity-self.last_trigger[name]<fault.cooldown:
            return False
        if self.fault_rng(name,opportunity).random()>=fault.probability:
            return False
        self.counts[name]+=1
        self.last_trigger[name]=opportunity
        self.failure_log.append(dict(type=name,step=self.steps,opportunity=opportunity))
        self.injected=True
        return True

    def relocate(self, name):
        radius=self.faults[name].radius
        # No teleport onto the target; remain on a connected, reachable board.
        choices=[[x,y] for x in range(5) for y in range(5)
                 if 0<abs(x-self.obj[0])+abs(y-self.obj[1])<=radius and [x,y]!=self.target]
        rng=self.fault_rng(name,self.opportunities[name])
        rng.random()  # Same first draw as the trigger decision; second picks location.
        self.obj=rng.choice(choices)

    def step(self, action):
        if self.done: raise RuntimeError('Reset before stepping a completed episode')
        if type(action) is not int or action not in range(7): raise ValueError('Unknown action')
        previous=self.potential()
        self.steps+=1; self.age+=1; self.event=ACTIONS[action]
        invalid=False
        if action==0:
            self.seen=self.obj.copy(); self.age=0; self.bad_grasp=0
        elif action in (1,2,3,4):
            dx,dy={1:(0,-1),2:(0,1),3:(-1,0),4:(1,0)}[action]
            old=self.hand.copy()
            self.hand=[max(0,min(4,old[0]+dx)),max(0,min(4,old[1]+dy))]
            invalid=old==self.hand
            if self.holding:
                self.obj=self.hand.copy(); self.seen=self.obj.copy()
                if not invalid and self.trigger('drop'):
                    self.holding=False; self.relocate('drop'); self.bad_grasp=1
                    self.event='Object slipped; holding sensor is empty'
            elif not invalid and self.trigger('shift'):
                self.relocate('shift')
                self.event='Object displaced (hidden from policy)'
        elif action==5:
            if self.holding: invalid=True
            elif self.hand!=self.obj:
                self.bad_grasp=1; self.event='Grasp empty'
            elif self.trigger('miss'):
                self.bad_grasp=1; self.event='Grasp missed'
            else:
                self.holding=True; self.bad_grasp=0; self.event='Object grasped'
        else:
            if not self.holding: invalid=True
            else:
                self.holding=False; self.obj=self.hand.copy(); self.seen=self.obj.copy()
                self.success=self.obj==self.target
                self.event='Placement verified' if self.success else 'Released outside target'
        self.done=self.success or self.steps>=self.horizon
        reward=(10 if self.success else -2 if self.done else 0)-.03-.08*invalid
        reward+=GAMMA*(0 if self.done else self.potential())-previous
        return self.obs(),reward,self.done,dict(success=self.success,injected=self.injected,
             failure_counts=self.counts.copy(),opportunities=self.opportunities.copy())

    def frame(self, action=None):
        return {**super().frame(action),'failure_counts':self.counts.copy()}

    def configuration(self):
        return {'horizon':self.horizon,**{k:asdict(v) for k,v in self.faults.items()}}
