"""Two-action, semi-Markov grasp choice with hidden persistent risk and deadlines.

Fixed controllers execute reach and delivery. A policy chooses grasp style only.
V1 and v2 grid environments remain unchanged.
"""
import hashlib
import math
import random

VERSION='grasp-deadline-v1'
ACTIONS=('fast','secure')
GRASP_COST=(1,4)
SECURE_RISK_MULTIPLIER=.1
TIME_COST=.01


def distance(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])


class GraspEnv:
    def reset(self,seed=0,risk=.5,slack=3):
        if not isinstance(risk,(int,float)) or not math.isfinite(risk) or not 0<=risk<=1:
            raise ValueError('risk must be in [0,1]')
        if type(slack) is not int or not 0<=slack<=16: raise ValueError('slack must be an integer in [0,16]')
        self.seed,self.risk=seed,float(risk)
        rng=random.Random(seed)
        self.hand,self.obj,self.target=[list(p) for p in rng.sample([(x,y) for x in range(5) for y in range(5)],3)]
        self.deadline=distance(self.hand,self.obj)+distance(self.obj,self.target)+2+slack
        self.elapsed=0; self.attempts=0; self.failures=[0,0]
        self.done=self.success=self.holding=False
        self.event='Ready'; self.last_action=None
        return self.obs()

    def transport_cost(self):
        return distance(self.hand,self.obj)+distance(self.obj,self.target)+1

    def obs(self):
        remaining=self.deadline-self.elapsed
        route=self.transport_cost()
        # Both counts are observable interaction history, not privileged risk labels.
        return [remaining/40,route/20,(remaining-route)/20,
                self.failures[0]/20,self.failures[1]/20,self.attempts/20]

    def uniform(self):
        # Same draw at the same attempt index for all strategies. No risk in RNG key.
        key=f'{VERSION}:{self.seed}:{self.attempts}'.encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8],'big')).random()

    def step(self,action):
        if self.done: raise RuntimeError('Reset before another action')
        if type(action) is not int or action not in (0,1): raise ValueError('Action must be fast=0 or secure=1')
        self.last_action=ACTIONS[action]
        start=self.elapsed; remaining=self.deadline-start
        approach=distance(self.hand,self.obj)
        grip_time=GRASP_COST[action]
        self.attempts+=1
        if approach+grip_time>remaining:
            self.elapsed=self.deadline; self.done=True
            self.event='Deadline reached before grasp completed'
        else:
            self.elapsed+=approach+grip_time; self.hand=self.obj.copy()
            probability=self.risk*(SECURE_RISK_MULTIPLIER if action else 1.)
            if self.uniform()<probability:
                self.failures[action]+=1; self.event=f'{ACTIONS[action].title()} grasp slipped'
                self.done=self.elapsed>=self.deadline
            else:
                delivery=distance(self.obj,self.target)+1
                if self.elapsed+delivery<=self.deadline:
                    self.elapsed+=delivery; self.obj=self.target.copy(); self.hand=self.target.copy()
                    self.success=self.done=True; self.event='Delivered before deadline'
                else:
                    self.elapsed=self.deadline; self.done=True; self.holding=True
                    self.event='Grasp succeeded, but delivery missed deadline'
        duration=self.elapsed-start
        reward=float(self.success)-TIME_COST*duration
        return self.obs(),reward,self.done,dict(success=self.success,duration=duration,elapsed=self.elapsed,
                                               deadline=self.deadline,failures=self.failures.copy())

    def frame(self):
        # Debug/replay-only ground truth. This dictionary is never passed to the actor.
        return dict(hand=self.hand.copy(),obj=self.obj.copy(),target=self.target.copy(),
                    elapsed=self.elapsed,deadline=self.deadline,attempts=self.attempts,
                    failures=self.failures.copy(),holding=self.holding,success=self.success,
                    event=self.event,action=self.last_action,hidden_fast_risk=self.risk)


def strategy(obs,name):
    budget=obs[2]*20
    failed=obs[3]*20+obs[4]*20
    if name=='always-fast': return 0
    if name=='always-secure': return 1
    if name=='switch-after-failure': return int(failed>.5 and budget>=4-1e-6)
    if name=='secure-when-feasible': return int(budget>=4-1e-6)
    raise ValueError('Unknown strategy')
