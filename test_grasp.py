import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from grasp_env import GraspEnv,strategy
from grasp_benchmark import build,episode,freeze,load,GraspSampler


class GraspTests(unittest.TestCase):
    def test_hidden_risk_and_repeatability(self):
        a=GraspEnv();b=GraspEnv()
        self.assertEqual(a.reset(42,.05,6),b.reset(42,.95,6))
        row=dict(id='x',slice='x',seed=42,risk=.75,slack=6)
        self.assertEqual(episode(row,'switch-after-failure',record=True),episode(row,'switch-after-failure',record=True))

    def test_secure_costs_more_time(self):
        a=GraspEnv();b=GraspEnv();a.reset(42,0,6);b.reset(42,0,6)
        a.step(0);b.step(1)
        self.assertTrue(a.success and b.success)
        self.assertEqual(b.elapsed-a.elapsed,3)

    def test_secure_can_miss_deadline_despite_successful_grasp(self):
        a=GraspEnv();b=GraspEnv();a.reset(42,0,0);b.reset(42,0,0)
        a.step(0);b.step(1)
        self.assertTrue(a.success);self.assertFalse(b.success);self.assertTrue(b.done)
        self.assertEqual(b.elapsed,b.deadline)

    def test_risk_does_not_expire_after_failures(self):
        env=GraspEnv();env.reset(42,1,10)
        while not env.done: env.step(0)
        self.assertFalse(env.success);self.assertGreater(env.failures[0],2)
        self.assertEqual(env.risk,1)

    def test_secure_is_more_reliable_when_both_fit(self):
        fast=secure=0
        for seed in range(100):
            a=GraspEnv();b=GraspEnv();a.reset(seed,.8,6);b.reset(seed,.8,6)
            a.step(0);b.step(1)
            fast+=a.success;secure+=b.success
            if a.success:self.assertTrue(b.success)
        self.assertGreater(secure,fast)

    def test_reward_charges_physical_time_and_late_delivery_never_scores(self):
        env=GraspEnv();env.reset(42,0,0)
        _,reward,done,_=env.step(1)
        self.assertTrue(done);self.assertAlmostEqual(reward,-.01*env.elapsed)
        with self.assertRaises(RuntimeError):env.step(0)
        with self.assertRaises(ValueError):GraspEnv().reset(0,float('nan'))

    def test_switch_rule_uses_observed_failures_and_feasibility(self):
        env=GraspEnv();obs=env.reset(42,1,10)
        self.assertEqual(strategy(obs,'switch-after-failure'),0)
        obs,_,_,_=env.step(0)
        self.assertEqual(strategy(obs,'switch-after-failure'),1)


class GraspSuiteTests(unittest.TestCase):
    def test_risk_and_seed_holdout(self):
        data=[build(s) for s in ('train','dev','test')]
        for i in range(3):
            for j in range(i):
                self.assertFalse({r['seed'] for r in data[i]['scenarios']}&{r['seed'] for r in data[j]['scenarios']})
                self.assertFalse({r['risk'] for r in data[i]['scenarios']}&{r['risk'] for r in data[j]['scenarios']})

    def test_training_only_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'suite'
            with contextlib.redirect_stdout(io.StringIO()):freeze(root)
            sampler=GraspSampler(root);env=sampler.make_env()
            train,_=load(root,'train');allowed={r['seed'] for r in train['scenarios']}
            for seed in range(100):sampler.reset(env,seed);self.assertIn(env.seed,allowed)
            with (root/'train.json').open('a') as f:f.write(' ')
            with self.assertRaises(ValueError):load(root,'train')

    def test_dynamic_model_checkpoint_and_rollout_shapes(self):
        import torch
        from recoverrl import make_model,load_model,save_model
        from ppo import Collector,independent_value_model
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'suite'
            with contextlib.redirect_stdout(io.StringIO()):freeze(root)
            sampler=GraspSampler(root)
            model=make_model(input_dim=6,action_dim=2)
            model=independent_value_model(model)
            path=Path(tmp)/'model.pt';save_model(model,path)
            restored=load_model(path);obs=torch.zeros(1,6)
            self.assertTrue(torch.equal(model(obs)[0],restored(obs)[0]))
            batch=Collector(7,count=2,sampler=sampler).collect(model,horizon=4)
            self.assertEqual(tuple(batch['obs'].shape),(8,6))
            self.assertEqual(tuple(batch['logits'].shape),(8,2))


if __name__=='__main__':unittest.main()
