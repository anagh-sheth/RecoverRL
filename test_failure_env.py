import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmark import build_manifest, freeze, load_manifest, ManifestSampler, rollout, summarize
from failure_env import FailureEnv, Fault, scripted_recovery_bound
from recoverrl import RecoveryEnv, heuristic


class FailureEnvironmentTests(unittest.TestCase):
    def test_v1_clean_compatibility(self):
        for seed in range(20):
            v1=RecoveryEnv(); v2=FailureEnv()
            o=v1.reset(seed); self.assertEqual(o,v2.reset(seed,{'horizon':50}))
            while not v1.done:
                action=heuristic(o)
                a=v1.step(action); b=v2.step(action)
                self.assertEqual(a[:3],b[:3]); o=a[0]

    def test_repeatable_combination_and_budgets(self):
        config={x:{'count':2,'probability':1.,'cooldown':2} for x in ('miss','drop','shift')}
        row=dict(id='synthetic',seed=42,slice='synthetic',required_types=['miss','drop','shift'],config=config)
        a=rollout(row,'scripted',record=True); b=rollout(row,'scripted',record=True)
        self.assertEqual(a,b)
        self.assertTrue(a['success']); self.assertEqual(a['counts'],{'miss':2,'drop':2,'shift':2})

    def test_config_not_in_observation_and_shift_stays_hidden(self):
        a=FailureEnv(); b=FailureEnv()
        self.assertEqual(a.reset(2),b.reset(2,{'shift':{'count':2}}))
        before=b.seen.copy(); b.step(4 if b.hand[0]<4 else 3)
        self.assertEqual(b.seen,before); self.assertNotEqual(b.obj,before)
        b.step(0); self.assertEqual(b.seen,b.obj)

    def test_opportunity_delay_cooldown_and_probability(self):
        env=FailureEnv(); env.reset(3,{'miss':{'count':2,'after':2,'cooldown':3}})
        self.assertEqual([env.trigger('miss') for _ in range(8)], [False,True,False,False,True,False,False,False])
        env.reset(3,{'miss':{'count':8,'probability':0}})
        self.assertFalse(any(env.trigger('miss') for _ in range(100)))

    def test_horizon_and_false_success(self):
        env=FailureEnv(); env.reset(2,{'horizon':3})
        env.hand=env.target.copy()
        for _ in range(3): env.step(6)
        self.assertTrue(env.done); self.assertFalse(env.success)
        with self.assertRaises(RuntimeError): env.step(0)

    def test_invalid_configs(self):
        for kwargs in ({'count':-1},{'count':1.5},{'probability':1.1},{'after':0},{'radius':0}):
            with self.assertRaises(ValueError): Fault(**kwargs)
        with self.assertRaises(ValueError): FailureEnv().reset(0,{'typo':{}})


class BenchmarkTests(unittest.TestCase):
    def test_split_disjointness_and_composition_holdout(self):
        manifests={s:build_manifest(s) for s in ('train','dev','test')}
        seeds={s:{r['seed'] for r in m['scenarios']} for s,m in manifests.items()}
        for a,b in [('train','dev'),('train','test'),('dev','test')]: self.assertFalse(seeds[a]&seeds[b])
        self.assertTrue(all(len(r['required_types'])<=1 for r in manifests['train']['scenarios']))
        self.assertTrue(all(len(r['required_types'])<=2 for r in manifests['dev']['scenarios']))
        self.assertTrue(any(len(r['required_types'])==3 for r in manifests['test']['scenarios']))
        for manifest in manifests.values():
            for row in manifest['scenarios']:
                self.assertLessEqual(scripted_recovery_bound(row['config']),row['config']['horizon'])
        # Only inspect final-test metadata; do not run policies on its scenarios.

    def test_freeze_integrity_and_training_only_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'suite'
            with contextlib.redirect_stdout(io.StringIO()): freeze(root)
            with self.assertRaises(FileExistsError): freeze(root)
            sampler=ManifestSampler(root); train,_=load_manifest(root,'train')
            allowed={r['seed'] for r in train['scenarios']}
            env=FailureEnv()
            for seed in range(100):
                sampler.reset(env,seed); self.assertIn(env.seed,allowed)
                self.assertLessEqual(sum(f.count>0 for f in env.faults.values()),1)
            with (root/'dev.json').open('a') as f: f.write(' ')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): load_manifest(root,'dev')

    def test_untriggered_fault_is_not_counted_as_recovery(self):
        row=dict(id='x',seed=4,slice='x',required_types=['drop'],config={'drop':{'count':1,'probability':0}})
        result=rollout(row,'scripted'); stats=summarize([result])
        self.assertTrue(result['success']); self.assertFalse(result['all_requested_triggered'])
        self.assertEqual(stats['fault_exposed_episodes'],0); self.assertIsNone(stats['recovery_rate'])

    def test_final_test_requires_explicit_opt_in(self):
        from argparse import Namespace
        from benchmark import evaluate
        with self.assertRaisesRegex(ValueError,'Final test is reserved'):
            evaluate(Namespace(split='test',final_test=False))


if __name__=='__main__': unittest.main()
