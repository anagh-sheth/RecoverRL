import unittest
from recoverrl import RecoveryEnv, heuristic, episode, KINDS


class EnvironmentTests(unittest.TestCase):
    def test_replay_determinism(self):
        for kind in KINDS:
            self.assertEqual(episode(42,kind,'heuristic',True),episode(42,kind,'heuristic',True))

    def test_recovery_on_seed_sweep(self):
        for kind in KINDS:
            for seed in range(100):
                result=episode(seed,kind,'heuristic')
                self.assertTrue(result['success'],(kind,seed))

    def test_no_success_from_empty_release(self):
        env=RecoveryEnv(); env.reset(1)
        env.hand=env.target.copy()
        for _ in range(50): env.step(6)
        self.assertFalse(env.success)
        with self.assertRaises(RuntimeError): env.step(0)

    def test_inspection_refreshes_hidden_shift(self):
        env=RecoveryEnv(); env.reset(2,'shift')
        before=env.seen.copy()
        env.step(4 if env.hand[0]<4 else 3)
        self.assertTrue(env.injected)
        self.assertEqual(env.seen,before)
        self.assertNotEqual(env.obj,before)
        env.step(0)
        self.assertEqual(env.seen,env.obj)

    def test_fault_injection_and_hidden_labels(self):
        for kind in KINDS[1:]:
            result=episode(42,kind,'heuristic',True)
            self.assertTrue(result['injected'])
        env=RecoveryEnv()
        self.assertEqual(env.reset(42,'clean'),env.reset(42,'drop'))

    def test_invalid_action(self):
        env=RecoveryEnv(); env.reset()
        with self.assertRaises(ValueError): env.step(7)


if __name__=='__main__': unittest.main()
