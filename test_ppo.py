import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from ppo import gae_returns, independent_value_model
from recoverrl import make_model, load_model, save_model


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_value_update_preserves_actor_exactly(self):
        torch.manual_seed(12)
        model=independent_value_model(make_model())
        obs=torch.randn(32,10)
        before=model(obs)[0].detach().clone()
        params=list(model.value_body.parameters())+list(model.critic.parameters())
        opt=torch.optim.Adam(params,lr=.01)
        loss=(model(obs)[1]-5).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        self.assertTrue(torch.equal(before,model(obs)[0]))
        self.assertTrue(all(p.grad is None for p in model.body.parameters()))

    def test_conversion_and_checkpoint_preserve_logits(self):
        old=make_model(); new=independent_value_model(old); obs=torch.randn(8,10)
        self.assertTrue(torch.equal(old(obs)[0],new(obs)[0]))
        with tempfile.TemporaryDirectory() as d:
            for model in (old,new):
                path=Path(d)/'model.pt'; save_model(model,path)
                self.assertTrue(torch.equal(model(obs)[0],load_model(path)(obs)[0]))

    def test_terminal_stops_bootstrap_and_future_episode_leakage(self):
        rewards=np.array([[1.],[100.]])
        dones=np.array([[True],[False]])
        values=np.array([[.5],[2.]])
        adv,ret=gae_returns(rewards,dones,values,np.array([3.]),gamma=.9,lam=1.)
        self.assertAlmostEqual(float(ret[0,0]),1.)
        self.assertAlmostEqual(float(ret[1,0]),102.7,places=4)

    def test_finite_episode_discounted_returns(self):
        _,ret=gae_returns(np.array([[1.],[2.],[3.]]),np.array([[False],[False],[True]]),
                          np.zeros((3,1)),np.array([999.]),gamma=.5,lam=1.)
        np.testing.assert_allclose(ret[:,0],[2.75,3.5,3.])


if __name__=='__main__': unittest.main()
