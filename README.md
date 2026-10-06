# RecoverRL Lite

A local, robotics-inspired reinforcement-learning laboratory: a gripper must recover from missed grasps, dropped objects, and stale object positions to place a block in a target.

**Scope:** this is a 5×5 discrete grid environment, not rigid-body physics, a vision model, an LLM, or real-robot control. The learned policy is a small PyTorch MLP. It is first trained on scripted demonstrations, then updated with PPO. No paid API, account, robot hardware, or GPU is required.

## Run

Python 3.10–3.12 is recommended. Create a virtual environment and install the requirements on a fresh machine:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python recoverrl.py train --steps 100000
python recoverrl.py evaluate --episodes 100
python -m http.server 8765 --bind 127.0.0.1
```

Open http://localhost:8765 in a browser. You can also open `index.html` directly after evaluation; all data is local. The environment and scripted-policy tests use only the standard library. Training/evaluation of model checkpoints requires PyTorch and NumPy.

On the original development computer, `.venv` reuses the existing Python 3.10 PyTorch installation and isolates NumPy 1.26.4 to resolve an existing global ABI mismatch. Nothing was installed into global Python.

## What is implemented

- Seeded environment with seven actions: inspect, four movement directions, grasp, release.
- Missed grasp: the first otherwise-valid grasp fails once.
- Drop: the first movement while holding releases the object at a neighboring cell.
- Displacement: the first movement before grasp shifts the block to another cell; the observation remains stale until inspection.
- Filtered 10-dimensional observation: gripper position, cached object offset, target offset, holding state, cache age, failed-grasp feedback, elapsed time. Fault labels and true displaced positions are not policy inputs.
- Independent placement verification: releasing a held object at the target ends successfully. Empty release does not.
- Fixed-sequence baseline and reactive inspect/retry heuristic.
- Supervised warm start, clipped PPO objective, value regression, entropy bonus, generalized advantage estimation, gradient clipping, model checkpoints, experiment logs.
- Matched-seed evaluation with completion rates, successful episode lengths, fault-trigger counts, conditional recovery rates, and Wilson confidence intervals.
- Browser replay with scrubbing, selectable policies, and measured benchmark table.

## Reward and training

Terminal success is +10, horizon failure is -2, each action costs 0.03, and invalid actions cost an extra 0.08. Potential shaping uses the true simulator state to estimate remaining transport distance. The shaping term is `gamma * next_potential - current_potential`, with terminal potential zero. This privileged training reward is intentional; the policy still receives filtered observations. A future sparse-reward ablation should measure reliance on shaping.

PPO uses gamma 0.99, GAE lambda 0.95, clipping 0.2, entropy coefficient 0.01, 16 synchronous environments, and rollout length 128. Steps are rounded up to a full rollout. The 50-step horizon is part of the finite-horizon task and is included in observations; terminal failures are not bootstrapped.

Demonstration seeds are 0–1499. Default PPO seed 7 uses environment seeds starting at 710001. Evaluation defaults to **validation** seeds 1000000 onward. Reserve 2000000 onward for final testing after development:

```sh
python recoverrl.py evaluate --episodes 500 --start-seed 2000000
```

Use separate output directories for different training seeds (`--out runs/seed8`). Keep training seed choices and seed partitions disjoint; the current formula is intended for small training seed values such as 0–9. Do not tune on final-test results. `evaluate` replaces the viewer's `demo-data.js` with the selected evaluation run.

## Reading results honestly

The scripted recovery baseline should perform very well: it explicitly solves this small environment. Compare supervised + PPO against **both** the supervised checkpoint and the heuristic. A higher PPO training success rate does not establish held-out improvement. The initial run is one training seed and is only a prototype result.

The replay exposes true object positions to the human viewer for explanation, including hidden displacement events. These are not part of the learned agent's observation. Five fixed demo seeds per condition are recorded without searching for successful examples.

All four conditions are encountered in training; new seeds are not evidence of novel-failure or real-world generalization. No collisions, contact physics, visual perception, or genuinely unseen failure combinations are modeled yet. Fault-conditioned recovery can have selection effects if a weak policy never reaches the trigger; use overall completion and trigger counts as well.

## Next milestones

1. Repeat training with three seeds; compare supervised/PPO and remove reward shaping in an ablation.
2. Add held-out disturbance combinations, train/test spatial splits, and stochastic severity.
3. Introduce recurrent policies for longer partial-observation tasks.
4. Port the action and observation contract to ManiSkill with fixed motion primitives.
5. Only then add language-model planning or GRPO as a separate experiment.

## Files

- `recoverrl.py`: environment, policies, PPO training and evaluation CLI.
- `test_recoverrl.py`: deterministic replay, recoverability, observation isolation and success-verifier tests.
- `index.html`: local replay viewer.
- `runs/default/`: generated checkpoints, training log and evaluation JSON.
- `demo-data.js`: generated browser replay data.

This prototype is not yet evidence for any resume performance claim. Use measured held-out comparisons after repeated experiments.
