# Development results

## Grasp choice and deadlines — October 6, 2026

The new two-action experiment is no longer saturated. Fast and secure grasps trade time for reliability under persistent hidden risk. See `GRASP.md` for the task and frozen protocol, and `benchmarks/grasp-v1/development-summary.json` for machine-readable scores, slices, hashes and training configurations.

| Policy | Greedy completion | Sampled completion | Greedy mean reward |
|---|---:|---:|---:|
| Always fast | 62.125% | — | 0.5254 |
| Always secure | 46.875% | — | 0.3647 |
| Switch after failure (teacher) | 65.625% | — | 0.5594 |
| Secure when delivery fits | **72.375%** | — | **0.6209** |
| Supervised seed 6 | 62.125% | 63.250% | 0.5254 |
| Supervised + PPO seed 6 | 65.625% | 65.375% | 0.5575 |
| Supervised seed 7 | 62.125% | 62.875% | 0.5254 |
| Supervised + PPO seed 7 | 65.000% | 65.000% | 0.5522 |
| Supervised seed 8 | 62.125% | 63.000% | 0.5254 |
| Supervised + PPO seed 8 | 62.125% | 63.375% | 0.5254 |

Every cell uses the same 800 development scenarios. Three independently initialized supervised models each received 30 epochs on 2,157 teacher transitions, then 100,352 PPO transitions. Critic warmup added 611, 604 and 593 transitions respectively. All final checkpoints were evaluated; the final-test split remains unused by policies.

PPO improves over its supervised starting point in two of three greedy runs, but does not beat the stronger deadline-aware rule in either completion or reward. The supervised checkpoints choose only fast grasps under greedy development evaluation, falling short of their switch-after-failure teacher. Thus these results are confounded by imperfect imitation; they do not establish that RL is better than well-trained supervision. Extra PPO interactions are also additional data. The next controlled experiment should first fit the teacher reliably, add supervision from the stronger deadline-aware teacher, then compare post-training across seeds without altering this frozen benchmark.

The small state/action space still permits strong hand-written policies. Harder stochastic outcomes remove the 100% ceiling but do not automatically make RL the best approach. Fixed navigation and the assumed 10× reliability difference also limit robotics claims.

Validation: 30 unit tests pass, including all previous environments, persistent risk, deadline enforcement, hidden observations, action timing, train-only sampling, and dynamic checkpoint/rollout dimensions. The viewer at `/grasp.html` uses fixed first-seed replays per scenario cell. Full reports/checkpoints remain local under `runs/grasp-v1/`; the compact summary is included with the source.

## Frozen failure-composition development benchmark

Implemented in `failure_env.py` and `benchmark.py`; see `BENCHMARK.md` for the protocol and exact commands. The suite contains 800 training, 700 development and 800 reserved final-test configurations, frozen before evaluating the learned policies.

On the 700 development scenarios, the fixed sequence completed 201 (28.71%). Scripted recovery, the frozen supervised model and corrected v1 PPO all completed 700 (100%); learned models also scored 100% with reproducibly sampled actions. No model was adapted to the new suite for this evaluation. The 100-action v2 budget differs from the 50-action v1 budget; do not compare the aggregate rates across versions directly.

The training-only manifest adapter passed a 2048-step PPO smoke run. This is pipeline validation, not evidence of adaptation gains. Final-test policies have not been evaluated. Development saturation is reported rather than changing the frozen scenarios to manufacture an RL advantage.

## PPO stabilization — October 6, 2026

The corrected trainer preserves the supervised policy's completion rate in the measured validation set. It does **not** outperform the already-perfect supervised baseline on this task.

| Checkpoint | Greedy completion | Sampled completion |
|---|---:|---:|
| Supervised, 60 epochs | 100% | 100% |
| Original PPO | 69.75% | 98.5% |
| Independent value features, original actor LR/entropy (seed 7) | 100% | 100% |
| Conservative PPO (seed 6) | 100% | 100% |
| Conservative PPO (seed 7) | 100% | 100% |
| Conservative PPO (seed 8) | 100% | 100% |

Each cell covers 400 validation episodes: 100 seeds per condition, seeds 1000000–1000099. The three corrected PPO runs share the same supervised starting checkpoint, run 100352 PPO transitions apiece, and use separate training environment/action seeds. Critic warmup adds 2973, 2990, and 2951 transitions for seeds 6, 7, and 8 respectively. All final checkpoints are evaluated; none is selected by validation score. These are repeated measurements on the same development scenarios, not 1200 unique scenes or an untouched final test. A 100/100 condition-level result has a 95% Wilson interval of approximately 96.3–100%.

### What the diagnosis established

On training-state observations from 100 scripted episodes, **32 value-only optimizer steps** changed **10.53% of greedy actor decisions** in the shared-network model. Mean actor KL was 0.1791. With independent value features, the same value-only optimization changed **zero** actor decisions and produced zero actor KL, while obtaining the same value MSE. This directly demonstrates gradient interference in the original architecture, though it does not prove it was the only cause of the full-run regression.

The original PPO checkpoint also behaved differently under evaluation modes: 69.75% greedy versus 98.5% sampled. Thus comparing sampled training success with greedy evaluation had obscured the problem. Both modes are now measured separately.

### What changed

- Independent value features prevent critic fitting from moving actor logits.
- Actor and critic have separate optimizers and gradient clipping.
- Default actor LR is reduced from 3e-4 to 3e-5; entropy coefficient from 0.01 to 0.001.
- Frozen-policy critic warmup precedes PPO.
- Exact categorical KL is logged, with per-rollout early stopping of actor updates above 0.01. This safeguard did not activate in these corrected runs; it should not be credited for the observed recovery.
- Logs include entropy, clipping fraction, value fit, interaction counts and run configuration.
- Legacy checkpoints load, legacy training remains available, and existing run checkpoints are protected from overwriting.

The separated-network ablation also succeeds without the lower learning rate, critic warmup or KL safeguard. It additionally changes optimizer separation, optimizer initialization and clipping relative to the legacy trainer, so it is not a perfect single-variable full-run experiment. The isolated value-update diagnosis above is the cleaner causal check.

### Verification and remaining limits

All 10 unit tests pass, including actor/value gradient isolation, legacy/new checkpoint round trips, terminal bootstrap handling and finite-episode discounted returns. Training generated new actor weights and final checkpoints; this is not a frozen-policy workaround. Old experiments remain intact.

The benchmark is saturated: supervised and scripted recovery already achieve 100%. The next useful experiment needs harder conditions and unseen failure combinations before claiming an RL advantage. No final-test seeds (2000000+) were used in this work.

The viewer now compares the original PPO with `runs/stable7/ppo.pt`. Rebuild it with:

```sh
.venv/bin/python recoverrl.py evaluate --episodes 100 --out runs/stable7 --previous-ppo runs/warmstart60/ppo.pt
```

Reproduce the diagnosis and post-training in fresh run directories:

```sh
.venv/bin/python diagnose_value.py --checkpoint runs/warmstart60/supervised.pt --output runs/diagnosis/value_interference.json
.venv/bin/python ppo.py --checkpoint runs/warmstart60/supervised.pt --out runs/reproduce-stable7 --seed 7
.venv/bin/python compare_policies.py runs/warmstart60/supervised.pt runs/warmstart60/ppo.pt runs/reproduce-stable7/ppo.pt --output runs/reproduce-stable7/comparison.json
```

## Original experiments — retained for comparison

These are validation results from 100 previously unseen seeds per condition (1000000–1000099). All failure types were present in training. One model seed (7) was used; these are not final-test or real-robot results.

| Policy | Clean | Miss | Drop | Shift | Overall |
|---|---:|---:|---:|---:|---:|
| Fixed sequence | 100% | 0% | 0% | 0% | 25% |
| Scripted recovery | 100% | 100% | 100% | 100% | 100% |
| Supervised, 12 epochs | 35% | 49% | 53% | 28% | 41.25% |
| Supervised 12 + PPO | 44% | 44% | 40% | 32% | 40% |
| Supervised, 60 epochs | 100% | 100% | 100% | 100% | 100% |
| Supervised 60 + PPO | 76% | 65% | 65% | 73% | 69.75% |

Both PPO experiments ran 100352 environment steps. The PPO weights changed, checkpoints reload, and evaluation uses greedy actions. Stochastic training success is not directly comparable to greedy evaluation success. Longer supervision fixed the initial underfitting, but PPO degraded the strong supervised policy. We do not claim an RL improvement.

Original checkpoints and logs are preserved in `runs/default`; the second experiment is in `runs/warmstart60`. New training defaults to 60 supervised epochs, based on this development finding.

Original commands (choose a fresh output directory to rerun training):

```sh
.venv/bin/python recoverrl.py train --steps 100000 --bc-epochs 60 --legacy-ppo --out runs/legacy-reproduction
.venv/bin/python recoverrl.py evaluate --episodes 100 --out runs/warmstart60
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

Validation: six environment tests passed, including seeded replay, 400 scripted recoverability checks, false-success rejection, observation isolation and invalid actions. Browser playback and scrubbing were verified.

Next: diagnose PPO policy drift (shared actor/value features and initially untrained critic are hypotheses), compare stochastic and greedy evaluation explicitly, then add unseen failure combinations and repeat across training seeds. Reserve seeds 2000000 onward for final testing. The current task is intentionally small and solvable by a heuristic.
