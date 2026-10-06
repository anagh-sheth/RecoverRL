# Development results

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
