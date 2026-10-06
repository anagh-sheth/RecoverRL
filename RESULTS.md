# Initial development results

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

The viewer currently displays the 60-epoch experiment. Original checkpoints and logs are preserved in `runs/default`; the second experiment is in `runs/warmstart60`. New training defaults to 60 supervised epochs, based on this development finding.

To reproduce the current viewer (from the project directory):

```sh
.venv/bin/python recoverrl.py train --steps 100000 --bc-epochs 60 --out runs/warmstart60
.venv/bin/python recoverrl.py evaluate --episodes 100 --out runs/warmstart60
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

Validation: six environment tests passed, including seeded replay, 400 scripted recoverability checks, false-success rejection, observation isolation and invalid actions. Browser playback and scrubbing were verified.

Next: diagnose PPO policy drift (shared actor/value features and initially untrained critic are hypotheses), compare stochastic and greedy evaluation explicitly, then add unseen failure combinations and repeat across training seeds. Reserve seeds 2000000 onward for final testing. The current task is intentionally small and solvable by a heuristic.
