# Failure composition benchmark

RecoverRL v2 adds configurable and repeated failures without changing the original `RecoveryEnv`, checkpoints, or v1 results. It keeps the same seven actions and ten-dimensional observation interface; the last feature now represents the fraction of the **100-action** v2 budget used. Compare models within v2; v1 used 50 actions, so aggregate scores across versions are not directly comparable.

## Failure configuration

```python
from failure_env import FailureEnv

env = FailureEnv()
observation = env.reset(seed=42, config={
    'horizon': 100,
    'miss': {'count': 2, 'probability': 0.8, 'after': 1, 'cooldown': 2},
    'drop': {'count': 2, 'probability': 0.55, 'radius': 2},
    'shift': {'count': 1, 'probability': 1.0, 'radius': 3},
})
```

`count` is a maximum number of injected events, not a promise that they all occur. `probability` is the per-eligible-opportunity trigger probability until that cap is reached. For `miss`, it is the probability of failure on an otherwise-valid grasp; after the cap is exhausted, grasps are reliable. This models bounded disturbance episodes, not permanently unreliable hardware.

`after` is the first eligible opportunity index; `cooldown` is the minimum gap in opportunity indices between triggers. Neither is measured in global simulation timesteps. Opportunities are valid grasps for misses, successful grid movements while holding for drops, and successful movements while empty for shifts. Inspecting or bumping a wall does not consume a movement opportunity. A drop and a shift cannot both fire on the same movement.

`radius` limits Manhattan displacement for drops and shifts. Object locations stay on the connected grid and are never relocated onto the target. Failed grasps do not move objects. Separate seed/type/opportunity random streams make replay deterministic and prevent unrelated actions from consuming another fault's randomness. Different policies can encounter different opportunities, so matching seeds do not guarantee identical realized trajectories.

Fault configurations, counters, labels and ground-truth displacements are evaluator-only metadata. The actor sees the same filtered state fields as before. Replays show privileged state to help a human understand failures.

## Frozen split

The files under `benchmarks/composition-v1/` were generated before running learned-policy evaluations. They contain explicit configurations and environment seeds. `LOCK.json` pins each manifest's SHA-256. Loading checks the hash and version; `freeze` refuses to overwrite an existing directory. This is a version-controlled reproducibility convention, not a cryptographic access-control boundary. Create a new version instead of modifying these files.

| Split | Scenarios | Conditions | Seed base |
|---|---:|---|---:|
| Train | 800 | Clean and individual miss/drop/shift families; at most one event per family | 3000000 |
| Development | 700 | Same individual conditions plus miss+drop, miss+shift, drop+shift | 4000000 |
| Final test | 800 | Clean, repeated individual failures, repeated pairs, and triple combinations | 5000000 |

Each profile has 200 training seeds or 100 evaluation seeds, offset by 10000 between profiles. These seed ranges are disjoint from the old experiments. This is a seed split, not a guarantee of disjoint physical layouts on a small 5×5 grid. Development compositions are unseen in training but available for model selection. Triple compositions are reserved for final evaluation.

Training/development use probabilities 0.55, 0.8 and 1.0; displacement radii 1–2; event cap 1. Final metadata reserves event caps 2–4 and radii 2–4. Final-test metadata and integrity are inspected, but policies have **not** been run on the final-test episodes.

### Recoverability without selecting favorable model outcomes

The loader verifies a conservative action bound for the scripted inspect/retry policy:

`18 + 2*miss_count + (shift_radius+2)*shift_count + (2*drop_radius+2)*drop_count`

Clean transport costs at most 18 actions on this board. A miss adds a failed attempt and inspection; a stale-location shift adds at most its displacement plus a failed grasp and inspection; a drop adds relocation travel, regrasp and inspection. The largest frozen configuration has a bound of 90 actions, below the common 100-action budget. Thus bounded disturbances remain recoverable by this policy without evaluating final-test models or filtering scenarios based on learned success. Arbitrary custom configurations can exceed this bound and may be impossible within their chosen horizon.

## Run the benchmark

```sh
# Check hashes, configuration validity and split separation.
.venv/bin/python benchmark.py verify

# Use a fresh output path; prior reports are never overwritten.
.venv/bin/python benchmark.py evaluate \
  --checkpoint runs/warmstart60/supervised.pt \
  --checkpoint runs/stable7/ppo.pt \
  --out runs/composition-v1/my-dev-report.json

# Export that report to the separate browser viewer.
.venv/bin/python benchmark.py export --input runs/composition-v1/my-dev-report.json
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

Open `/benchmark.html` on the local server, or open the file directly after export. The original viewer and its data remain available at `/index.html`. The new viewer accepts development reports and shows the first enumerated replay in each condition, never a success-selected example.

Evaluation runs both greedy and reproducibly sampled actions for learned models, and greedy scripted baselines. Each model gets identical scenario definitions and action budgets. Results include overall completion, steps across all episodes, steps on successes, Wilson intervals, opportunity/event counts, realized failure combinations, and conditional recovery. A configured pair can realize only one failure if the policy never reaches the other trigger; always examine exposure counts. Conditional success can be biased by a policy's exposure to faults.

Final evaluation requires both `--split test` and `--final-test`; keep it unused until model and hyperparameter choices are finalized. The developer-facing flag guards accidental use, not deliberate access. Do not tune models on final-test results.

## Train using only the training manifest

```sh
# Adapt an existing model with PPO; only train.json is loaded by the sampler.
.venv/bin/python ppo.py \
  --checkpoint runs/warmstart60/supervised.pt \
  --suite benchmarks/composition-v1 \
  --out runs/composition-v1/my-ppo-run --steps 100000 --seed 7

# Alternatively generate new supervised demonstrations, then train PPO.
.venv/bin/python recoverrl.py train \
  --suite benchmarks/composition-v1 \
  --out runs/composition-v1/my-full-run --steps 100000
```

Both imitation demonstrations and PPO/critic warmup use the locked training manifest when `--suite` is provided. Logs record the training-manifest hash and the number of distinct PPO/warmup scenarios seen. The original training path remains unchanged when the flag is omitted. A 2048-step integration smoke run completed successfully in `runs/composition-v1/smoke`; it is **not** an adaptation-performance experiment.

## Initial development result

The frozen supervised model and the corrected v1 PPO model both completed **700/700** development episodes under both greedy and sampled evaluation. Scripted recovery also completed 700/700. The fixed sequence completed 201/700 (28.71%). The models were not adapted to this new suite before this measurement.

These results show that the implemented combinations are still easy for the existing recovery strategy. They do not show an RL advantage. The manifests were not changed to force a performance gap, and the final-test set was left unused. Before spending on longer training, consider a separately versioned development task with a genuine planning tradeoff (for example, information-gathering cost or alternate grasp strategies), and compare additional supervision with PPO under explicit interaction budgets.

The full checked report is `runs/composition-v1/verified-dev.json`; generated reports, checkpoints and viewer data remain local under the existing ignore rules. The manifests and this specification are source files intended for version control.
