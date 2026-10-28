# Select one, two or three noise levels

For the next study, use an explicit noise section. The default is **one absolute
Gaussian standard deviation, sigma=0.01**. Keep this choice for the current
study; changing it requires a fresh output directory.

```bash
# NOISE SETTINGS — change these two lines together.
NOISE_LEVELS=1
NOISE_VALUES=(0.01)

python scripts/prepare_cart_study.py \
  --noise-levels "$NOISE_LEVELS" --noise-values "${NOISE_VALUES[@]}"
```

This preview prints the selected levels and configuration count without GPU
discovery, data loading or starting training. Defaults are CART only, MNIST and
CIFAR-10, IID and Dirichlet, seeds 2025/2026/2027, 100 nodes and 100 rounds.

| Levels | Example settings | CART-only configurations |
| --- | --- | ---: |
| 1 | `--noise-levels 1 --noise-values 0.01` | 12 |
| 2 | `--noise-levels 2 --noise-values 0.005 0.02` | 24 |
| 3 | `--noise-levels 3 --noise-values 0.005 0.01 0.02` | 36 |

Each count is 2 datasets x 2 partitions x 3 seeds x the selected number of
levels. To test only CIFAR-10, add `--datasets cifar10`, giving 6/12/18 cases.
Every sigma must be finite, positive and distinct. The number of values must
equal `--noise-levels`; values are stored in increasing order. Without explicit
values, the defaults are .01 for one, .005/.02 for two, and .005/.01/.02 for three.
Sigma is an absolute coordinate standard deviation, not a percentage.

The original scientific configuration functions are reused unchanged. The new
selection layer chooses how many to create. With all three original values,
it generates exactly the original configuration identities. Clean and other
noise-free conditions have sigma zero and are created once per seed/split,
not once per selected noise value. For the six-condition focused design the
counts are therefore 72, 132 and 192, not 72 multiplied by the level count.

## Preparing a future frozen plan

The preparation tool also accepts the existing dataset, seed, condition,
node/round, model, GPU-slot and resource options. `--preflight --output FRESH`
checks TRAIN inputs and saves their identities. `--verify-gpus --output FRESH`
freezes the selected configurations and runtime profiles, then runs the original
GPU checks and compiled-transfer exactness gate. It never starts research training.
The resulting plan uses the existing `transfer --resume-plan` execution backend.

The plan stores `noiseSelection.count` and `noiseSelection.values`. Its generator
and selection-helper hashes are recorded separately from the frozen mathematical
source. Existing plans cannot change noise levels, conditions, source or resources
in place. Recovery reads the saved selection, rather than changing it from new
command-line defaults. The active HPC plan and runtime fingerprint are unchanged.

The preview reports `controlReferenceNeededForComparison: true` for CART-only
plans. Reusing completed controls and replacing a CART series in combined figures
is a separate reporting step: controls must match data, initialization and every
non-CART setting, and the old CART results stay archived. The planner does not
silently merge old result folders or overwrite published figures. A final exporter
that requires paired controls must receive those references before producing the
combined comparison. Do not rerun unchanged controls merely to change the noise
selector.

Averages always combine the three seed results at the **same noise level**.
They never pool different sigma values. With only 12 configurations, at most
12 workers can be active even if the scheduler permits 24 slots.

Validation: 33 targeted tests passed, including configuration counts, exact
agreement with the frozen three-level matrix, invalid-input rejection and
preservation of existing plans. A two-level tiny GPU plan completed both cases
and passed independent CPU verification; its [receipt](validation/CART_MODULAR_NOISE_20261028.json)
documents execution scope. This does not establish CART accuracy improvements.
