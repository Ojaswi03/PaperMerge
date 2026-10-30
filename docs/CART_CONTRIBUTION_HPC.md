# Matched CART contribution study

The completed 24-case comparison is preserved. This follow-up uses a new output
and a separate producer adapter. It does not overwrite the earlier measurements
or change their interpretation.

## What runs

The default `contribution` preset contains 66 independent configurations:

- CIFAR-10: nine variants × two partitions × three seeds = 54.
- MNIST: full CART and checked-pooling ACDS × two partitions × three seeds = 12.

Every full configuration has 100 clients, 100 complete ring rounds and five
mini-batch SGD steps per activation. Seeds are 2025, 2026 and 2027. The Hidden
attack has 33 identities and starts at displayed round 21. Parameter-coordinate
noise has standard deviation 0.01; Dirichlet concentration is 0.5. The primary
endpoint is mean client accuracy over rounds 91–100. Final round, honest-client
mean, worst-client accuracy and class recall are supplementary measurements.

| Variant | Controlled change from full CART |
| --- | --- |
| Full CART | Exactly reproduces the completed local-prior/fallback CART producer |
| Checked ACDS | Ordinary SHARP training with 5% fitting-example sharing; score the realized pool against its best selected member before use |
| Checked SHARP | Same pool check without shared examples; separates sharing from the pool rule |
| Macro only | Remove the class-deficit penalty; its now-unused calibration is also skipped |
| Admission only | Retain ingress, local-prior training and classifier repair; omit the final full-update gate |
| Update check only | Use the SHARP pool as ingress and anchor; build the class-quality context for checking its complete update |
| No training prior | Keep full CART decisions but use ordinary CE in the SGD objective |
| No calibration | Set allowance draws to zero while retaining the class penalty |
| No head repair | Omit ridge repair/reference admission; retain ingress and full-update checking, keeping previous reference metadata |

All CART variants retain consistent peer fallback eligibility. The macro,
admission-only and other interventions therefore do not silently revert to the
older candidate eligibility or optimizer. Each intervention is part of the saved
case and checkpoint contract. Optional `--variants` includes `local_only` and
`acds_single` for additional controls; those are not in the default 66-case plan.
The default matrix isolates individual components, not every factorial interaction.
Published BASIL, Bristle, BALANCE and distillation methods have not thereby been
reproduced as matched baselines.

## Four-GPU launch

After pulling the commit containing this guide:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

python scripts/run_cart_contribution.py \
  --preset contribution --gpu-count 4 \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029" \
  --execute --wait-for-resources --tmux cart-contribution
```

All cases are independent worker processes. The placement balances total counts
and distributes each method across devices. On four free 97,887 MiB GPUs, the
66 cases occupy at most 17 workers per GPU. Each worker has a 4,096 MiB TensorFlow
allocator cap plus a reserved 1,024 MiB context allowance; each GPU retains
8,192 MiB headroom. The largest full reservation is
`17 × (4096 + 1024) + 8192 = 95,232 MiB`. Host reservation is 16 GiB per worker
plus 32 GiB headroom, or 1,088 GiB for all 66. These budgets fit the server
inventory supplied by the author, subject to live availability.

Before training, each selected physical GPU must pass objective/transfer
verification and warm informative CART/pooling checks under the actual saved
allocator cap. Source snapshots are verified before workers start. The supervisor
then admits workers round-robin with full startup GPU and host-RAM reservations.
It fills all four GPUs as memory permits; occupied memory or unavailable telemetry
causes waiting. Shared-machine conditions cannot support an unconditional promise
that every case starts immediately or that OOM is impossible.

`--gpu-count 1`, `2` or `3` supports smaller allocations; `--gpus 0,2` selects
specific physical devices. Smaller allocations queue cases when needed. Numerical
profiles, GPU UUIDs, host, source and output path remain fixed for checkpoint
recovery. Do not change the plan to relocate a partially trained model.

To print the case count without loading data, accessing GPUs or training:

```bash
python scripts/run_cart_contribution.py --describe \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029"
```

The smaller `comparison` preset runs 24 full-length cases (CART and checked ACDS
on both datasets). `screen` runs eight ten-round cases with seed 2025 and attack
onset at displayed round 4; it is a development screen, not a full-result substitute.

## Monitoring and recovery

```bash
tail -f .papermerge/local-tests/cart-contribution-100r-20261029/adaptive_supervisor.log

python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029"

python scripts/share_cart_monitor.py \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029" \
  --tmux cart-contribution-browser
```

The browser publisher prints the run's GitHub Pages URL. Terminal and browser
share the same short method/environment labels. Closing a monitor leaves training
running. Each successful worker logs `GPU n: started ...`; `pending 0` in the
monitor confirms that all planned cases have been admitted. Inspect failures and
each worker's log if a case fails; other cases continue.

An interrupted supervisor is resumed, after its workers have checkpointed, with:

```bash
python scripts/run_cart_contribution.py --resume --execute \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029" \
  --wait-for-resources --tmux cart-contribution
```

Use the original checkout/source and environment. A duplicate live supervisor
is refused; completed cases are skipped, and incomplete cases recover their saved
activation cursor. Recovery never changes the saved method or numerical profile.

## Automatic figures and completed-run audit

After all cases succeed, the supervisor generates line figures in
`publication_curves/all_methods/`, `comparison/`, `admission_and_commit/`, and
`training_and_repair/`. Outputs include editable SVG, vector PDF, 600 dpi PNG,
round/seed CSVs and checksummed provenance. Lines show actual recorded rounds,
with sample SD across seeds; they are not smoothed. No bar graphs are generated.
The axes say validation accuracy because the current recorded series uses the
training-derived development holdout. A benchmark-test curve cannot be recovered
from validation arrays or from final-only checkpoint inference.

```bash
python scripts/verify_cart_comparison_results.py \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029"

python scripts/export_cart_contribution.py \
  --output "$PWD/.papermerge/local-tests/cart-contribution-100r-20261029" \
  --destination "$PWD/docs/results/cart-contribution-100r-20261029"
```

Independent final CPU inference is separate from the launch gates. Keep it before
exporting thesis evidence. The exporter omits checkpoints and raw examples. For
the already completed 24-case run, use the existing
`verify_cart_comparison_results.py` and `export_cart_comparison.py` with its original
output directory; do not use the new contribution exporter on that legacy plan.

CART's contribution and its conditional empirical contract are discussed in
[CART_NOVELTY_POSITION_20261029.md](CART_NOVELTY_POSITION_20261029.md) and
[CART_EMPIRICAL_CONTRACT.md](CART_EMPIRICAL_CONTRACT.md). This is a runnable
attribution study, not a certificate of originality or efficacy.

## Local release checks

The [saved readiness receipt](https://github.com/Ojaswi03/PaperMerge/blob/a731e45b08c4fc66c613fa33342d900a65bb7eda/docs/validation/CART_CONTRIBUTION_READY_20261029.json)
records 125 passing Python tests and eight browser tests. A local RTX 4070 Ti
passed 21 objective/transfer/intervention checks under the 4 GiB allocator cap.
Four separate 100-client, two-round GPU integration cases completed, produced
line figures and passed independent CPU inference for all 400 final client
models with zero accuracy discrepancy. These short cases validate execution and
provenance, not full-length efficacy. The actual HPC environment and each selected
GPU still have to pass their enforced launch gates.
