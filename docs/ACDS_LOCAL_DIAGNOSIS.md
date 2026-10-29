# Local ACDS learning diagnosis and checked averaging

The data-sharing implementation delivers the requested examples. The local
evidence instead identifies a gap in SHARP's candidate-to-average transition:
ranking complete models by local loss does not check the quality of their
parameter average. The opt-in checked-pooling selector tests that realized
average against its best selected coherent member on the same private ranking
batch before using it as the SGD initialization.

The running HPC comparison retains its frozen producer and original baseline.
This local variant is separate from the ACDS sharing algorithm and does not add
CART admission, class probes, classifier repair or a complete-update gate.

## Matched local measurements

All cases use CIFAR-10, 100 clients, Dirichlet alpha 0.5, seed 2025, 5% ACDS
sharing, the same initialization and development holdout, Gaussian coordinate
noise sigma 0.01 and five SGD steps per activation. They run in the same local
TensorFlow 2.18 environment on the RTX 4070 Ti. The attack is the existing
Hidden transform with 33 configured identities.

| Diagnostic | Original SHARP + ACDS | Checked averaging | Always best single |
| --- | ---: | ---: | ---: |
| Three rounds, before attack onset | 10.000% | 10.890% | 10.890% |
| Ten rounds, attack begins at displayed round 4 | 10.000% | 22.644% | 21.921% |

These are final-round mean client-model validation accuracies, with one seed.
Each three-round case uses 1,500 optimizer steps; each ten-round case uses
5,000. The ten-round onset is deliberately earlier than the full HPC study's
round 21 and its outcomes are not pooled with the 100-round results.

Fresh CPU inference verified all 600 final client models across the six
completed cases. Every final client accuracy matched its saved value exactly;
the maximum observed accuracy difference was zero. This verifies measurements,
not a proof of generalization or a replay of every training step.

## What the checks establish

The shared pool contains 2,402 distinct fitting examples in this partition.
Every receiver has all ten classes after sharing; its expanded data contain
2,523 to 3,653 examples. Mean dominant-class fraction falls from 36.76% in the
owned partitions to 14.33% after sharing. The fitting/share indices are disjoint
from the 1,000-example evaluation holdout.

The unchanged three-round baseline finishes with all clients predicting class
0 throughout that holdout. There are no active attack transmissions in these
three rounds. In all 265 activations that average multiple copies, the average
has higher measured ranking-batch CE than its best selected member. This
establishes an empirical regression at the pooling step even before attack
onset; it does not establish that pooling is the sole source of degradation.

The new selector retains the average when its measured CE is no greater than
the best member's CE, within a numerical tolerance of 1e-12. Otherwise it
reconstructs the best member using the same keyed link perturbation and uses
that complete vector. It records the original and actual selected counts and
updates the actual count to one on fallback. Consequently, the existing RLA
formula uses the actual number of averaged noisy copies; no variance benefit
from a rejected average is retained.

At round 10, checked averaging uses the original 33-copy average for 33 clients
and a single member for 67, giving mean selected count 11.56. Always selecting
one member recovers most of the observed improvement. Checked averaging is
only 0.723 percentage points above that simpler control in this seed, so the
data do not establish a reliable incremental benefit or a new robustness
guarantee. The selection intervention also changes RLA strength through the
existing actual-count formula; this experiment does not independently isolate
those two effects.

## Code and evidence

- `basil_core/sharp_checked_pool.py` implements the opt-in selector.
- `scripts/diagnose_acds_pool.py` runs matched, source-frozen local cases.
- `scripts/verify_acds_pool_diagnostic.py` performs fresh CPU inference.
- `scripts/report_acds_pool_diagnostic.py` exports verified metrics and curves.
- `tests/test_sharp_checked_pool.py` checks fallback, useful-average retention,
  JSON-safe evidence, exact baseline reproduction and checkpoint recovery.
- `docs/results/acds-pool-local-20261029/` contains the six metric archives,
  per-round CSV, verification records, frozen diagnostic sources and a line
  figure in SVG, PDF and 600-dpi PNG. Model checkpoints remain local.

The source fingerprints of the full HPC producer still match
`docs/validation/CART_FULL_HPC_READY_20261029.json`. Completed diagnostic exports
contain their original contracts and source identities. Initial instrumentation
errors remain in local logs and are excluded from the six completed cases.

For another bounded local comparison, use a fresh output directory:

```bash
environment/basil-noise-env/bin/python scripts/diagnose_acds_pool.py \
  --execute --output .papermerge/local-tests/acds-new-check \
  --partition dirichlet --rounds 10 --attack-start 3 \
  --modes raw checked single

environment/basil-noise-env/bin/python scripts/verify_acds_pool_diagnostic.py \
  --output .papermerge/local-tests/acds-new-check
```

A full-length, multi-seed evaluation of this separately named selection variant
is still needed before changing the main comparison or thesis conclusions.
