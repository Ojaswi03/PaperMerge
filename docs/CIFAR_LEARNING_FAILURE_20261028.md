# CIFAR-10 learning failure before the attack

The current 100-node HPC study shows a substantive learning problem rather than
an execution failure. Keep that study unchanged and preserve its negative results.
This note separates user-supplied HPC observations from a new local diagnostic.

## What the HPC audit establishes

The user supplied the checkpoint audit of CIFAR-10 IID, seed 2025, absolute
sigma .01. Attack onset is zero-based round 20, displayed as round 21.

| Condition | Last pre-attack accuracy, round 20 | First attack-active accuracy, round 21 |
| --- | ---: | ---: |
| Byzantine mitigation | 11.61% | 11.82% |
| Both mitigations | 11.64% | 11.85% |
| Both mitigations + CART | 12.08% | 12.26% |

The failure to learn therefore predates the active Hidden transformation. Local
training CE stays close to log(10)=2.3026, the uniform-prediction baseline. This
is consistent with limited learning, not proof that every prediction is uniform.

CART is active: the saved CART checkpoint contains 1,592 activations with
positive gates and 452 accepted corrections by activation 5,800. At the same
completed round 58, CART reaches 19.19% versus 14.33% for both mitigations without
CART, a +4.86 percentage-point difference for this seed. The absolute accuracy is
still poor and the other seeds/complete endpoint are needed.

The peak screen retains almost all honest and directly attacked copies here;
local-loss ranking subsequently selects relatively few directly attacked copies.
The 100% ancestry flag means at least one selected lineage has previously been
exposed to an attack. It is not a measure of present corruption magnitude or
proof that every current input is harmful. Zero gate/correction counts in
conditions without CART are expected.

These numbers are transcribed from the user's HPC audit output, not independently
recomputed locally from the unavailable HPC raw files.

## Matched local experiment

`scripts/probe_cart_handoff.py` ran eight pre-attack arms on the local RTX4070Ti:
CIFAR-10 IID, seed 2025, all 49,000 fitting images and the existing 1,000-image
TRAIN holdout, 100 nodes, five rounds and five local SGD steps. CART and RLA are
disabled in every arm so they cannot explain these differences. The actual
Hidden attack remains inactive for the whole diagnostic. Data partitions,
initialization, learning-rate schedule and keyed sampling are paired.

"Original selection" below means the existing survivor-aware, receiver-local
ranking and selected-model averaging policy. Its output count is dynamic. In
particular, it falls back to one candidate in the clean cases here; these rows
must not be described as averaging 33 models when their recorded count is one.
The best-single counterfactual takes the highest-ranked member of the original
selected set instead of their average. It is explicitly recorded as a diagnostic
intervention, not claimed as an approved Byzantine defense or novel method.

| Input rule | Sigma | Proximal coefficient | Five-round accuracy | Mean selected count | Mean selected hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediate predecessor | 0 | 0 | 29.472% | 1.0 | 1.0 |
| Immediate predecessor | .01 | 0 | 32.220% | 1.0 | 1.0 |
| Original selection | 0 | 1 | 10.453% | 1.0 | 17.9 |
| Original selection | .01 | 1 | 11.257% | 33.0 | 33.0 |
| Best single counterfactual | 0 | 1 | 10.131% | 1.0 | 18.6 |
| Best single counterfactual | .01 | 1 | 20.971% | 1.0 | 17.4 |
| Original selection | 0 | 0 | 10.165% | 1.0 | 16.5 |
| Original selection | .01 | 0 | 11.169% | 33.0 | 33.0 |

Count and hop statistics exclude the first ring round to remove bootstrap.
Hop is the number of clockwise client activations separating receiver and
selected sender. Larger hops indicate older positions in the sequential ring.
Noisy handoff exceeding clean in this single short trial is not evidence that
noise generally improves learning.

## Supported interpretation

1. The network and five-step optimizer can learn under immediate handoff.
2. The original historical-selection path remains near chance even with both
   noise and the proximal term removed. Neither is a sufficient explanation.
3. Reducing noisy selection to its best single copy improves the result here,
   but does not solve the clean historical-selection failure. Averaging count
   alone therefore cannot be the complete explanation.
4. The poor-learning rows repeatedly choose models from substantially earlier
   positions in the ring. Loss-based selection and loss of recent training
   progress are implicated. These measurements do not prove a universal causal
   theorem about snapshot age or isolate every aspect of the selection package.

The current CART v2 correction changes the classifier while preserving its
proposal backbone. Its acceptance checks can improve local probes without
repairing a weak representation or restoring every discarded training update.
A future candidate should test class-aware preservation of useful learning
progress through the whole handoff, rather than merely increasing the existing
head-correction strength. That is a design hypothesis, not established novelty
or a promised accuracy improvement. A base-selector change also needs new
matched controls; it cannot be evaluated as a CART-only change against stale
controls. Changes confined to CART can retain compatible unchanged controls
with explicit version/source provenance.

The [public validation receipt](validation/CART_HANDOFF_5R_20261028.json) records
all eight cases and their source/data identities. All eight passed independent
CPU inference of the saved final models, with exact final node accuracies. Eleven
targeted tests passed. The
test suite checks the counterfactual's chosen-copy behavior and pre-attack setup.
This remains a single-seed, IID, five-round development experiment. It does not
demonstrate non-IID efficacy, robustness after attack onset, or thesis novelty.

```bash
# Local separate diagnostic; never use the active HPC campaign output here.
python scripts/probe_cart_handoff.py \
  --output "$PWD/.papermerge/local-tests/cart-handoff-factorial-5r-new" \
  --nodes 100 --rounds 5 --workers 2

python scripts/verify_cart_handoff_probe.py \
  --output "$PWD/.papermerge/local-tests/cart-handoff-factorial-5r-new"
```

The diagnostic uses a separate output, local GPU0, two bounded 3 GiB workers
and an exclusive project GPU lease. It refuses to replace an existing diagnostic
plan. Original training and active runtime files are unchanged.
