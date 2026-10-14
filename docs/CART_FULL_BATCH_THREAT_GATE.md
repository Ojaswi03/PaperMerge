# CART full-batch EBM, channel and adversarial-helper gate

This is a bounded **correctness replay**, following the
[one-round clean learning diagnostic](CART_VERIFIED_LEARNING_DIAGNOSTIC.md).
It is not A/B/C/D, a convergence experiment or a claim that twenty rounds were
executed. No TEST images are loaded and no W&B run is created.

## Fixed replay design

| Item | Definition |
|---|---|
| Starting evidence | Completed clean one-round checkpoint, preserved byte-for-byte |
| Boundary fixture | Its trained node models are placed in explicitly constructed logical R19 snapshots |
| Incoming links | Independent absolute Gaussian noise, σₑ=0.010; Hidden inactive at R19 |
| Active trainer | Node 0, logical R20; no fabricated R1–R19 metrics |
| Local data | Complete 5,000 class-0 TRAIN examples |
| Optimizer | Existing zero-momentum SGD; unchanged round decay gives LR 0.025 at R20 |
| Epochs / batches | Five full epochs; nine batches of 512 plus one of 392 per epoch |
| Objective | Mean available-owner CE + local-CE EBM + fixed μ=0.01 selected-snapshot anchor |
| Honest reference | One complete 50-update activation with ten honest verifier processes |
| Matched fault replay | Same activation with four faulty verifier identities |
| Fault identities | 0: poisoned reply; 1: forged signature; 5: withheld reply; 7: process exit |
| Active-trainer fault checks | Incorrect update at step 0; explicit timeout at step 1; verified coordinator recomputation |
| Outbound model | Existing Hidden transformation at R20, then independent Gaussian noise on five links |
| Next receiver | Node 1 evaluates actual refreshed memory; no extra local activation |
| Total optimizer work | Exactly 100 committed updates across the two matched activations |

Fault injectors know which identities to corrupt. The certificate consumer does
not use the attacker list as an honesty oracle. An owner is excluded from its
own nine-member committee; depending on which owner is queried, five or six
honest replies remain under these four faults. Five matching authenticated
results are still required.

## Mathematical checks

The unchanged CART objective is

\[
L=\frac1{10}\sum_{c=0}^{9}CE_c
+\sigma_e^2\|\nabla CE_0\|_2^2
+\frac{0.01}{2}\|\theta-\theta_{selected}\|_2^2.
\]

At both an actual 512-example batch and the actual 392-example final batch,
the checker compares the applied gradient against an independent nested-tape
construction. Certified CPU helper gradients enter that reference through a
linear surrogate with the same derivative at the current parameter state.
This avoids incorrectly demanding bitwise CPU/GPU equality. EBM differentiates
receiver-local CE, not the balanced helper mean or the anchor.

The scalar objective is independently reconstructed from the receiver CE,
certified helper CEs, gradient-norm penalty and frozen-anchor penalty.
Gradient tolerance is fixed beforehand at `atol=2e-6, rtol=1e-4`; objective
tolerance is `atol=2e-6, rtol=1e-6`. These tolerances never relax the helper
certificate's exact-byte matching rule.

Legacy lambda values 1 and 999999 must produce **identical loss, gradients and
prospective parameter updates**. A CE-only objective evaluation at the same
weights/data confirms that the extra applied-gradient contribution equals the
measured EBM correction. This is an objective check, not a third research run.

## Hidden versus Gaussian channel

The existing project's Hidden transformation (not claimed as an exact paper
reproduction) is checked algebraically, per parameter tensor w:

\[
m=-1.4w+z,\qquad
z\sim N\left(0,[0.05\max(\operatorname{std}(w),0.01)]^2I\right),
\]
\[
w_{attack}=0.45w+0.55m.
\]

Only after this transformation is applied, each directed link receives
\(\epsilon\sim N(0,0.0001I)\), so \(w_{received}=w_{attack}+\epsilon\).
Before R20 the attack transformation is absent, but channel noise is active.
This Hidden implementation contains its own small random component; that does
not mean the separate `gaussianAttack()` implementation is being selected.
Neither attack nor channel parameters are retuned.

Independent keyed-RNG reconstruction checks exact received arrays, hashes,
link independence, σₑ=0.010 and attack inactivity at R0/R19 versus activity at
R20. Empirical coordinate standard deviation is checked within 0.0001 of
0.010 on the actual 117,706-coordinate vectors. The same underlying draws must
produce identical received arrays in the matched honest/fault replays.

## Safety, evidence and scope limits

GPU allocation remains capped at 6 GiB with startup headroom; CPU replicas keep
their 2 GiB RAM / zero-swap / 64-task / one-core limits. Full snapshots are saved
after selection and each epoch. On a failure, `failure.json` identifies the
case, cursor, exception and last saved checkpoint. No remaining steps are invented.
This checker refuses to overwrite its output directory and has no automatic retry.

Kernel counters are recorded for live scopes. Deliberately closed peers are
marked `memoryCountersAvailable=false`, not assigned fabricated zero OOM counts.
This is still a trusted-coordinator, replicated-TRAIN architecture, not a privacy
guarantee, general Byzantine consensus or arbitrary hostile-kernel protection.

```bash
# Zero training by default
environment/basil-noise-env/bin/python scripts/check_verified_cart_full_batch.py

# Explicit bounded replay; NOT a campaign launcher
environment/basil-noise-env/bin/python scripts/check_verified_cart_full_batch.py --check
```

Evidence directory: `gui/worker_state/cart_full_batch_threat_v1/`.
Only a successful `summary.json` is a passing gate. An incomplete directory is
not success. The completed outcome is recorded below.

## Pre-run tests

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/cart-resource-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_verified_cart_full_batch.py tests/test_cart_owner_integration.py tests/test_replica_resource_gates.py
environment/basil-noise-env/bin/python -m compileall -q scripts/check_verified_cart_full_batch.py tests/test_verified_cart_full_batch.py
```

39 focused tests passed, with two dependency deprecation warnings, in 15.11 s.
Compilation passed. A fresh SHA-256 baseline covers 24,678 saved research,
plot and paper files, including the completed one-round diagnostic.

## Completed results

The bounded replay **passed in 420.20 seconds (7.00 minutes)**. This duration
includes both activations, process startup, verification and checkpoint writes;
the CPU unit suite also ran concurrently, so it is not an isolated speed benchmark.
No full communication round or test-set evaluation was executed.

| Check | Measured outcome |
|---|---|
| Full-data local training | 2 × 5 epochs × 5,000 examples; 100 committed SGD updates |
| Partial batches | One 392-example batch per epoch, not dropped |
| Honest versus four faulty helper identities | All 50 output hashes and objectives matched exactly |
| Incorrect active-trainer proposal / explicit timeout | Both recovered by verified coordinator recomputation |
| Selected input / outbound received arrays | Exactly matched between replays |
| EBM legacy lambda 1 versus 999999 | Identical losses, gradients and prospective updates at both tested batch sizes |
| Independent nested-tape gradient, batch 512 | Maximum absolute error 6.52 × 10⁻⁹ |
| Independent nested-tape gradient, batch 392 | Maximum absolute error 6.75 × 10⁻⁹ |
| R20 outgoing channel empirical coordinate standard deviations | 0.0099698–0.0100267 across five distinct keyed links |
| Hidden boundary and order | Inactive at R0/R19; active at R20, before channel noise |
| Numerical failures / observed OOM | None |
| Maximum recorded live-replica memory peak | 1,698,033,664 bytes, about 1.58 GiB, below 2 GiB cap |
| TensorFlow GPU allocator peak | 346,774,016 bytes; this is not total process/driver VRAM |
| Test images / W&B / A/B/C/D | None loaded / none started / none started |

All available kernel OOM counters were zero. Final counters for deliberately
closed peers 5 and 7 were unavailable, as explicitly recorded; their absence is
not evidence of zero events. The bounded success does not guarantee overnight
OOM freedom or CPU/GPU numerical equivalence.

At the tested 512/392 batches, the EBM correction norms were 0.00259992 and
0.00250689. Relative to the **ordinary receiver-local CE gradient**, the ratios
were 0.00121201 and 0.00118273 (about **0.12%**). This is a measured small,
nonzero correction, not proof of improved noise robustness. The coefficient
was the float32 representation of 0.0001; no multiplier was introduced.

### Important negative finding: corrupted snapshot selected

Node 1 selected sender 0's **actually Hidden-corrupted, then channel-noisy**
R20 model in both replays. The independent verified TRAIN scores were:

| Candidate sender | Snapshot round | Hidden-corrupted? | Balanced TRAIN CE | Receiver-local CE |
|---|---:|---|---:|---:|
| 6 | 19 | No | 3.651538 | 1.950953 |
| 7 | 19 | No: attack warm-up | 2.992471 | 1.951410 |
| 8 | 19 | No | 2.508577 | 2.126618 |
| 9 | 19 | No | 2.305581 | 2.204807 |
| **0, selected** | **20** | **Yes** | **2.303237** | **2.317119** |

The selector obeyed its configured balanced-loss rule; forged helper responses
did not cause this choice. **Correctly measuring loss does not certify the
honesty or usefulness of a candidate.** Scores close to log(10) do not establish
class retention. We did not evaluate global accuracy here, so no downstream
accuracy improvement or degradation is inferred.

This is the CART balanced-owner selection adaptation, not paper BASIL's
receiver-local selection. These explicitly constructed R19/R20 fixtures came
from the preserved one-round checkpoint; they are not a 20-round trained
trajectory. Their failure to reject this corruption must not be extrapolated
into a universal attack success rate.

### Validation commands and results

```bash
MPLCONFIGDIR=/tmp/cart-resource-mpl environment/basil-noise-env/bin/python scripts/check_verified_cart_full_batch.py --check
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/cart-resource-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q -rs tests
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py --report gui/worker_state/cart_full_batch_threat_lazy.json
```

| Validation | Actual result |
|---|---|
| Full suite | **649 passed, 2 skipped, 644 subtests passed**, 3 warnings; 105.79 s |
| Skips | Two optional display-dependent GUI checks; not claimed passed |
| Warnings | Two dependency deprecations and one existing Keras input-structure warning |
| Compile | Passed |
| Lazy-import audit | 255 files; zero imports changed or retained as eager third-party imports |
| Saved-artifact SHA-256 verification | **24,678 before / 24,678 after / 24,678 matched / 0 changed** |
| Process cleanup | No checker/training/helper processes remained; GPU had 11,303 MiB free afterward |

The artifact baseline is `gui/worker_state/cart_full_batch_threat_before.json`
and includes all saved IID/non-IID results, plots and papers. Verification used:

```bash
environment/basil-noise-env/bin/python - <<'PY'
import json, sys
import scripts.verify_saved_evidence as v
p = 'gui/worker_state/cart_full_batch_threat_before.json'
v.TREES = tuple(json.load(open(p))['trees'])
sys.argv = ['verify_saved_evidence', '--manifest', p]
v.main()
PY
```

Raw evidence: `gui/worker_state/cart_full_batch_threat_v1/summary.json`,
`honest_evidence.json`, `four_faults_evidence.json`, and the per-epoch checkpoints.
The checkpoint/input/source hashes remain available for reproduction.

### Readiness decision

**NOT READY FOR FULL A/B/C/D.** Full-batch EBM/channel algebra and the injected
helper-fault recovery gate now pass under the replicated-TRAIN,
trusted-coordinator architecture. They do not solve the observed snapshot
filtering or learning limitations. The previous one-round learning result
remains 13.90% Node-9 accuracy, with five zero-recall classes; it was not rerun
or replaced by this correctness gate. No algorithm was retuned, and no longer
experiment was launched.
