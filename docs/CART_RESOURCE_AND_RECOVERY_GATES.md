# CART resource limits, trainer recovery and GPU gate

## Current outcome

| Gate | Result |
|---|---|
| Kernel resource enforcement | Passed controlled memory-limit test |
| Full TRAIN replica memory | 50,000 images loaded; no OOM; one forward evaluation, no SGD |
| Isolated CPU integration | Passed with four faulty replica endpoints |
| Explicit trainer-refusal recovery | Matched the honest update, without skipping a step |
| Process restart | Resumed next CPU step bit-identically |
| GPU-coordinator integration | Passed bounded synthetic check after freeing VRAM and fixing checker oracle placement |
| Learning diagnostic / A/B/C/D | **Not started** |

Revision: `cart_isolated_verified_steps_v2`. This continues the
[isolated runner work](CART_ISOLATED_RUNNER_VALIDATION.md). Existing research
engines, scientific loss functions, Hidden attack, completed evidence and GUI
queue were not changed.

## 1. Hard resource limits

Each replica now runs in its own transient, user-owned systemd scope, outside
the network-disabled `bwrap` namespace boundary. Limits apply to the whole
scope, including descendants; the child cannot increase them from its sandbox.
There is no uncapped or unsandboxed fallback.

| Resource | Limit |
|---|---:|
| RAM per replica | 2 GiB |
| Swap per replica | 0 |
| Tasks per replica | 64 |
| CPU per replica | At most one core's quota |
| Pool population | Exactly ten; duplicate pools prevented by an OS lock |
| Sum of replica RAM ceilings | 20 GiB |
| Required available RAM at startup | 24 GiB, including 4 GiB outside-pool headroom |
| Checkpoint size | At most 256 MiB for this diagnostic runner |
| Free disk before checkpoint write | Temporary checkpoint size plus 1 GiB |

The coordinator reads **kernel** `memory.max`, `memory.swap.max`,
`memory.oom.group`, `pids.max` and `cpu.max` to verify effective settings. Scope
names are unique and restricted to the PaperMerge replica prefix; cleanup never
targets another application's scope. `OOMPolicy=kill` sets whole-scope OOM
handling on this host's systemd 255. An initially attempted unsupported property
was rejected before execution, then corrected to the host-documented setting.

### Actual enforcement test

A disposable Python process attempted a **128 MiB** allocation inside a
**64 MiB** scope. It was killed with exit code **-9** before allocation completed.
The kernel settings were read back before the attempt. This was an intentional,
bounded resource-limit test—not an OOM in a research run or the GPU worker.

Evidence: `gui/worker_state/CART_KERNEL_RESOURCE_GATES.json`.

### Full-data memory check

One CPU replica loaded the existing complete CIFAR-10 TRAIN cache and evaluated
one 16-example selection batch:

| Measurement | Result |
|---|---:|
| Training images | 50,000 |
| Images per class | 5,000 |
| Test images loaded | 0 |
| Peak cgroup memory | 1,519,034,368 bytes (approximately 1.42 GiB) |
| RAM cap | 2,147,483,648 bytes |
| Kernel `oom` / `oom_kill` events | 0 / 0 |
| Optimizer updates | 0 |

Evidence: `gui/worker_state/CART_FULL_TRAIN_REPLICA_MEMORY.json`.
This establishes that the full TRAIN corpus fits in one capped replica during
this forward check. It is not a full-pool training throughput benchmark or a
promise that unrelated host workloads can never exhaust machine memory.

The checkpoint guard preserves the previous committed checkpoint on inadequate
disk space. The 256 MiB diagnostic cap is **not** permission to launch a long
campaign whose projected telemetry/checkpoint size has not been budgeted.

## 2. Explicit recovery from active-trainer refusal

Two policies are supported and recorded in the configuration:

| Policy | Behavior |
|---|---|
| `abort` | Existing behavior: reject an invalid proposal or refusal without progress |
| `verified_coordinator_recompute` | Trusted coordinator recomputes the same step from committed TRAIN data and certified helper gradients |

The recovery policy is opt-in. It handles reported bounded-transport outcomes
`timeout`, `exit`, `refused`, `invalid_message`, and a proposed update that differs
from the canonical computation. Receipts record the failure and recovery. It
does not skip the node, discard an epoch, copy an unverified update or change a
coefficient. Missing helper quorum still aborts; recovery never lowers the
five-signature threshold.

The unchanged gradient is:

\[
\frac{1}{|C|}\sum_{c\in C}\nabla CE_c
+2\sigma_e^2 H_i\nabla CE_i
+0.01(\theta-\theta_{selected}).
\]

EBM is optional, has no lambda, and uses the receiver-local CE inside its
gradient-norm term. The selected anchor stays frozen. Five full local epochs,
batch 512 with final batch 392, SGD and the existing round-level LR schedule
remain unchanged. Hidden corruption and link noise remain outbound operations.

**Trust qualification:** this recovery relies on the trusted coordinator and
replicated TRAIN evidence. It is not decentralized consensus, privacy-preserving
data isolation, or a proof that an arbitrary malicious owner trains honestly.
The coordinator already performs canonical local optimization in this research
runner. The refusal API consumes an explicit transport-failure outcome; the
current physical RPC endpoints are helper replicas, not a newly deployed fleet
of remote active trainers.

### Measured recovery checks

The real-process CPU checker used the 117,706-parameter CNN and synthetic
16-example batches. It exercised poisoned results, forged signatures, an actual
withheld reply and a process exit. Five honest matching signatures remained.

It committed four synthetic test steps across the reference/restart/recovery
paths. Restoring after all replica processes had been replaced reproduced the
next CPU parameter state exactly. The explicit timeout-recovery step also
matched the honest step exactly. These are correctness checks, **not a learning
trajectory or an accuracy improvement claim**.

Evidence: `gui/worker_state/CART_RESOURCE_LIMITED_RECOVERY_CPU.json`.
Recorded source hashes match the implementation used for these checks.

## 3. GPU coordinator, CPU-only untrusted helpers

The runner now accepts an explicit coordinator device. GPU placement must match
the actual model variables; the existing float32, TF32-off, 6 GiB-cap policy is
checked. Canonical initialization is produced on CPU, then copied to the
coordinator model. The device and helper backend are recorded and bound into
checkpoints; changing device does not silently resume a checkpoint.

Untrusted helper sandboxes deliberately remain CPU-only and cannot access GPU
devices. This is a **GPU coordinator / CPU verifier** design, not an all-GPU
execution claim. All four future paired conditions would need the same declared
backend policy. Exact replica certificates compare homogeneous CPU helper
computations; no CPU/GPU numerical equivalence is assumed.

The actual integrated GPU attempt failed before starting replicas or optimizer
updates:

```text
GPU has 307 MiB free; at least 7168 MiB is required
for the 6144 MiB cap plus safety headroom.
```

The separate GPU query reported 11,688 MiB used, 307 MiB free and 12,282 MiB
total. Its compute-process query returned no owner information in WSL. No
unknown process was stopped, and the guard was not weakened.

### Follow-up: GPU gate passed, 2026-10-13

Windows GPU counters subsequently identified `RazerAxon.Player` (PID 30996)
using approximately 10,997 MiB of dedicated GPU memory. With explicit user
approval, only that identity-checked process was stopped. Free VRAM rose to
11,303 MiB. The 6,144 MiB TensorFlow cap and 1,024 MiB headroom rule were unchanged.
The wallpaper application may be reopened later; no startup settings were changed.

The first newly unblocked attempt exposed a **checker placement defect**:
constructing the reference model's variables on CPU did not force its forward
and backward operations to stay on CPU when a GPU was visible. The helper
replicas are deliberately CPU-only, so this was not a valid bitwise oracle.
`scripts/check_isolated_replicas.py` now encloses the reference calculation in
`tf.device('/CPU:0')`, records the oracle backend and checker hash, and provides
the compared digests/losses on a mismatch. No objective, gradient, certificate
threshold or numerical tolerance was changed. The initial attempt failed at
the reference comparison, before any optimizer update.

The corrected integrated check completed successfully:

| Actual GPU-check result | Evidence |
|---|---|
| Coordinator / helpers / oracle | GPU / CPU / CPU |
| Gradient and selection CE certificates | Exact CPU-reference matches; signers 3, 4, 6, 8, 9 |
| Injected helper faults | Poisoned result, forged signature, withheld reply, process exit |
| Incorrect active update | Rejected |
| Committed synthetic updates | 4 across reference, restart and recovery paths |
| Next update after process replacement/checkpoint restore | Bit-identical on the same GPU backend |
| Explicit trainer timeout recovery | Matches the honest GPU update |
| OOM / non-finite aborts in successful check | None observed |
| Research training / W&B / queue execution | Not started |

Evidence: `gui/worker_state/CART_RESOURCE_LIMITED_RECOVERY_GPU.json`.
GPU memory sampled during the check showed 4,952 MiB free; this is one sample,
not a measured minimum or a long-run memory guarantee. CPU/GPU equivalence
is **not** claimed. This checks 16-example synthetic batches, not full-data
batch-512 throughput or a complete five-epoch activation. Full-pool learning,
checkpoint growth and long-run resource behavior remain to be measured before
an A/B/C/D campaign.

The bounded check command is:

```bash
environment/basil-noise-env/bin/python scripts/check_isolated_replicas.py --check --coordinator-device GPU
```

This command validates the backend using synthetic steps; it does not start a
CIFAR research campaign. Its synthetic GPU correctness gate has now passed;
the bounded learning diagnostic is the next separate stage.

## 4. Verification and files

| Check | Actual result |
|---|---|
| Full suite | 638 passed, 2 skipped, 644 subtests passed, 3 warnings; 99.16 s |
| Skips | Two optional display-dependent tests |
| Compileall | Passed |
| Lazy-import audit | 250 source files; zero heavy imports retained or rewrites needed |
| Saved research/paper integrity | 24,599 before/after; all matched; zero changed |
| Research runs / W&B / GUI queue | Not started |

Main additions: `basil_core/replica_resources.py`,
`scripts/check_replica_resource_limits.py`, `scripts/check_replica_train_memory.py`,
and `tests/test_replica_resource_gates.py`. The isolated endpoint, step coordinator
and checker were extended; the scientific objective implementation was not.

Commands executed:

```bash
environment/basil-noise-env/bin/python scripts/check_replica_resource_limits.py --check --output gui/worker_state/CART_KERNEL_RESOURCE_GATES.json
environment/basil-noise-env/bin/python scripts/check_replica_train_memory.py --check --output gui/worker_state/CART_FULL_TRAIN_REPLICA_MEMORY.json
environment/basil-noise-env/bin/python scripts/check_isolated_replicas.py --check --output gui/worker_state/CART_RESOURCE_LIMITED_RECOVERY_CPU.json
environment/basil-noise-env/bin/python scripts/check_isolated_replicas.py --check --coordinator-device GPU --output gui/worker_state/CART_RESOURCE_LIMITED_RECOVERY_GPU.json
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/cart-resource-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q -rs tests
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py --report gui/worker_state/cart_resource_gates_lazy_imports.json
```

The earlier GPU invocation failed its memory guard; a later attempt failed the
oracle comparison described above. The corrected invocation passed and created
the success-proof file. Follow-up focused regression tests completed with
**39 passed, 2 warnings in 18.13 seconds**:

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/cart-resource-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q -rs tests/test_isolated_cart_runner.py tests/test_replica_resource_gates.py tests/test_replicated_verification.py
environment/basil-noise-env/bin/python -m compileall -q scripts/check_isolated_replicas.py
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py --report gui/worker_state/cart_gpu_gate_lazy_imports.json
```

Compilation passed; the lazy-import audit again found zero retained heavy imports
across 250 source files. Historical research/paper hashes were rechecked:
24,599 matched, zero changed. The full-suite count above is the prior resource-gate
run, not a new full-suite execution for this checker-only fix.

Kernel/sandbox checks required permission outside the agent's
restrictive execution sandbox; no permanent system service was installed.

**Status: measured CPU resource/recovery and bounded GPU integration gates passed;
learning and A/B/C/D not started. No production-readiness claim.**
