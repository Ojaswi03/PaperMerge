# CART isolated runner: ordered implementation and validation

See the newer [resource and recovery gates](CART_RESOURCE_AND_RECOVERY_GATES.md)
for implemented kernel limits, explicit trainer-refusal recovery and the GPU
coordinator path. The measurements below describe the preceding v1 stage.

## Status

| Ordered stage | Result |
|---|---|
| 1. Isolated runner integration | CPU prototype implemented; no GUI/queue exposure |
| 2. Adversarial messages and recovery | Targeted unit and real-process checks passed |
| 3. GPU correctness and memory | Check stopped before computation: 307 MiB free; 7,168 MiB required |
| 4. Bounded learning diagnostic | **Not started: prerequisite gates incomplete** |
| A/B/C/D | **Not started** |

Revision: `cart_isolated_verified_steps_v1`. This advances the
[replicated TRAIN verification prototype](CART_REPLICATED_TRAIN_VERIFICATION.md).
It does **not** establish unrestricted full-node Byzantine resilience or a
production-ready distributed system.

## What is now implemented

```text
Trusted coordinator: schedule, committed TRAIN root, model and checkpoint
        |
        +-- bounded JSON request: weights/hash, indices, augmentation, context
        |
        +--> 9 independent non-owner replica processes
        |      Linux user/PID/network namespaces
        |      read-only code and TRAIN fixture; private /tmp and /proc
        |      no host home credentials; no network or GPU devices
        |      private Ed25519 key remains inside each process
        |
        <-- 5 identical authenticated replies required
        |
        +-- select one complete snapshot using certified balanced TRAIN CE
        +-- compute existing balanced CE + local EBM + fixed anchor update
        +-- commit a step checkpoint atomically
        +-- after 50 verified full-data steps: Hidden attack, then link noise
```

The coordinator is a trusted research harness, not another Byzantine participant.
The helper service has no attacker-ID oracle. Fault injection controls the
test endpoints, not the certificate acceptance rule.

| Module | What it does |
|---|---|
| `replica_wire.py` | Finite float32/base64 JSON; 4 MiB frames and at most 117,706 parameter coordinates; no peer pickle |
| `sandbox_replica.py` | Independently evaluates a committed TRAIN request, then signs its result; no helper SGD |
| `isolated_replicas.py` | Launches CPU-only `bwrap` sandboxes; bounds pipe reads/writes by deadline; kills/reaps only owned failed workers; never falls back to unsandboxed execution |
| `verified_cart_runner.py` | Connects certified selection/helpers to the existing objective; strict frozen-anchor handoff; deterministic batches; atomic step checkpoints |
| `check_isolated_replicas.py` | Explicit synthetic fault and process-restart checks; not a campaign launcher |
| `test_isolated_cart_runner.py` | Wire, scheduling, malicious update, corrupted model, checkpoint and lazy-import contracts |

The former source modules, GUI queue, approved Hidden transformation and
scientific loss implementations were not edited. Existing evidence remains
separate. There are no W&B calls.

## Mathematics and training semantics

The coordinator calls the existing `owner_objective_gradients` implementation:

\[
L_i(\theta)=\frac{1}{|C|}\sum_{c\in C}CE_c(\theta)
+\sigma_e^2\|\nabla CE_i(\theta)\|_2^2
+\frac{0.01}{2}\|\theta-\theta_{selected}\|_2^2.
\]

EBM is optional. When enabled, its coefficient is exactly the absolute channel
variance; there is **no lambda**. The EBM inner objective is local CE, not the
balanced mean CE or the anchor. Helpers certify their independent CE gradients;
the original nested-tape implementation supplies the local Hessian-gradient
correction. The frozen selected snapshot remains unchanged across steps.

The scheduler retains five complete epochs, batch 512, the final batch of 392,
and 50 updates per activation. It uses the existing keyed data-order,
augmentation and helper-sampling streams and the existing round-level LR decay.
Actual research scheduling rejects any local dataset other than 5,000 examples.
Only explicitly synthetic fixtures can use the unit-batch override.

SGD has zero momentum and no clipping/weight decay. The coordinator recreates
the stateless SGD optimizer each step; because LR is fixed within the activation
and there are no slots or iteration-dependent schedules, this produces the
same update as resetting that optimizer once per activation.

Outbound delivery reuses `snapshot_links`: the existing delayed Hidden
transformation at R20, then independent absolute Gaussian noise per link.
This does not claim an exact reproduction of the attack source paper.
An activation cannot deliver before its complete 50-step lineage is validated.

## Real-process evidence

Evidence file:
`gui/worker_state/CART_ISOLATED_RUNNER_RECOVERY_FINAL.json`.

The checker used the actual 117,706-parameter CNN with **synthetic** 16-example
batches. It is not a shortened-epoch CIFAR experiment and measures no accuracy.

| Injected faulty identity | Behavior | Observed handling |
|---|---|---|
| 0 | Signed poisoned gradient/false CE | Excluded from matching honest certificate |
| 1 | Invalid signature | Rejected |
| 5 | Stopped replying | Actual pipe deadline expired; process killed and reaped |
| 7 | Exited | Closed pipe detected; subsequent requests marked unavailable |

For an owner-2 request, matching honest signers were **3, 4, 6, 8, 9**.
The certified gradient and CE exactly matched the independently calculated
reference. Initial ten-process startup took **27.0541 seconds**. The first
fault-injected gradient request took **3.6735 seconds**, including the deliberately
unresponsive peer's deadline; the later CE request took **0.0770 seconds**.
These tiny-request timings are not a full-data runtime estimate.

The integrated check then:

1. Selected the nearest predecessor from identical snapshots via certified scores.
2. Committed one synthetic step and saved the checkpoint.
3. Committed a second step as the uninterrupted reference.
4. Rejected a deliberately wrong proposed active-node update without advancing.
5. Terminated all original replicas and started a fresh set with new private keys.
6. Restored the saved state and replayed the second step.

The resumed parameter hash and objective were **bit-identical** to the CPU
uninterrupted reference. Three synthetic steps were committed across the two
paths. The rejected proposal was evaluated but not committed. This is not a
claim of cross-device or arbitrary distributed bitwise reproducibility.

An additional unit test combines four poisoned helper endpoints with a
near-constant-class malicious outbound candidate. Certified balanced selection
rejects that candidate in this fixture, and the resulting step equals the
honest-helper reference. This is one adversarial example, not a guarantee that
all malicious models lose the selection comparison.

## Checkpoint/recovery scope

| Restored state | Validation |
|---|---|
| Current parameters and frozen anchor | Tensor hashes; exact selected-weight initialization |
| Ten trained-node states | Preserved separately from outgoing attacked/noisy snapshots |
| Ten five-predecessor memories | Sender/round/parameter/channel entries preserved |
| Round, node and optimizer-step cursor | Restored without resetting the activation |
| Selection and completed-step records | Retained, including certified response metadata |
| Randomness | Keyed from saved seed/cursor; no mutable global RNG to reconstruct |
| Scientific code/config/model/init/data | Fingerprints, model semantics, initialization hash and TRAIN root checked |
| Execution | TensorFlow version, CPU policy and relevant thread/determinism settings bound |
| Keys | Private keys are **not** saved; trusted coordinator explicitly starts a new pinned membership session |

Checkpoint files use temporary-file write, file fsync, atomic replace and
directory fsync. An injected pre-replace disk failure left the old checkpoint
unchanged. Corrupted/truncated checkpoint files were rejected. A malformed or
mismatched proposed step restores the in-memory model and does not advance.

A test initially caught an overly strict model comparison: Keras auto-generated
new cosmetic layer names on reconstruction. This was fixed for the supported
Sequential topology by comparing semantic configuration without cosmetic names,
while still checking parameter initialization, architecture settings and code
hashes. The recovery test subsequently passed.

## GPU and OOM boundary

The ordered GPU numerical check was attempted only after CPU tests. It exited
before computation because **307 MiB** was free; the unchanged guard requires
**6,144 MiB allocation cap + 1,024 MiB headroom**. No other process was stopped.
The numerical GPU check did not pass and produced no successful proof file.

The new namespace runner deliberately remains CPU-only. Freeing VRAM would
allow the separate kernel check, **not automatically validate GPU execution
inside this runner**. There is no silent mixed CPU/GPU continuation.

The sandboxes protect filesystem/process/network boundaries, but do not yet
have a verified aggregate cgroup RAM/CPU resource limit. Pipe deadlines and
bounded payloads are not a proof against arbitrary memory-exhaustion/fork-bomb
attacks. Full-data replication and checkpoint size also need a resource budget
before a long run. These are genuine remaining readiness gates.

## Tests and integrity

Full suite: **629 passed, 2 skipped, 644 subtests passed, 3 warnings in 87.20 s**.
The final focused run (isolated runner, certificates and existing integration
contracts, including the added malicious-outbound fixture) passed **55 tests**
with two dependency warnings in **8.71 s**.
The skips are the two optional display tests. Compilation passed. The lazy-import
audit inspected **246** source files and found **zero** retained heavy imports
or needed rewrites. Final source fingerprints in the real-process proof match.

Commands executed:

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/cart-isolated-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q -rs tests
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py --report gui/worker_state/cart_isolated_runner_lazy_imports.json
environment/basil-noise-env/bin/python scripts/check_isolated_replicas.py --check --output gui/worker_state/CART_ISOLATED_RUNNER_RECOVERY_FINAL.json
environment/basil-noise-env/bin/python scripts/check_replicated_training.py --check --device GPU --output gui/worker_state/CART_ISOLATED_STAGE_GPU_NUMERICAL_CHECK.json
```

The last command failed the VRAM guard as described above; it is not counted
as a passed GPU check. Namespace checks require Linux `bwrap` permission and
fail closed when the surrounding execution sandbox prohibits user/network namespaces.

The new hash baseline is `gui/worker_state/cart_isolated_runner_before.json`,
covering saved IID/non-IID results, plots and papers. No such artifacts were
changed by the implementation: **24,599 before, 24,599 after, 24,599 matched,
zero changed**. All owned sandbox workers were reaped; the final process check
found none still running.

## Remaining decision gates

1. GPU backend integration and numerical/memory checks—not just available VRAM.
2. Hard aggregate resource limits and a full-data/checkpoint storage budget.
3. Liveness policy for a malicious active trainer: the current optional proposal
   check rejects invalid updates and stops; it does not silently substitute a
   trusted update or claim continued progress under arbitrary refusal.
4. Only after those gates: a bounded learning diagnostic labelled as replicated
   TRAIN verification, not the original exclusive one-class information model.

**NOT READY for the full-node A/B/C/D research comparison.** The CPU integration
and restart result are real progress; they are not evidence of improved accuracy
or unrestricted Byzantine protection. The old training protocol, completed
results, queue and W&B project remain untouched.
