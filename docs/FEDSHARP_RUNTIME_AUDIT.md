# Runtime audit of the active 500-round continuation

Reviewed against commit `3f61a954` and the user's two-server activity reports.
This is a source audit plus isolated local CPU measurements, not a live HPC
function profile. No training worker, scientific source, checkpoint, frozen
plan, resource budget or host security policy was modified during the audit.

## Findings in priority order

### 1. Scheduler handover prevents available GPUs from accepting more jobs

The user's diagnostics confirm one original worker holding each themachine GPU
lease while its fleet agent reports `waiting for existing GPU lease`. This is
an operational design limitation introduced by the handover, not evidence that
other users exhausted GPU memory. The latest hal9k report instead says `waiting
for memory` on both GPUs; it no longer reports a lease wait on GPU0.

The attempted `pidfd_getfd` transfer was denied on both hosts before either
dispatcher was signalled. Repeating it does not solve that denial. The existing
agent retries ordinary lock acquisition when old holders finish. A durable
future design needs cooperative descriptor transfer or a persistent lease
owner designed before launch. Do not delete locks or weaken host security.

Source: `basil_core/fleet_runner.py`, admission loop; `live_gpu_scheduler.py`,
`lease_available`. This limits aggregate concurrency, independently of the
time spent in each individual activation.

### 2. Zero noise coefficient still uses the nested-gradient training graph

`cart_admitted_continuity_candidate.AdmittedWorker.train` passes coefficient
zero for the owned route. `cart_local_prior.LocalPriorWorker` selects its
`robust` branch from the configuration switch, not the realized coefficient.
Its nested gradient graph remains present with a runtime zero coefficient.
The same static robust branch exists in `sharp_worker.objective_gradients`.

An isolated CPU experiment compared the current prior-adjusted objective with
a zero-coefficient path computing CE gradients once and adding the mu=1
proximal gradient, while retaining the gradient-squared diagnostic. Model
weights were not updated. Both paths were warmed and timed nine times using
MNIST batch 14 and CIFAR-10 batch 80, fixed synthetic inputs, nonzero anchor
offsets, and a uniform prior. The original coefficient was a tensor argument,
not a constant captured inside the compiled graph.

| Dataset | Current median | Candidate median | Compared gradients and scalar values |
| --- | ---: | ---: | --- |
| MNIST | 0.462 ms | 0.348 ms | Exactly equal on this fixture |
| CIFAR-10 | 15.822 ms | 5.644 ms | Exactly equal on this fixture |

Runtime: local CPU, TensorFlow 2.18.0, NumPy 1.26.4. The campaign uses TensorFlow
2.21.0 on GPUs. This does not establish GPU equivalence, equivalence over all
states, multi-step trajectory equivalence, or a whole-run speedup. Five such
CIFAR steps save about 51 ms in this particular CPU measurement, which alone
cannot explain or eliminate a nine-hour ETA. A deployable path must preserve
diagnostics, finite checks, the nonzero-coefficient branch, and full trajectories.

### 3. Many small inference operations serialize model transfer and execution

`cart_batched_scoring.BatchedAdmittedWorker._candidate_logits` uses
`tf.map_fn(..., parallel_iterations=1)`. It removes Python calls between ranking
candidates, but does not evaluate their different models in parallel.
`sharp_compiled_transfer.CompiledTransferWorker._forward` assigns a shared
mutable model before inference. Merely raising `parallel_iterations` is not
a valid optimization; independent candidate execution needs stateless model
evaluation or separate model state and exactness tests.

FedSharp+ then separately scores its pool, up to eight peers, the owned model,
calibration perturbations, repair proposals and update-check trials. Its
`outputs` calls return NumPy arrays, introducing device-to-host completion
points. With 320 probe examples and the current batch limit 128, each such
call executes three inference batches and three `.numpy()` transfers.

Archived evidence establishes that calibration is frequent, not just a rare
branch. In each 100-round attack archive, across rounds 21–100 and three
seeds, CIFAR-10 FedSharp+ records 455,278 admission-calibration draws over
24,000 activations: 18.970 draws/activation. MNIST records exactly 19. These
counts exclude additional classifier-reference calibration. They are archived
100-round observations, not measurements of the current 500-round jobs.

Those 19 keyed draws are part of the executed method. Removing draws or
changing their reuse would change the experiment. Optimizations should retain
their values and decisions while reducing transfers and dispatch overhead.

Sources: `cart_prior_neutral_routing.py`, `sharp_cart_v2.py`,
`sharp_compiled_transfer.py`, and the two archived `mechanisms.json` collections.

### 4. Checkpoint bookkeeping repeatedly processes the complete history

`sharp_protocol.save_state` serializes all accumulated activation telemetry and
round metrics into JSON, packages complete model arrays, calls fsync, and
replaces the checkpoint. `cart_comparison_worker.save` then calls
`cart_mechanism_audit.telemetry_summary` over the entire history again.
`sharp_cart_v2_protocol.finish_round` does this after every round.

With N=100 clients and R rounds, the regular checkpoints process
100*R*(R+1)/2 activation records. That is 505,000 at 100 rounds versus
12,525,000 at 500 rounds: about 24.8 times the history-processing work for
five times the training horizon, excluding final/interruption saves. The
100-to-500 continuation alone processes 12,020,000 records through each full
history pass. Actual seconds and bytes still require measurement; the user's
write bursts do not by themselves prove disk saturation.

An incremental telemetry journal and incremental round summaries can remove
this repeated work. Such a change needs a tested checkpoint/export schema and
crash recovery, not deletion of reproducibility evidence or less validation.

### 5. Repeated deterministic work and duplicate evaluations are present

- `sharp_cart_v2_protocol.run` copies a receiver's whole local image/label
  partition with advanced indexing at every activation, although partitions
  are fixed. Caching costs additional RAM; that tradeoff must be budgeted.
- Fixed probes are regenerated in admission, commit-context construction and
  repair. Fixed local priors are recomputed at every local training call.
- `sharp_checked_pool.selector` reconstructs the best received copy after
  ranking; extra peer admission reconstructs up to eight more copies already
  generated by the base selector. Reuse must keep sender version, receiver,
  seed and sigma in the identity.
- `sharp_cart_v2.transport` recomputes final fit/check logits even when the
  final model equals the already-evaluated proposal. `cart_prior_neutral_commit`
  evaluates ingress at the start and evaluates it again if scale zero is
  reached. Cached evidence must preserve model-assignment side effects and
  validation checks, not merely return the same scalar.

These are concrete reuse opportunities. Their share of live runtime has not
been measured. Six inherited worker classes and several scoped monkeypatch
adapters also complicate reasoning about the active path. That is maintenance
debt; the existence of many files alone does not explain the runtime.

## Work that is deliberate, not an identified bug

The dataset is loaded and verified once per worker before the activation loop;
it is not repeatedly downloaded or loaded from disk each activation. The
protocol runs 100 logical clients sequentially. One configuration is not
distributed across all server GPUs. Every complete round also evaluates all
100 stored models on 1,000 validation examples: 100,000 example evaluations
per round, or 40 million for the added 400 rounds. This is required by the
current measurement contract, though its implementation can be optimized.

For the reported FedSharp+ job near round 146, roughly 35,400 activations
remain. A 9h33m estimate corresponds to about 1.03 activations/s. A long ETA
is consistent with that rate; it is not proof of incorrect learning. The
single-core CPU measurements and serialized work explain why a large GPU's
peak throughput is not a useful direct prediction of runtime here.

## Implementation order and current-run constraint

Prioritize correct lifecycle handling for GPU leases, then benchmark a
zero-coefficient path, exact probe/link reuse and duplicate inference removal.
Refactor checkpoint bookkeeping and stateless candidate evaluation separately,
with reference-versus-candidate trajectories and restart tests. Do not change
SGD counts, validation frequency, candidate policy, calibration draws or data
to manufacture a faster result.

No numeric optimization was deployed into the current campaign. Existing
Python workers retain their loaded code; a Git pull cannot hot-replace it.
Changing frozen numerical files also prevents future dispatch from passing
source checks. Under the requirement to leave workers uninterrupted, develop
and qualify changes in isolation; a later deployment needs an explicit,
provenance-preserving transition. There is no verified new HPC speedup yet.
