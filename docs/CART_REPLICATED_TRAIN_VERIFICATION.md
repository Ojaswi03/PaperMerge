# CART: independently verified TRAIN helper computations

Newer CPU process-isolation and restart work is documented in
[CART isolated runner validation](CART_ISOLATED_RUNNER_VALIDATION.md).
The measurements below describe the earlier in-process certificate stage.

## Decision and current status

Use **authenticated independent recomputation**, not trimming of unverified
owner claims. This is the separate experimental revision
`cart_replicated_train_verification_v1`.

**Helper verification implemented; full-node A/B/C/D research remains blocked.**
No training experiment, GUI queue or W&B run was started by this work.
The existing clean results and original one-class protocol are unchanged.

The important scientific change is explicit: verification replicas can access
committed raw TRAIN examples from multiple classes. Local SGD can retain
5,000 single-class examples and five complete epochs, but the overall system
is **no longer an exclusive one-class-per-node information model**. Replication
is not a privacy guarantee. Results from this revision must be labelled separately.

## Why this route

An unverified gradient from the only class owner cannot establish the correct
gradient for that class. Coordinate trimming can erase a legitimate rare-class
direction; trimming claimed losses can reverse candidate rankings. See the
[counterexamples](CART_FULL_NODE_THREAT_READINESS.md).

Here, independent replicas compute the requested result themselves. They do
not merely sign the class owner's claim.

```text
Committed TRAIN data + exact model + batch indices + augmentation key
                              |
                   nine non-owner replicas
                 compute independently and sign
                              |
                 verify pinned identities/signatures
                              |
                  >=5 identical signed results?
                       /                  \
                     yes                  no
                      |                    |
            certified gradient/CE      fail closed
                      |
          balanced TRAIN selection / CART objective
```

### Certificate assumptions

| Item | Contract |
|---|---|
| Population | Ten authenticated logical identities; at most four faulty |
| Committee | Nine identities other than the class owner |
| Acceptance | Five distinct valid signatures on exactly the same result |
| Request binding | Run, round, receiver, owner, optimizer step, model hash, TRAIN root, indices, augmentation key, operation |
| Payload | Finite CE and immutable float32 gradient bytes; no tolerance-based merging |
| Membership | Trusted pinned ten-key roster; private keys generated independently of experiment seed |
| Honest execution | Same approved deterministic model/computation and committed data |
| Failure | Missing quorum aborts; no lowering the threshold, zero filling or fake response |

With at most four faulty identities, five matching signatures include at least
one honest computation. Since honest computations of this **fixed request**
are identical, the certified value equals that computation. At least five
honest replicas remain in the nine-member committee even if four withhold.
This is not a general Byzantine-consensus protocol, does not provide distributed
ordering and does not claim arbitrary BFT consensus with ten nodes/four faults.
Five faulty identities exceed the guarantee; a test explicitly demonstrates it.

Signatures use the maintained `cryptography` implementation of
[Ed25519](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/),
not custom cryptography. Install the optional dependency with:

```bash
environment/basil-noise-env/bin/python -m pip install -r environment/requirements-replicated-verification.txt
```

## What changes mathematically—and what does not

For available classes C, the established owner-gradient CART objective is:

\[
L_i(\theta)=\underbrace{\frac{1}{|C|}\sum_{c\in C}CE_c(\theta)}_{\text{class-balanced TRAIN supervision}}
+\underbrace{\sigma_e^2\|\nabla CE_i(\theta)\|_2^2}_{\text{local-CE EBM, when enabled}}
+\underbrace{\frac{\mu}{2}\|\theta-\theta_{\rm selected}\|_2^2}_{\text{frozen snapshot anchor}}.
\]

The applied gradient is:

\[
\frac{1}{|C|}\sum_c\nabla CE_c
+2\sigma_e^2 H_i\nabla CE_i
+\mu(\theta-\theta_{\rm selected}).
\]

Certification changes how helper CE/gradients are validated, not this formula.
EBM remains the receiver-local CE gradient-norm objective, **not** the norm of
the balanced gradient, and there is no legacy lambda. The detached anchor is
unchanged. A connected-autodiff test compares the complete loss, gradients and
prospective SGD update against independently certified helper responses under
both poisoned and withheld replies. It never starts a learning trajectory.

Balanced selection evaluates each of the five candidates using the same
per-owner unaugmented TRAIN probes, then selects one using minimum mean class
CE (ties within 1e-8 select the nearest predecessor). This is the **CART
balanced-TRAIN selection adaptation**, not a relabelling of paper BASIL's
receiver-local criterion. All classes are available after the first traversal;
round-zero requests exclude future owners. The full selected snapshot is used;
there is no model averaging or global TEST feedback.

## Implemented files

| File | Responsibility |
|---|---|
| `basil_core/replicated_verification.py` | Immutable requests/payloads, Ed25519 votes, exact five-signature certificates |
| `basil_core/replicated_training_service.py` | Immutable TRAIN backing, independent model evaluations, certified owner-response adapter and balanced selection |
| `scripts/check_replicated_training.py` | No-SGD synthetic check on the real 117,706-parameter CNN |
| `tests/test_replicated_verification.py` | Adversarial identity/context/payload checks, full-objective equivalence and lazy-import contracts |
| `environment/requirements-replicated-verification.txt` | Optional pinned signature dependency |

Heavy libraries remain lazy. The service does not receive attacker IDs; only
test fault injectors know which identities to corrupt. No existing research
engine, threat formula or queued configuration was changed.

## Measured checks

CPU synthetic CNN checks used 16 examples of one TRAIN-labelled class per
request. Every accepted result exactly matched the independently computed
honest reference:

| Operation | Fault injection | CE | Gradient L2 | Nine-endpoint wall time |
|---|---|---:|---:|---:|
| Gradient | None | 2.3251945972 | 1.3016755791 | 0.1427 s |
| Gradient | Four poisoned replies | 2.3251945972 | 1.3016755791 | 0.1411 s |
| Gradient | Four withheld replies | 2.3251945972 | 1.3016755791 | 0.0759 s |
| Selection CE | None | 2.3236734867 | N/A | 0.0609 s |
| Selection CE | Four false scores | 2.3236734867 | N/A | 0.0637 s |
| Selection CE | Four withheld replies | 2.3236734867 | N/A | 0.0342 s |

These are small implementation measurements, **not accuracy results or a
campaign runtime prediction**. Nine recomputations cost substantially more than
one helper evaluation. Ten-class/five-candidate selection calls 450 endpoints;
the test verifies all 450 independent computations.

CPU evidence: `gui/worker_state/CART_REPLICATED_TRAIN_CPU_VERIFICATION_FINAL.json`.
GPU attempt stopped before evaluation because only **307 MiB** was free versus
the policy requirement of **7,168 MiB** (6,144 MiB cap plus 1,024 MiB headroom).
No memory cap was relaxed and no other process was stopped. This revision's
GPU behavior is **unverified**; CPU checks do not imply GPU equivalence.

## Remaining full-node gate

| Boundary | Current status |
|---|---|
| Poisoned/forged/stale helper messages | Tested against the certificate consumer |
| Four missing replies | Simulated explicit timeouts tolerated; no real RPC timeout implementation |
| TRAIN evidence | Immutable in-process backing with declared provenance; trusted bootstrap still required |
| Membership and keys | Logical simulation only; arbitrary hostile Python could read process memory |
| Active malicious SGD execution | Not certified; helper certificates alone do not prove honest training |
| Network transport | No physical RPC, process isolation, request ordering or deployment validation |
| Resume | No persisted key/certificate lifecycle or interrupted distributed-step restoration yet |
| Helper channel corruption | Authentication rejects tampering; do not reuse model-weight sigma as gradient noise |
| Learning/robustness benefit | Not measured for this replicated-data revision |
| A/B/C/D integration | Deliberately not enabled while the full-node boundary remains incomplete |

The next engineering gate is end-to-end authenticated execution and recovery
with isolated identities, including malicious active-node behavior, plus a
bounded performance check. A full experiment must disclose data replication and
the precise remaining trust assumptions. Do not substitute trusted-clean helpers
or claim full-node Byzantine protection from these unit tests alone.

## Reproduce without training

```bash
environment/basil-noise-env/bin/python scripts/check_replicated_training.py --check --device CPU
# Only when the GPU safety guard has sufficient headroom:
environment/basil-noise-env/bin/python scripts/check_replicated_training.py --check --device GPU

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_replicated_verification.py
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
```

The first pytest attempt failed before collection because the ambient ROS
plugin required unavailable `lark`. Disabling automatic unrelated plugin
loading avoids that environment issue; it does not exclude repository tests.

### Final verification results

| Check | Actual result |
|---|---|
| Focused certificate/objective tests | 20 passed, 2 dependency deprecation warnings, 4.10 s |
| Entire `tests/` suite | 619 passed, 2 skipped, 644 subtests passed, 3 warnings, 78.50 s |
| Skips | Two optional Tk/display integration tests; `DISPLAY` unset |
| Compileall | Passed for `gui basil_core reporting scripts tests` |
| Lazy-import audit | 240 source files; 0 changes needed; 0 retained heavy imports |
| CPU CNN certificates | Six cases passed; zero optimizer updates |
| GPU certificate check | Blocked by insufficient free VRAM before evaluation |
| CPU evidence source hashes | All four recorded implementation/test hashes match |
| Existing evidence hashes | 24,599 before/after; 24,599 matched; zero changed |
| Research/queue execution | Not started |

Full-suite command actually executed:

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-replicated-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q -rs tests
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py --report gui/worker_state/cart_replicated_lazy_imports.json
```

The hash baseline is
`gui/worker_state/cart_replicated_verification_before.json`, covering saved
IID/non-IID results, plots and `papers/`. Nothing in those trees was rewritten.
