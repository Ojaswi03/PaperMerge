# CART: full-node helper threat audit

## Decision: do not start A/B/C/D yet

The requested scope is **full-node Byzantine protection before any new A/B/C/D
learning run**. Trusted, clean helper services were explicitly rejected as the
experimental assumption. The existing execution gate therefore stays closed.

Work was completed on **both helper-gradient responses and selection-score
responses**, including adversarial numerical contracts and an isolated robust
aggregation candidate. Passing these contracts does **not** validate full-node
learning. The candidate is not wired into the clean worker or GUI queue.

| Path | Added diagnostic protection | Still not established |
|---|---|---|
| Helper gradients | Exact context/layout/owner coverage checks; coordinate-wise four-trimmed mean; reject missing/non-finite payloads | Honest gradients, useful multiclass descent, authentication, withholding tolerance |
| Selection scores | Exact candidate/request/round binding; four-trimmed per-candidate score; deterministic nearest-predecessor ties | Truth of an owner's score; ranking by honest mean loss; authentication |
| Model snapshots | Existing Hidden-then-independent-absolute-noise contracts re-tested | Protection of helper channels or arbitrary malicious local optimization |
| EBM | Existing nested-tape/local-CE/no-lambda contracts re-tested | A defense against malicious helpers or an EBM derivative for the new trimmed estimator |

**Zero research runs, zero additional optimizer updates in the threat-audit
script, no W&B, no production queue changes.** Synthetic TensorFlow unit tests
exercise objective updates separately; they are not research trajectories.

## 1. What the new estimator actually guarantees

For `n` distinct responses with at most `b=4` malicious senders, independently
sort each coordinate, remove the smallest four and largest four, and average
the remaining values:

```text
T_b(x_1,...,x_n)[q]
    = mean(sorted(x_1[q],...,x_n[q])[b : n-b])

required: n > 2b
with n=10, b=4: only 2 coordinate values remain
```

Assuming the sender identities really are distinct/authenticated, every retained
coordinate is within the honest coordinate minimum and maximum. At most four
arbitrary values can fall below that interval, and at most four above it; trimming
removes them. This **coordinate-range statement** requires no IID assumption.
It is much weaker than guaranteeing closeness to a good multiclass gradient.

The diagnostic does not use known attacker IDs to remove responses. It does not
clip gradients, alter SGD, or apply this estimator to model weights. Its output
is **not** the gradient of the previously validated equal-class mean loss. We
must not quietly keep writing the old objective beside a different update rule.

## 2. Executed adversarial counterexamples

| Check | Observed result | Interpretation |
|---|---|---|
| 100 trials, 128 coordinates, six honest + four arbitrary large responses | All outputs within honest coordinate ranges | Bounded-outlier contract passed |
| One legitimate class direction `1`, nine responses `0` | Mean `0.1`; trimmed result `0.0` | An honest minority/class signal can disappear |
| Forged per-candidate score vectors | Honest preferred candidate changes after robust aggregation | Range bounds do not preserve honest ranking |
| Missing/duplicate/stale/wrong-shape/wrong-request responses | Rejected by contracts | Fail closed; no fabricated replacement responses |
| NaN/Inf payloads | Rejected | No conversion to artificial accuracy or silent repair |

The explicit score example is:

| Owners | Candidate P score | Candidate Q score |
|---|---:|---:|
| Three honest owners | 0 | 1 |
| Three other honest owners | 10 | 11 |
| Four malicious owners | 100 | 0 |
| Honest mean | **5** | 6 |
| Trimmed aggregate after four false reports | 10 | **1** |

The preferred candidate reverses even though each trimmed score stays inside
its honest score range. These are mathematical fixtures, **not measured CNN
accuracy results**. They demonstrate why the candidate is not production-ready.

## 3. Why our particular setting needs a protocol decision

| Constraint | Consequence |
|---|---|
| Exactly one owner per CIFAR class | A malicious owner has no independent fresh counterpart for its class evidence |
| Four of ten owners may be Byzantine | Robust trimming retains only two coordinate values, potentially discarding diverse honest signals |
| Only visited owners available in round 0 | An always-on nine-response quorum cannot start at the first activation |
| Approved attacks are inactive during R0–19 | Honest warmup can avoid that startup issue, but a later mean-to-robust switch is a new, unvalidated training rule |
| Arbitrary Byzantine nodes may withhold replies after R20 | Six responses remain, below this estimator's required nine; timeout/recovery policy is not deployed |
| Warmup models contain earlier knowledge | Some attacked-class knowledge may persist; do **not** claim all learning is impossible merely because owners later misbehave |
| A Byzantine active trainer can ignore the algorithm | Existing Hidden outbound corruption does not cover arbitrary malicious SGD execution |

Hashes identify bytes and context; they do not show that a claimed gradient or
loss was honestly computed. Adding TLS/signatures would prevent impersonation
and transport modification, but still would not make an authenticated malicious
owner's own response truthful. No RPC authentication is deployed by this patch.

This audit is **not an impossibility theorem for every defense**. It establishes
that the implemented candidate and existing information/trust assumptions do not
meet the requested assurance. Further source-based methods can be evaluated, but
their convergence or all-class guarantees must not be assumed from a name.

## 4. Channel noise and EBM remain separate

The approved snapshot channel remains:

```text
received_model = attacked_outbound_model + Normal(0, 0.010² I)
```

The existing owner-assisted EBM objective remains:

```text
mean_c CE_c(theta)
  + 0.0001 * ||gradient CE_receiver(theta)||²
  + (0.01/2) * ||theta - selected_snapshot||²    [fixed-anchor candidate]
```

Its EBM correction is `0.0002 * Hessian(CE_receiver) * gradient CE_receiver`.
There is no lambda. The synthetic contracts confirm coefficient, second-order
derivative, attack-before-noise ordering and keyed per-link independence.

This mitigates neither arbitrary dishonest helper claims nor necessarily noisy
gradient messages. Weight noise, gradient noise and scalar score noise have
different units. **No helper noise variance was invented** and no snapshot sigma
was silently reused for another channel. An explicit helper transport/noise model
and its loss adaptation remain separate design work.

## 5. Sources checked and credit

Coordinate-wise trimming is established prior work, not a new CART invention:
[Yin, Chen, Kannan and Bartlett, ICML 2018](https://proceedings.mlr.press/v80/yin18a.html),
Definition 2. Their statistical learning analysis starts from worker samples from
a common distribution; it is not a direct guarantee for ten unique class owners.
We use the elementary coordinate-range property and explicitly do not import
their statistical rates into our experiment.

The already-downloaded
[Liu et al., ICML 2023, gradient splitting (GAS)](https://arxiv.org/abs/2302.06079)
specifically studies heterogeneous gradients. Its aggregation selects `n-f`
gradients (Eq. 11) and its analysis includes local stochastic-gradient and
heterogeneity assumptions. It is relevant subsequent work, not evidence that our
new helper-score protocol is defended. With a single class per owner, discarded
owners also represent discarded current class objectives. GAS has not been
implemented or validated here; no attribution or novelty claim is made for it.

The Firecrawl source check influenced the decision to **keep the estimator
isolated**, rather than advertise a paper's guarantees outside its assumptions.
No retained PDFs were modified.

## 6. Files and reproducible checks

| File | Responsibility |
|---|---|
| `basil_core/cart_helper_robustness.py` | Lazy diagnostic estimators, immutable score records, exact context checks, explicit blocked readiness |
| `tests/test_cart_helper_robustness.py` | Poisoning, ranking-reversal, rare-class loss, quorum/context and lazy-import contracts |
| `scripts/check_cart_helper_threats.py` | No-training adversarial evidence generator; default lazy blocked plan |
| `gui/worker_state/CART_FULL_NODE_HELPER_THREAT_AUDIT.json` | Actual synthetic results, code hashes, zero research runs, remaining blockers |

```bash
environment/basil-noise-env/bin/python scripts/check_cart_helper_threats.py --check --output gui/worker_state/CART_FULL_NODE_HELPER_THREAT_AUDIT.json
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests run_gui.py
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_helper_robustness.py tests/test_cart_owner_integration.py
```

The evidence command refuses to overwrite its output. Omit `--output` to inspect
the diagnostic again. Focused verification: **50 passed, 2 warnings, 5.61 s**.

Full-suite command:

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-helper-threat-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py --report gui/worker_state/cart_helper_threat_lazy_imports.json
```

| Verification | Result |
|---|---|
| Complete suite | **599 passed, 2 skipped, 644 subtests passed**, 3 warnings, 76.20 s |
| Skips | Display-dependent checks; TensorFlow mathematical tests ran on CPU |
| Compileall | Passed |
| Lazy imports | 236 source files, zero changes needed, zero retained heavy imports |
| Synthetic threat audit | 100 × 128-coordinate trials passed; both limitation counterexamples reproduced |
| GPU | No new GPU learning/device-equivalence claim in this patch; estimator diagnostics are NumPy-only |
| Clean learning behavior | Previously validated worker/formula/source files unchanged; no estimator injected into training |
| Retained evidence | **24,599 before / 24,599 after / 24,599 hash-matched / zero changed** across existing IID/non-IID results, plots and papers |

## 7. Decision required before expanding implementation

The narrow trusted-helper option was rejected and has **not** been enabled.
For full-node assurance, the next architectural decision is how honest class
evidence can be independently established:

1. **Independent class evidence/redundancy:** changes the one-owner-only data
   constraint or introduces an explicitly approved verification dataset. This
   must be TRAIN-derived, never the CIFAR test set.
2. **Verifiable computation:** preserves private owner-local data in principle,
   but requires committed data/probe provenance and a real proof/attestation
   mechanism; a local recomputation oracle must not masquerade as decentralized
   Byzantine protection. Availability/withholding still needs a policy.

Neither option is silently selected. Authentication alone, finite checks, EBM,
or more training rounds do not resolve these questions. **The A/B/C/D execution
gate stays closed until the required protocol change is approved and tested.**
