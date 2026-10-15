# Dirichlet rerun readiness

**Status: ready for a manual 30-round GPU confirmation. The full queue has not started.**

## Completed learning evidence

Values are measured from saved CSV/NPZ/activation records. Node 9 is the trained
end-of-ring model evaluated on all 10,000 TEST images before outbound corruption.
Node mean/worst are the mean/minimum across all ten stored trained node states.

| Completed check | Rounds | Node 9 | Node mean | Worst node | GPU runtime |
|---|---:|---:|---:|---:|---:|
| Original raw CE + anchor, clean | 10 | 19.13% | 13.99% | 10.67% | 6.95 min |
| Local-prior CE + anchor, clean | 10 | **32.90%** | **30.34%** | **28.14%** | 7.12 min |
| Local-prior CE + anchor + noise, no attack | 3 | 24.00% | 20.95% | 14.77% | 2.34 min |
| Local-prior CE + anchor + noise + EBM, no attack | 3 | 23.95% | 20.90% | 14.62% | 2.32 min |
| B: local-prior CE + anchor + Hidden | 23 | **41.29%** | **39.54%** | **35.33%** | 17.10 min |
| C: local-prior CE + anchor + Hidden + noise | 23 | **45.04%** | **43.29%** | **34.98%** | 17.47 min |
| D: local-prior CE + anchor + Hidden + noise + EBM | 23 | **46.11%** | **44.45%** | **36.91%** | 17.73 min |

The matched clean ten-round gain is **+13.77 percentage points** for Node 9.
Mean absent-class forgetting per activation fell from 9.62 pp to 0.96 pp; this
metric alone cannot prove learning when incoming and outgoing models are poor.

At 23 rounds, D minus C is +1.07 pp for Node 9, +1.16 pp for node mean and
+1.93 pp for worst node. The no-attack three-round EBM comparison was −0.05 pp.
These single-seed observations do not establish a general EBM benefit or
statistical significance. C exceeding B is not proof that noise is beneficial.
**Do not subtract the ten-round clean result from a 23-round attacked result.**

| CIFAR-10 class | B, R22 | C, R22 | D, R22 |
|---|---:|---:|---:|
| Airplane | 40.6% | 49.7% | 48.5% |
| Automobile | 48.9% | 52.9% | 49.1% |
| Bird | 34.1% | 47.1% | 54.6% |
| Cat | 12.5% | 26.3% | 37.7% |
| Deer | 26.2% | 34.7% | 27.0% |
| Dog | 43.6% | 33.0% | 35.2% |
| Frog | 63.0% | 59.2% | 59.2% |
| Horse | 43.9% | 53.0% | 54.8% |
| Ship | 55.7% | 53.9% | 53.3% |
| Truck | 44.4% | 40.6% | 41.7% |

No final node in B/C/D was an exact constant-class predictor. Performance is
still uneven: the clean prior-adjusted R9 model had only 5.4% cat recall.
Retention improved; it is not solved.

This work separates a real local-accuracy reporting defect from two scientific
problems: class-prior specialization and acceptance of attack-corrupted models.
The existing results are preserved; these are new, explicitly bounded diagnostic
runs, not replacements for the previous A/B/C/D study.

## Fixed checks and scope

| Stage | Conditions | Full ring rounds | Purpose |
|---|---|---:|---|
| Clean | Original CE + anchor; local-prior adjusted CE + anchor | 10 | Check learning and retention beyond the initial three rounds |
| Noise | Adjusted CE + anchor; adjusted CE + EBM + anchor | 3 | Verify actual absolute noise and second-order EBM, with attacks disabled |
| Attack admission gate | B Hidden; C Hidden + noise; D Hidden + noise + EBM | 23 | Exercise the actual R20 activation and three attacked rounds; no shortened warm-up |
| Full feasibility rerun | A/B/C/D | 30 | Approved idle preset; not started |

All retain the full CIFAR-10 TRAIN/TEST sets, the original Dirichlet α=0.2
partition, five complete local epochs, batch 512 and partial batches, the same
117,706-parameter CNN, reset SGD and approved round decay. No data sharing,
helper services, CART, averaging, clipping, momentum, weight decay or W&B is
enabled. Test measurements do not control training, selection or stopping.

These stages have a distinct revision:
`dirichlet_prior_readiness_diagnostic_v1`. Their configurations cannot enter the
original production queue; `researchValid=false` and `fullResearchReady=false`.
There is no automatic extension or hyperparameter search.

The attack gate was declared after the clean/noise checks, **before any new
attacked run began**. Its 23-round limit reduces diagnostic cost without changing
the R20 attack or five local epochs. Three post-onset rounds check immediate
admission and failures, not long-term resilience. Clean A's ten-round results
must not be subtracted from a 23-round B as an attack-effect estimate.

## Objective and selection are deliberately distinct

Define receiver-local TRAIN counts (n_{i,c}), fixed floor-one prior

$$
\pi_{i,c}=\frac{\max(n_{i,c},1)}{\sum_{k=0}^{9}\max(n_{i,k},1)}.
$$

The explicitly named training adaptation is

$$
F_i(\theta)=\operatorname{mean}CE(y,z_\theta(x)+\log\pi_i).
$$

With the frozen selected received snapshot (\theta_{sel,i}), the tested loss is

$$
L_i=F_i+c\|\nabla F_i\|_2^2+
\frac{0.01}{2}\|\theta-\theta_{sel,i}\|_2^2.
$$

| Component | Exact meaning |
|---|---|
| $F_i$ | Training-side local-prior logit adjustment; not post-hoc inference correction |
| $c=0$ | Non-EBM conditions, including noisy CE control C |
| $c=0.0001$ | EBM condition D with fixed coordinate sigma 0.010 |
| Anchor | Same immutable received snapshot throughout all five epochs |
| Applied gradient | $\nabla F_i+2cH_{F_i}\nabla F_i+0.01(\theta-\theta_{sel,i})$ |

No lambda is read. EBM regularizes **adjusted** cross-entropy in this new method;
it is not silently substituted for the original raw-CE EBM experiment.

Snapshot Selection and inference still use **raw logits**, with the same
unaugmented seeded receiver-local TRAIN batch for all five candidates. Selection
is minimum ordinary CE, with the original deterministic tie rule. No guard has
been introduced or relabelled as paper BASIL.

The training logit adjustment follows [Menon et al., ICLR 2021, §5.1/Eq.10](https://openreview.net/forum?id=37nvvqkCo5);
receiver-local priors, floor-one missing counts, ring handoff and the snapshot
anchor are our declared adaptations. It is not a guarantee of class retention
or Byzantine robustness. See [source and initial evidence](DIRICHLET_CALIBRATION_SMALL_TESTS.md).

## Reproducibility and memory safety

- CPU/GPU mathematical contracts and every actual partial-batch shape are tested.
- Execution is serial deterministic float32; TF32 is disabled. CPU/GPU results
  are tolerance-equivalent, **not bit-identical**.
- GPU allocator cap remains 6 GiB, with startup headroom and exclusive lease.
- Checkpoints contain distinct node states, rolling memories and completed
  telemetry. An interrupted partial activation is discarded and replayed.
- Every new diagnostic archives the executed source bytes by SHA-256. Resume
  rejects changed configuration, source, dataset or numerical execution policy.
- Existing evidence and PDFs were hashed before work: **26,191 files matched**.

## Commands (default is dry-run)

```bash
environment/basil-noise-env/bin/python scripts/run_dirichlet_calibration.py --stage clean
environment/basil-noise-env/bin/python scripts/run_dirichlet_calibration.py --verify-runtime --stage clean
```

An explicit `--execute` is required for any bounded learning run. A stage's
completion is not permission to launch the full GUI queue or a 500-round study.
Output namespaces are `newResults/nonIID/dirichlet_prior_readiness_<stage>_<rounds>r_v1/`
and corresponding `newPlots/nonIID/` directories; old runs are never overwritten.

## Readiness criteria

Passing code tests alone is insufficient. Before a full rerun, verify complete
epochs and memories, finite objectives/updates/evaluations, correct attack/noise
order, paired identities, checkpoint recovery, actual class learning and
post-attack behavior. Accuracy is descriptive, not a >50% software assertion.

If corrupt snapshots still enter or a condition collapses to a constant class,
report that failure. Do not weaken the attack, retune coefficients from test
accuracy or call the experiment ready to demonstrate robust learning.

## Verified attack, noise and memory evidence

```text
five distinct latest predecessor snapshots
       ↓ same local TRAIN batch, raw CE; choose ONE snapshot
complete replacement → freeze received snapshot as anchor
       ↓ reset SGD, five full epochs
honest trained model → Hidden if active → independent noise per link
       ↓
refresh five successor memories; ONLY immediate successor trains next
```

Before Node 0 at R11: `5:R10, 6:R10, 7:R10, 8:R10, 9:R10`.
Before Node 1: `6:R10, 7:R10, 8:R10, 9:R10, 0:R11`.
The gates verify these rolling-memory, raw-local-minimum and handoff contracts.

The retained project Hidden attack is not claimed as an exact paper reproduction:

$$
w_{attack}=0.45w+0.55(-1.4w+\delta)=-0.32w+0.55\delta,\quad
\delta_t\sim N(0,[0.05\max(\operatorname{std}(w_t),0.01)]^2).
$$

C/D then transmit \(\widetilde w=w_{attack}+\xi\), with
\(\xi\sim N(0,0.0001I)\), independently per directed link. Standard deviation
is 0.010; no dynamic model-norm scaling is applied.

| Attacked gate | Decisions | Designated-attacker sender selections | Actually corrupted selections | Attacked activations |
|---|---:|---:|---:|---:|
| B | 230 | 73 | **0** | 12 |
| C | 230 | 77 | **0** | 12 |
| D | 230 | 72 | **0** | 12 |

Designated attackers are honest before R20. Their cached pre-R20 snapshots are
not corrupted even when selected later. Zero admission across R20–22 is a
bounded observation, not a guarantee about arbitrary Byzantine behavior.

All **350 final directed transmissions** (50 × seven conditions) were regenerated
read-only and matched saved hashes and rolling memories. C/D Gaussian noise
hashes match across **1,150** links; the three-round noise pair across **150**.
No SGD occurred in this verification.

For D's 23-round gate, mean EBM penalty was **0.00037847**; mean correction/base
gradient ratio **0.008807** (0.88%), maximum **0.02801** (2.80%). C/D mean noise
L2 norm was **3.43079**; mean noise/model ratios were **0.13533 / 0.13570**.
Those ratios are diagnostics only and never determine the absolute sigma.

The source distinction remains explicit: BASIL Eq.3 supplies receiver-local
selection; Ang et al.'s gradient-norm regularization is adapted to adjusted
multiclass CE. The paper's Eq.14 squared-loss approximation is not an exact
expected CNN-CE identity; Eq.23 scaling is not the general Hessian-gradient
derivative. The anchor has FedProx-like proximal structure adapted to the
selected received snapshot; it is not BASIL/Ang's proposed anchor mechanism.

## Full-data steps and resource measurements

Dirichlet α=0.2 is neither the equal-sized IID split nor one class per node.
The reused partition covers all 50,000 TRAIN examples disjointly.

| Node | TRAIN samples | Batches/epoch including partial | Five-epoch steps |
|---|---:|---:|---:|
| 0 | 2,938 | 6 | 30 |
| 1 | 4,407 | 9 | 45 |
| 2 | 4,768 | 10 | 50 |
| 3 | 989 | 2 | 10 |
| 4 | 9,660 | 19 | 95 |
| 5 | 6,693 | 14 | 70 |
| 6 | 2,945 | 6 | 30 |
| 7 | 5,622 | 11 | 55 |
| 8 | 4,115 | 9 | 45 |
| 9 | 7,863 | 16 | 80 |
| **Per traversal** | **50,000** | **102** | **510** |

Each clean ten-round condition executed 5,100 steps; each noise three-round
condition 1,530; each attacked 23-round condition **11,730**. A 30-round run
will perform **15,300**. All partial batches are retained.

Partition SHA-256: `2c792027992625ff914827842fc41e27a88411c0ae0fcc4ef900160ec5463f1e`.
Canonical initialization SHA-256: `7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df`.

All seven learning checks ran serially on RTX 4070 Ti GPU. The maximum measured
allocator peak was **415.40 MiB**. No non-finite failure or OOM was recorded;
future OOM cannot be ruled out under external resource pressure. CPU/GPU
synthetic prior-CE/EBM gradient relative errors were approximately
7.76×10⁻⁷ / 7.91×10⁻⁷, not bit identity. Multiprocessing remains disabled/unverified.

## Run the idle confirmation preset

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

1. Open **Queue → Load Queue**.
2. Select `gui/queues/noniid_dirichlet_prior_abcd_30r_gpu.json`.
3. Confirm A/B/C/D order, **30 rounds**, **GPU**, and **0/30** done/total.
4. Confirm idle, then click **Run Queue** when ready.

| Condition | Attack | Channel | Training loss |
|---|---|---|---|
| A | None | Clean | Local-prior CE + anchor |
| B | Hidden R20 | Clean | Local-prior CE + anchor |
| C | Hidden R20 | Absolute σₑ=0.010 | Local-prior CE + anchor |
| D | Hidden R20 | Absolute σₑ=0.010 | Local-prior CE + EBM + anchor |

The existing user queue was not silently replaced. Loading the new preset does
not auto-start workers. All four start fresh under
`newResults/nonIID/dirichlet_prior_anchor_confirmation_30r_v1/{A,B,C,D}/`, with
automatic completed-run plots under matching `newPlots/nonIID/` paths. Live
dashboard/network JSON events and measured CSV rows update during execution.
All remain `researchValid=false`, `fullResearchReady=false`: bounded confirmation,
not a validated long-horizon production result.

Estimate from measured gates: about **90 minutes compute** plus startup/plots;
allow **1.5–2 hours** under similar load, not a guaranteed ETA.

Use **Stop**, wait for exit, then resume without editing sources/configs. Exact
activation-boundary checkpoints restore distinct node models, five-sender
memories, sender/round identity, measured arrays and immutable telemetry. An
interrupted partial activation is replayed. Missing/corrupt checkpoints,
changed source/data/numerical policy and genuine failed runs block recovery
instead of silently restarting from weights. Repeated stops get distinct audit
directories; a validated previous checkpoint can replace a corrupt current one.

After all four finish, generate the comparison without training:

```bash
python scripts/report_dirichlet_prior_study.py
```

It refuses incomplete runs and existing comparison outputs. B−A measures the
attack environment with BASIL active in both; C−B noise; D−C EBM; D−A total gap.
No expected ordering is coded.

## Engineering verification and files

| Files | Change |
|---|---|
| `basil_core/research_protocol.py` | Full-local TRAIN accuracy instead of the class-grouped first-512 prefix; no SGD or TEST change |
| `basil_core/dirichlet_calibration.py` | Explicit TRAIN-prior objective and true EBM; original raw CE stays separate |
| `basil_core/dirichlet_diagnosis.py`, `reporting/dirichlet_diagnosis.py` | Read-only collapse, memory, epoch and corruption analysis |
| `basil_core/dirichlet_prior_study.py` | Evidence-gated configurations and exact checkpoint recovery |
| `scripts/run_dirichlet_calibration.py`, `scripts/run_dirichlet_prior_condition.py` | Single-condition GUI execution, source archives, live events and atomic round CSV |
| GUI app/state/queue/execution modules | Clear experiment labels, approved preset loading, done/total and serial GPU routing |
| `reporting/dirichlet_calibration.py`, `scripts/report_dirichlet_prior_study.py` | Completed-data reviews and comparison plots without retraining |

All **100** first-ten-round input/trained/attacked hashes, selected sources/rounds
and TEST values matched across the clean/audit wiring change. This is a
same-device check, not CPU/GPU bit identity.

```bash
python -m compileall -q gui basil_core noise_comm reporting scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 \
  MPLCONFIGDIR=/tmp/papermerge-test-mpl xvfb-run -a python -m pytest -q tests
python scripts/prepare_dirichlet_prior_queue.py  # read-only inspection
```

The real runner/checkpoint I/O test uses a synthetic engine: stop after one
committed activation, resume immutable state, check 300 live activation events
and 30 contiguous CSV rows, **zero SGD**. Separate mathematical tests use real
nested tapes, no-lambda comparisons, frozen references and full epochs. Actual
GUI startup and preset loading passed across five workspaces with no worker.

Checked data: `newResults/nonIID/dirichlet_prior_readiness_review_v1/analysis.json`.
Twelve comparison PNGs: matching `newPlots/nonIID/dirichlet_prior_readiness_review_v1/`.
An additional **135 detailed PNGs** are in its `detailed/{B,C,D}/` subdirectories,
including per-epoch class-retention and confusion-matrix panels. The plotting
adapter reads older manifests' config-level run IDs without modifying them.
Approval: `gui/worker_state/dirichlet_prior_readiness.json`.
GUI check: `gui/worker_state/DIRICHLET_PRIOR_GUI_VERIFICATION.json`.

### Final verification results

| Verification | Executed outcome |
|---|---|
| Compileall command above | Passed, exit 0 |
| Complete pytest command above, with `--junitxml=gui/worker_state/dirichlet_rerun_tests.xml` | **772 passed; 644 subtests passed; 0 failed; 0 skipped; 3 warnings**, 146.92 s |
| Focused calibration/GUI/recovery tests | 62 passed, 2 warnings, 17.97 s before the final extra guard test |
| Actual GUI startup and preset load | All five workspaces; A/B/C/D, 30 rounds, 0/30, no worker launched |
| GPU objective/source certificate | Passed for current unchanged objective SHA-256; prior CE and EBM, CPU/GPU tolerance check, partial shapes and no-lambda identity |
| D CLI dry-run | `configuration_valid`, `executionReady=true`, `trainingStarted=false`, fresh 30-round plan |
| Publication policy | Passed: no tracked handoff documents |
| `git diff --check` | Passed |
| Historical integrity | 26,191/26,191 SHA-256 matches, 0 changed |
| Completed-gate integrity | 1,227/1,227 SHA-256 matches, 0 changed |

The three final-suite warnings are two protobuf/Python deprecations and an
existing Keras input-structure warning in a legacy CART unit test. No tests were
excluded. One intermediate run mixed an in-flight restored-history round-index
correction with an already-collected older test expectation; the clean final
rerun above supersedes it. No training mathematics changed for that correction.
The final console log is `gui/worker_state/dirichlet_rerun_tests.log`.

The historical integrity manifest covers **26,191 result/plot/PDF files**. A
separate snapshot covers **1,227 completed gate files** before GUI I/O edits.
Executed diagnostic source bytes remain archived with results. Source changes
do not rewrite or falsely resume those completed diagnostics. Approval depends
on local completed evidence; a clone without it correctly stays blocked.

## Verdict

**Ready for the fixed 30-round GPU confirmation, not 500-round production.**
The revised matched learning checks improved and B/C/D survived the first three
attacked rounds without constant-class collapse or corrupt admission. Later
attacks, long-horizon retention, multiple seeds and universal robustness remain
unestablished. No coefficient sweep maximized final TEST accuracy, and 50%
accuracy is not a software correctness assertion.
