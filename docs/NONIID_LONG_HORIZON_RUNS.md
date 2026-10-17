# Non-IID A/B/C/D: fixed 100- and 500-round runs

## Completed status

All eight queued runs completed. See the [completed results analysis](../newResults/nonIID/500round/FINAL_ABCD_RESULTS_ANALYSIS.md)
for measured results, actual attack admissions, mathematical checks and limitations.
Final mean-node accuracies at 500 rounds were A/B/C/D = **60.17% / 59.73% /
35.17% / 35.83%**. Correct execution did not yield satisfactory long-horizon
noise mitigation. The instructions below document the queue workflow; they do
not authorize a new run or an automatic extension.

These are **Dirichlet α=0.2 CIFAR-10**, not one-class or IID runs. Preparation
does not launch training. The 30-round evidence remains unchanged.

## Combined queue (recommended)

The loaded preset is `gui/queues/noniid_dirichlet_prior_abcd_100r_then_500r_gpu.json`:

```text
A100 → B100 → C100 → D100 → automatic read-only audit → A500 → B500 → C500 → D500
```

Click **Run Queue** once. The four 500-round entries are visible as **Waiting**
until all four 100-round runs contain 100 completed/evaluated rounds and 1,000
activations each. Then the GUI performs the full protocol/numerical audit on a
background thread. Only a successful receipt permits the first 500-round worker
to launch. A failure or interruption pauses the queue; it never skips a failed
100-round condition or silently starts a 500-round run. Closing/stopping the GUI
during the audit cancels automatic advancement.

Queue totals include all **eight** runs: **2,400 rounds**, **24,000 activations**,
and **1,224,000 optimizer batches**. Estimated serial runtime is approximately
**34.2 hours** for both stages, with device/load/upload variability.

The individual stage presets remain available:

| Stage | Queue preset under `gui/queues/` | Order | Tracking | Execution gate |
|---|---|---|---|---|
| Validation | `noniid_dirichlet_prior_abcd_100r_gpu.json` | A → B → C → D | Local only | Completed 30-round protocol/numerical audit |
| Longer research run | `noniid_dirichlet_prior_abcd_500r_gpu.json` | A → B → C → D | W&B, `NONIID RUN` | All four 100-round runs completed and validated |

Each condition starts **fresh at canonical round 0**. A 500-round run is not
a weight-only continuation of a 100-round result. Output/checkpoint paths are
separate, and existing evidence is not overwritten.

```text
Completed 30-round A/B/C/D evidence
                 |
       protocol/numerical audit
                 |
       100r: A → B → C → D       local tracking; user starts combined queue
                 |
       automatic validation     no SGD, no accuracy target; failure pauses queue
                 |
       500r: A → B → C → D       W&B; starts only after all prerequisites pass
```

Passing means correct full-data coverage, five complete epochs, finite metrics,
correct BASIL memory/selection, actual attacks and actual channel noise, paired
initialization/data/noise, and valid checkpoint/transmission evidence. It does
**not** require a chosen accuracy, EBM superiority, or zero corrupted selections.
Scientific outcomes must be reported even when disappointing.

## The mathematics is unchanged

See [the complete loss and threat guide](NONIID_LOSS_AND_ABCD_GUIDE.md).
With fixed local TRAIN priors and raw logits $z_\theta(x)$:

$$
F_i(\theta)=\frac{1}{|B_i|}\sum_{(x,y)\in B_i}
\mathrm{CE}\bigl(y,z_\theta(x)+\log\pi_i\bigr).
$$

$$
\mathcal L_i=F_i+c\lVert\nabla F_i\rVert_2^2
+\frac{0.01}{2}\lVert\theta-\theta_{\mathrm{sel},i}\rVert_2^2.
$$

| Condition | Actual attackers | Channel | EBM coefficient $c$ | Anchor |
|---|---|---|---:|---:|
| A | None | Clean | 0 | μ=0.01 |
| B | [0,1,5,7], retained project Hidden from R20 | Clean | 0 | μ=0.01 |
| C | Same as B | Absolute Gaussian σₑ=0.010 from R0 | 0 | μ=0.01 |
| D | Same as B | Same paired Gaussian draws as C | 0.0001 | μ=0.01 |

Only the horizon and observation/reporting change. The local-prior loss worker
is checked against the archived 30-round implementation using syntax-tree
equivalence. SGD, LR decay, augmentation, attack, channel and selection source
files are checked against the completed reference. There is no extra lambda,
CART, helper service, data sharing, adaptive μ, clipping, momentum or weight decay.

## Exact work counters versus estimated time

The dashboard shows **individual experiment** and **loaded queue** progress:

| Work unit | One 100-round condition | Four-condition 100r queue | One 500-round condition | Four-condition 500r queue |
|---|---:|---:|---:|---:|
| Complete ring rounds | 100 | 400 | 500 | 2,000 |
| Node activations | 1,000 | 4,000 | 5,000 | 20,000 |
| Optimizer batches | 51,000 | 204,000 | 255,000 | 1,020,000 |

The full-data Dirichlet partition has **510 optimizer batches per ring**, not
500. Five-epoch per-node counts are `[30,45,50,10,95,70,30,55,45,80]` because
node datasets have unequal sizes and retain every partial batch.

Counters update at **completed node activations**. They deliberately do not
guess how many batches an in-progress activation has finished. Exact counts
are restored from the activation-boundary checkpoint after interruption.
Progress is weighted by these batch counts, not by file size or CNN parameters.
The parameter count stays constant and output sizes depend on compression, so
neither would be a meaningful training clock.

Experiment ETA excludes startup and restored work. Once each node has a measured
duration, it uses recent per-node median wall times, including evaluation and
checkpoint overhead; before that it uses the observed weighted work rate.
Queue ETA combines the running experiment's ETA
with measured 30-round per-condition estimates for pending work. Completed
entries remain in the saved longer-horizon queue so totals do not shrink.
Times are still estimates: evaluation, checkpoint writes, compilation, GPU load
and uploading can vary. Completed work is exact; seconds remaining are not.

Approximate planning estimates from the completed GPU runs:

| Condition | 100 rounds | 500 rounds |
|---|---:|---:|
| A | 1.34 h | 6.70 h |
| B | 1.35 h | 6.75 h |
| C | 1.38 h | 6.89 h |
| D | 1.63 h | 8.13 h |
| Total, serial | **5.69 h** | **28.47 h** |

These are extrapolations, not runtime promises. W&B overhead is not measured by
the local-only reference. GPU execution retains the 6,144 MiB limit, headroom
check, exclusive lease, streamed evaluation and cleanup. These reduce OOM risk;
they cannot guarantee immunity to external memory pressure.

## Launch and unlock

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

Open **Queue**, confirm all eight entries in the order above, then click
**Run Queue**. To reload it, use **Queue → Load Queue** and choose the combined
`100r_then_500r` preset. The preparation task does not click Run Queue.

The combined queue automatically performs this read-only validation after all
four 100-round runs finish. For standalone stage presets, it is also available
manually:

```bash
python scripts/prepare_dirichlet_horizon_queues.py --validate-100
```

Only a passing receipt unlocks the **500r** stage. New W&B runs are created only when
those conditions actually execute, under entity `ojaswisinha15-ohio-university`,
project `Research Paper`, group `dirichlet_prior_anchor_500r_v1`.

W&B uses the same `NONIID RUN/Accuracy`, `NONIID RUN/Loss` and `round/*` metric
keys across A/B/C/D, with `round` as the horizontal axis. Select those four runs
in the project workspace to overlay their live curves. No old IID run IDs are
reused. Raw images, model weights, checkpoints, source and terminal output are
not uploaded by this tracker. Accuracy/loss are logged every completed round;
noise, gradient, anchor and EBM aggregates accompany them.

W&B must initialize successfully before the first 500-round SGD update. If
connectivity later fails, local checkpoints/results continue and the GUI reports
tracking unavailable; cloud delivery is not silently claimed successful. A retry
uses the saved cloud run identity. No online W&B run is created merely by
loading a preset or running unit/configuration checks.

Initialization must also succeed before any SGD in a resumed attempt. A failure
there leaves the manifest `interrupted`, retains its committed work and cloud
identity, and requires verified checkpoint recovery on retry. Only a genuinely
zero-work startup without a current/previous checkpoint or committed scientific
artifacts can use `startup_retry`. Older nonzero `tracking_unavailable`
manifests are routed through the same verified checkpoint reader, including
previous-checkpoint fallback; missing/corrupt evidence never becomes a fresh run.

The GUI can display recorded completion from compact manifests and measured
CSV rows independently of current-source execution approval. This does not
establish exact-resume readiness. Software changes invalidate source-sensitive
approval receipts and checkpoint compatibility checks; do not rewrite executed
manifests, regenerate approvals silently, or bypass those checks. The completed
results remain readable with their original archived source identities.

## Results, recovery and interpretation

Results: `newResults/nonIID/dirichlet_prior_anchor_{100,500}r_v1/{A,B,C,D}/`.
Plots: the matching paths under `newPlots/nonIID/`.

Every completed activation checkpoints all logical node models, received
memories, sender/round metadata, metric arrays, telemetry references and the
cursor. An interrupted activation is replayed from its start; completed
activations are not repeated. Changed config/source/numerical policy blocks an
exact resume. A numerically failed run requires investigation rather than an
automatic restart. These longer profiles do not modify or extend saved 30-round
checkpoints.

`full_test_accuracy` remains the honestly trained Node-9 model before outbound
corruption, not an ensemble or corrupted outgoing accuracy. Mean/worst metrics
refer to the ten stored trained models. TEST is evaluation-only. Current Hidden
is the retained project transformation, **not an exact BASIL-paper Hidden
reproduction**. All comparisons remain single-seed; small C–D differences are
not statistical-significance evidence.
