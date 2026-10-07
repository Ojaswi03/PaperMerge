# PaperMerge repository and completed IID study — catch-up prompt

Copy the text below into your next research-review conversation and attach the
ZIP downloaded from the pushed repository. Attach any additional local evidence
only if needed; the committed ZIP intentionally excludes large recovery
checkpoints and raw batch traces. This document is context, not authorization
to run more experiments.

---

## Your role and immediate task

Act as a careful Python/TensorFlow research-software reviewer and decentralized-learning research collaborator. I am sharing the current PaperMerge repository ZIP and its latest reports. First catch up with what was implemented, what was actually executed, and what the evidence supports. Do not begin by proposing another architecture or rewriting the protocol. Do not treat planned features, old preflights, or archived CPU runs as current completed results.

Read the attached repository itself. Treat the values below as orientation, then verify them against the included manifests, metrics, reports, and source. If a referenced file is absent from the ZIP, say exactly what cannot be independently verified. Never invent missing metrics or say a raw-trace check was rerun when only its saved audit summary was available.

No additional training, experiment extension, third/fifth condition, queue execution, production matrix, non-IID experiment, or parameter tuning is authorized by this prompt. I want an evidence-grounded review and a clear explanation of the results first.

## 1. Read these files first

1. `README.md`, especially the latest-results links at the top.
2. `newResults/IID/ABCD_100r_comparison_gpu/FINAL_ABCD_100R_ANALYSIS.md`.
3. `docs/IID_CODE_CHANGES_AND_TRAINING_MATHEMATICS.md`.
4. `newResults/IID/ABCD_100r_comparison_gpu/POST_RUN_VERIFICATION.json`.
5. `docs/IID_BASIL_PROTOCOL.md`, `docs/SOURCE_EQUATIONS.md`, and `docs/IID_GPU_EXECUTION.md`.
6. Current A/B/C/D `config.json`, `run.json`, `round_metrics.csv`, `metrics.npz`, and `iid_partition_audit.json`.
7. Relevant actual source, not just reports: `basil_core/models.py`, `basil_core/data/cifar.py`, `basil_core/research_protocol.py`, `basil_core/iid_study.py`, `basil_core/iid_campaign.py`, `basil_core/iid_gpu_worker.py`, `basil_core/iid_runtime.py`, `basil_core/attacks.py`, `basil_core/trainer.py`, `basil_core/research_audit.py`, and `basil_core/protocol_checkpoint.py`.
8. Runners/reporting: `scripts/run_iid_condition.py`, `scripts/run_iid_basil_pair.py`, `scripts/analyze_completed_iid.py`, `reporting/iid_study_plots.py`, and `reporting/iid_campaign_plots.py`.
9. Tests: `tests/test_research_protocol.py`, `tests/test_iid_study.py`, `tests/test_iid_campaign.py`, `tests/test_iid_gpu.py`, `tests/test_protocol_recovery.py`, `tests/test_iid_plot_pipeline.py`, and `tests/test_completed_iid_analysis.py`.
10. The BASIL and noisy-communication PDFs under `papers/`, particularly the architecture and equations discussed in the source-equation note.

The code report has 38 verbatim Python excerpts with one-based inclusive current source line ranges. Those are navigation aids, not an assertion that old uncommitted development had a complete commit-by-commit history. Read whole functions where context is needed.

## 2. Project and development history

PaperMerge is a Python/TensorFlow research framework for Byzantine-resilient decentralized/federated learning over noisy communication. It retains BASIL, historical merged/consensus modes, CART, adaptive studies, plotting, execution policies, workers and telemetry.

The desktop application remains Tkinter. The former roughly 5,400-line GUI monolith is now a compatibility shim; `python run_gui.py` launches `gui/app.py`. State, services and views implement five persistent workspaces: Dashboard, Experiment Builder, Queue, Results and Network. Configuration adapters preserve camelCase and compatible unknown fields. Numerical/scientific logic belongs in `basil_core/`, not widgets.

Operational work added readable light-slate inputs/dropdowns, stable incremental updates instead of widget reconstruction, distinct blue/green/orange live accuracy curves, the running experiment name, separate experiment/queue elapsed and ETA labels, completed-round Done/Total and device columns, network events/replay, explicit execution authorization, and graceful stop/recovery. Queue ETA may still say “Estimating…” when pending GPU estimates are unavailable; that is not evidence of stalled training.

Source files use descriptive research terminology. Legacy names may remain in protected provenance or explicit deprecated aliases. Do not cosmetically rewrite historical metadata or claim the current IID engine is the old numbered campaign algorithm merely because a generic source identifier contains “campaign”.

The earlier extreme one-class-per-node work is separate and paused. It observed approximately 10% balanced test accuracy, 100% own-class and 0% other-class accuracy, consistent with strong specialization/forgetting. High relative-L2 noise also produced numerical problems. Those findings are not the results of the current IID experiment.

Snapshot-proximal, adaptive-anchor and Monte Carlo losses were proposed for that separate study. They are not active in these IID runs. Do not infer completed implementation or validation merely from unused configuration field names.

## 3. Current study: four matched IID conditions

All four completed 100 measured rounds, R0 through R99:

| Condition | Attack | Channel | Local loss |
|---|---|---|---|
| A | None; actual attackers 0 | Clean | CE |
| B | Hidden; IDs [0,1,5,7]; starts R20 | Clean | CE |
| C | Same as B | Absolute Gaussian, coordinate sigma 0.010 | CE |
| D | Same as B/C | Same channel and sigma as C | CE + 0.0001 times squared CE-gradient norm |

The protocol ID is `sequential_basil_iid_v1`; the one-class family retains `sequential_basil_one_class_v1`. Seed is 2025. BASIL remains enabled in A as well as B/C/D. A has assumed Byzantine bound four and S=5 even though it has no actual attackers.

Prespecified contrasts:

- B−A: attack-environment effect while BASIL is active in both. Never call this the “cost of BASIL”.
- C−B: communication-noise effect, CE unchanged.
- D−C: primary EBM comparison under the same noisy environment.
- D−A: total remaining gap between combined threats/EBM and ideal clean BASIL.

Do not assume a preferred ranking. Report endpoints that disagree rather than choosing the one that makes EBM look best.

## 4. Fixed data, model and training settings

CIFAR-10 training: 50,000 examples, ten disjoint random IID partitions of 5,000 each. Every node has all ten classes. Counts range from 450 to 545 per class per node; this is compatible with random equal-size IID, which does not require exactly 500 per class. The partition audit checks full coverage, no duplicates/omissions and determinism. The existing correct IID partitioner was reused, not duplicated or replaced.

CIFAR-10 test: 10,000 separate evaluation-only examples, 1,000 per class. Test data must never select snapshots, adjust sigma, choose learning rates, tune objectives, choose stopping time, select attackers, or train the model.

The approved CNN has exactly 117,706 trainable scalar parameters:

```text
32x32x3 input
 -> Conv16, 3x3, valid, ReLU
 -> Pool3, stride3
 -> Conv64, 4x4, valid, ReLU
 -> Pool4, stride4
 -> Flatten64
 -> Dense384, ReLU
 -> Dense192, ReLU
 -> Dense10 logits
```

Conv kernels use seeded Glorot; dense layers use seeded fan-in uniform; biases initialize to zero. Historical VGG-style models remain separate. Do not compare modern ResNet/VGG CIFAR scores as if they were a software correctness threshold for this small pooled network.

Five complete local epochs are fixed. Batch size is 512. Every 5,000-example epoch has nine full batches plus a 392-example remainder, so 50 updates per node activation. There is no dropped partial batch. Each condition has 1,000 activations, 50,000 optimizer batches and 25 million training-image visits.

Optimizer is reset SGD per activation, initial LR 0.05, momentum zero. The approved round schedule is eta(r)=0.05/(1+0.05r): approximately 0.014493 at R49 and 0.008403 at R99. There is no clipping, weight decay, proximal penalty, CART, class registry, snapshot anchor or MC noisy loss. Do not modify these after observing results.

Training-only augmentation is keyed horizontal flip, four-pixel pad and crop to 32x32. Separate keyed streams cover initialization, partition, order, augmentation, attacker choice, attack and channel. Changing objectives does not shift a mutable shared random stream.

## 5. BASIL means memory and selection, not averaging

Each receiver stores the latest snapshot from five distinct counterclockwise predecessors. A sender updates the next five clockwise memories, but only the immediate successor trains next.

```text
Before Node0 at R11: 5:R10 6:R10 7:R10 8:R10 9:R10
After Node0 sends, before Node1:
                    6:R10 7:R10 8:R10 9:R10 0:R11
```

The receiver evaluates every candidate on exactly the same unaugmented receiver-local training batch, up to 512 examples. The current implementation reuses the first 512 indices of the randomized local assignment. Select minimum receiver-local CE; ties within 1e-8 choose nearest counterclockwise sender. Sender-reported metrics and global test accuracy are telemetry only.

The complete selected received snapshot initializes local weights. No averaging candidates, stale receiver mixture, FedAvg or consensus is performed. Saved selected-input and first-batch hashes were checked in the completed audit. An older/different selected branch can discard the newest model's update lineage; executed optimizer steps are not equivalent to one uninterrupted centralized trajectory.

## 6. Attack, channel and objective mathematics

Order: receive/cache -> local SS -> complete load -> reset SGD -> five epochs -> honest trained state/evaluation -> active outbound Hidden attack -> independent per-link noise -> successor memories.

The existing delayed Hidden parameter transform is retained. With current strength/blend defaults it approximately transmits -0.32 times the honest model plus a small perturbation. Attacker nodes still train honestly before corrupting outbound copies. Do not call this a verified exact reproduction of a different attack paper.

C/D use paper-style **absolute** Gaussian corruption:

```text
theta_received = theta_outbound + xi
xi ~ Normal(0, sigma_e^2 I)
sigma_e = 0.010     coordinate standard deviation
sigma_e^2 = 0.0001 coordinate variance
```

No model-norm scaling is used. The separate relative-L2 branch remains available but was not executed in A/B/C/D. Model/noise ratios are diagnostic only. Sigma 0.010 was declared from calibration before results, not selected by maximizing test accuracy.

For D:

```text
F = stable ten-class logits CE
g = gradient(F)
H = Hessian(F)
L = F + sigma_e^2 ||g||^2
gradient(L) = g + 2 sigma_e^2 H g = g + 0.0002 H g
```

Nested TensorFlow tapes differentiate the real modified loss, without materializing a full Hessian. The coefficient is stopped/fixed. No extra lambda is used; an extreme legacy lambda changes neither loss, gradients nor updates in the IID source mode. Legacy scalar gradient scaling remains separate historical behavior. C is CE-only and has zero EBM penalty/correction in its traces.

Source attribution must be precise: BASIL supplies logical-ring memory-assisted receiver-local selection. The noisy-communication work supplies Gaussian noise and motivates expectation-based gradient-norm regularization. Its Eq.14 is a squared-loss approximation; our multiclass CE/CNN/ring is an adaptation. Expected noisy CE is not generally identical to CE plus squared-gradient norm; its usual Taylor expansion involves a Hessian trace. Eq.23 scalar scaling is not the general nonlinear-CNN derivative. Do not erase these distinctions.

## 7. Expected primary results to verify

| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Initial Node-9 accuracy | 9.01% | 9.01% | 9.01% | 9.01% |
| Final Node-9 accuracy | 58.83% | 58.32% | 50.09% | 50.26% |
| Best accuracy | 58.83% | 58.36% | 51.76% | 51.14% |
| Best zero-based round | 99 | 95 | 65 | 93 |
| Final mean-node accuracy | 58.25% | 57.64% | 50.11% | 49.77% |
| Final worst-node accuracy | 57.00% | 56.91% | 49.63% | 47.00% |
| Final Node-9 test CE | 1.181190 | 1.199362 | 1.404584 | 1.394451 |
| Runtime, minutes | 73.06 | 76.96 | 77.46 | 76.76 |

Full-test accuracy means the fixed end-of-ring **honest Node9 model before outbound corruption**. Mean/worst refer to the ten stored trained models. Per-class values in the report are Node9; manifest `finalPerClassAccuracy` instead averages nodes. Do not merge these definitions.

Final Node9 contrasts in percentage points: B−A=-0.51; C−B=-8.23; D−C=+0.17; D−A=-8.57. D−C best=-0.62, late mean approximately +0.58, final node mean approximately -0.34, final worst=-2.63. EBM results are mixed across endpoints; no strong consistent single-seed benefit is established.

Current-run R49 -> R99:

| Condition | R49 | R99 | Change, pp |
|---|---:|---:|---:|
| A | 55.04% | 58.83% | +3.79 |
| B | 54.00% | 58.32% | +4.32 |
| C | 51.62% | 50.09% | -1.53 |
| D | 49.83% | 50.26% | +0.43 |

Mean R80–89 -> R90–99: A 57.55 ->58.30; B 56.91 ->57.67; C 49.68 ->49.92; D 50.07 ->50.50. Late slopes in pp/round: A .0676, B .0691, C .0249, D .0376. A/B still improved slowly; a strict plateau or hard 58% ceiling was not proved.

Historical separate CPU 50-round references were 54.35% clean/attack/CE and 50.57% noisy/attack/EBM. They are not R49 of these GPU runs and should never be spliced into current curves.

## 8. Attack filtering, classes and node consistency

Each condition had 1,000 selection decisions. Attacker-designated sender selections were A0/B73/C74/D70. **Actually attack-corrupted selections were zero in every condition.** C/D each selected one still-honest source-R19 snapshot from a designated attacker during the R20 transition. Source round, not just receiver round, determines corruption.

B/C/D each had 320 attack-active activations and 1,600 attack-corrupted outgoing transmissions. This demonstrates filtering of this attack realization, not universal Byzantine resilience.

| Class | A | B | C | D |
|---|---:|---:|---:|---:|
| airplane | 64.0% | 62.6% | 59.3% | 55.2% |
| automobile | 68.1% | 68.7% | 61.1% | 57.2% |
| bird | 47.1% | 47.6% | 42.4% | 42.0% |
| cat | 36.5% | 37.1% | 31.0% | 28.0% |
| deer | 53.2% | 51.3% | 37.2% | 37.0% |
| dog | 51.9% | 50.8% | 46.6% | 50.0% |
| frog | 70.9% | 70.1% | 63.8% | 57.0% |
| horse | 60.4% | 59.7% | 54.3% | 57.2% |
| ship | 69.2% | 69.1% | 54.6% | 61.5% |
| truck | 67.0% | 66.2% | 50.6% | 57.5% |

Cat is weakest throughout. C->D improves ship/truck by 6.9 pp, but frog falls 6.8 pp. All final models predict all ten labels; this is not constant-class collapse. Final across-node spreads: A1.83, B1.41, C1.06, D4.15 pp. D Node7 is 47.00%, which must not be hidden by Node9's 50.26%.

## 9. Noise/EBM evidence and reproducibility

C/D each have 5,000 independently keyed link draws. All corresponding noise hashes match across C/D, while hashes are distinct within each condition. Matching additive draws does not imply matching transmitted models because objectives change trajectories.

Mean noise L2 is 3.430778, range 3.405944–3.455492, matching .010*sqrt(117706). Mean noise/model ratio is C12.65%, D12.81%; ratios are larger for attacked outbound copies because the attack shrinks model norm, not because sigma changes.

D mean CE=1.516180; EBM penalty=.000589; base-gradient norm=2.270363; correction norm=.046757; applied norm=2.312014. Correction/base ratio averages 1.8385%, maximum5.6458%, late mean2.9389%. This is measurable but modest, not intentionally forced to 2x or generally dominant. Per-link before/after-noise test accuracy was not measured; do not invent that ablation.

All current conditions are **fresh GPU R0–R99**. The latest A rerun removed the earlier mixed-training-device comparison caveat. Canonical initialization and keyed augmentation on CPU are intentional, not CPU-trained prefixes. Old interrupted/mixed evidence remains archived. Fresh manifests do not claim exact CPU-to-GPU continuation or bit-identical arithmetic.

Device was RTX4070Ti, TF2.18.0, Python3.12.3, CUDA12.5.1/cuDNN9; serial single-process float32, TF32 off, one intra/inter-op thread. GPU cap4,096MiB; minimum free memory5,120MiB; one-worker lock. Representative GPU checks do not prove identical CPU/GPU long trajectories. Historical multiprocessing gradient differences were bypassed, not conclusively resolved.

Shared partition hash:
`db79edd6b35d709b0a40fcc9ca5b5253e1da52a414cfba30a87de6aff4339b29`

Shared canonical initialization hash:
`7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df`

Runs recorded commit `09132877c69473785bbd9f6eb8b69538e663db32` and a dirty tree. Do not retroactively change that manifest commit to the later documentation/source-publication commit. Source fingerprints establish what executed. The earlier audit matched scientific source hashes and GPU worker hash.

## 10. Numerical safety, recovery and tests

The completed audit found 100 consecutive measured rounds, all 1,000 activations and 50,000 batches per condition, five complete epochs/remainders, finite saved parameters/gradients, no recorded numerical failure or OOM in these fresh runs, and no fresh-run recovery event. Every scheduled update changed parameters. First-failure checks remain in place.

Activation-boundary checkpoints retain ten node states, five-sender memories with rounds/weights/channel metadata, cursor and metrics. Atomic writes/fsync and previous-version fallback are used; config/source/partition/init mismatches are rejected. Optimizer resets and keyed RNGs permit logical boundary recovery. Loading only final weights is not an exact resume. The sharing ZIP excludes checkpoints exceeding100MB, so do not promise resume from the ZIP alone.

Prior focused verification: **158 passed, two display-dependent skips**. Five analysis-only tests also passed. Tk display connection and unrelated globally auto-loaded ROS pytest plugins were environment limitations, not scientific failures. Do not claim the full legacy suite or multiprocessing problem was fixed. Read current commit/test output for any newer verification count rather than confusing it with the recorded audit count.

The publication check subsequently expanded that focused group with configuration-library, runtime-estimator and worker-pool suites: **178 passed, two display-dependent skips, 644 subtests passed**, two deprecation warnings, in17.50s. Compilation also passed. These are lightweight fixtures, not new research conditions. The older report's158 count describes its original audit command, not a contradiction.

Protected integrity: original roots `experiments/results4/` 2,153 files and `plots4/`737 files, all hashes unchanged. The broader audit recorded5,363 historical files and an expanded5,813-file raw/plot/archive scan unchanged. Do not rewrite, rename, normalize, regenerate or remove historical files, including legacy-named protected trees.

## 11. ZIP evidence scope

Current results are under:

```text
newResults/IID/A_clean_no_attack_ce_100r_gpu/
newResults/IID/B_attack_clean_ce_100r_gpu/
newResults/IID/C_attack_noise_ce_100r_gpu/
newResults/IID/D_attack_noise_ebm_100r_gpu/
newResults/IID/ABCD_100r_comparison_gpu/
```

Plots use corresponding paths under `newPlots/IID/`. Look at the four ABCD comparison PNGs and individual accuracy/class/selection/noise/EBM figures. Cross-check CSV/NPZ, not appearance alone. Accuracy axes in existing PNGs are fractional (0.6=60%).

The publication commit includes bounded existing evidence: reports, verification JSON, configs/manifests, round CSVs, metric NPZs, partition audit/indices, compressed activation telemetry, final node weights, execution logs and current PNGs. It does not include datasets/caches, giant current/previous checkpoints, raw batch traces, local process locks/state or all ignored earlier archives. Full checkpoint/raw-trace checks in the saved audit are historical verified outcomes; absent local inputs cannot be independently rerun from a limited ZIP. State this distinction honestly.

## 12. Main question: why is clean A only about58%?

Do not answer “IID should be80–90%, so code must be broken”. No partition, dropped-batch, missing-epoch, wrong-device, non-finite, objective, handoff or memory defect was found in the completed checks. Correct IID removes the extreme one-class data problem, not model/optimization limitations.

Separate confirmed evidence from hypotheses:

- Small aggressively pooled representation: second feature map7x7x64 becomes1x1x64. Plausible limitation, not proven sole cause.
- Reset SGD without momentum and approved decay slow optimization. No alternate-optimizer/schedule causal comparison exists.
- BASIL can choose older/different branches, discarding the immediate predecessor's update lineage. In A, nearest sender was selected242/1,000 times. Not every other choice is harmful.
- Reused fixed512-example local SS batch may bias branch choice. This is training-side sampling, not test leakage; effect unmeasured.
- Five local epochs on finite random IID subsets may change node/branch specialization, but this is not the one-class experiment.
- Late A augmented pre-update minibatch accuracy56.79% and mean CE1.2298 are not near-perfect training memorization. They are also not clean training-set accuracy; do not use them to prove underfitting or rule out overfitting definitively.
- A still gained3.79pp from its own R49 to R99 and .75pp between final ten-round windows. There is no demonstrated hard58% cap.

The source architecture alone does not reproduce another paper's full benchmark. Its memory/decay/update conventions/reporting differ. Do not import its accuracy numbers as guaranteed results here. Any follow-up ablations require separate authorization and training-side validation, not selecting hyperparameters on the final test set.

## 13. What I want you to return

1. Confirm what files you actually read and what is absent.
2. Explain the implementation and development history accurately, without conflating planned non-IID losses with executed IID objectives.
3. Verify A/B/C/D completion, configs, metric definitions and pairing using attached evidence.
4. Give concise tables of final/best/mean/worst/loss/runtime, contrasts, R49/R99 and late trends, corruption selection, classes, noise/EBM, device and integrity.
5. Explain why approximately58% is not automatically a defect or hard ceiling; rank hypotheses by evidence, not confidence invented from general CIFAR expectations.
6. Interpret D−C across all endpoints, acknowledging its tiny final advantage and worse worst-node result; no universal benefit or significance from one seed.
7. Classify issues as CONFIRMED DEFECT, POSSIBLE CONCERN, EXPECTED VARIATION or NO ISSUE FOUND.
8. Recommend a small number of scientifically interpretable next investigations, clearly separating read-only diagnostics from new experiments. Do not execute them or silently mutate the approved protocol.
9. State single-seed, model-size, fixed-protocol, missing-measurement and reproducibility limitations. The current four conditions are all fresh GPU; do not repeat the obsolete mixed-prefix caveat as if it applied to current results.
10. Finish with a short assessment of what the study actually demonstrates and what additional evidence would be required for stronger claims.

The goal is scientific traceability and correct interpretation—not manufacturing a higher accuracy curve or assuming that source-grounded EBM must outperform its control.
