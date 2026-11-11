> Publication update, 11 November 2026: This brief describes the current 500-round validation study as reviewed on 4 November. Official test evaluation has since been completed for eight runs. Read [the research index](../RESEARCH_INDEX.md) and [the test report](../results/official-test-20261110/four-method-official-test-20261110/report.md) before revising the thesis. Earlier statements that official testing was not performed are historical and are superseded by those results. The equation approval rule below still applies.

# Thesis results and revision brief for Claude

Prepared for Ojaswi Sinha, MS Computer Science, Ohio University; advisor Dr. Animesh Yadav. Evidence reviewed on 4 November 2026. This document is a self-contained revision request and numerical evidence packet. Read it together with the supplied thesis source, bibliography, rendered PDF, SVG figures, and repository. Do not treat it as permission to change the experiments.

## 1. Instructions from the thesis author

Please revise my thesis to reflect the latest completed results. Update the experimental design, Results, Discussion, Conclusions, relevant introduction/contribution statements, and abstract. Explain both **FedSharp and FedSharp+** professionally, in clear language appropriate for a master's thesis. Expand the FedSharp+ explanation enough that a reader can follow the decisions before and after local training. Prepare an updated **Algorithm 2 for FedSharp+**, consistent with the code and subject to the mathematical approval rule below. Tell me exactly where to put every figure and provide Overleaf-ready LaTeX.

**EQUATIONS ARE LOCKED. Do not change, add, remove, renumber deliberately, or reinterpret any equation without my explicit approval.** This applies to numbered and unnumbered mathematics, objectives, assumptions, mathematical rules inside pseudocode, and mathematical material currently commented out in the source. Do not silently correct an equation as a typo or modify it to match the code. Do not activate a commented equation without approval. Formatting or prose must not change mathematical meaning.

If a mathematical change is necessary, put it in a separate **Pending mathematical changes  -  approval required** list. For each item show: (1) exact original LaTeX, (2) exact proposed LaTeX, (3) source label and original line range, (4) reason and code/result evidence, (5) effect on interpretation and dependent text, algorithms, cross-references and proofs. Ask for my explicit approval and wait before applying it. No reply is not approval. Continue independent prose, numerical tables and figure-placement work while approval is pending. Do not present a final internally consistent mathematical revision as completed while unresolved items remain.

Clarifying Algorithm 2's wording, inputs, outputs and stage order is wanted. Changes to its mathematical behavior, or restoring its commented mathematical body to active LaTeX, require the same before-and-after approval. Do not replace the evaluated method with a simpler algorithm to make the thesis easier to explain.

Use these supplied files:

- `thesis.tex`: latest editable thesis source, supplied from my Downloads folder.
- `references.bib`: its bibliography, supplied separately.
- `OSinha_MSCS_Thesis_Robust_DFL_2026.pdf`: rendered appearance and layout reference; it is not proof that the supplied `.tex` currently builds identically.
- Eight individual result SVGs listed in Section 9 below. The corresponding PNGs and SVGs are already in **`Images/` in Overleaf**, with exactly that capitalization.

You have my permission to read `https://github.com/Ojaswi03/PaperMerge` for this revision. Verify the repository commit. The verified results used here are at **`b49fcad66fcb2adf14e4d35531d3d657c82cecd0`**. If HEAD is newer, inspect the diff. Inspect both current code and the **frozen source snapshots packaged with each result export**; the frozen producer establishes what ran. Do not substitute a newer implementation or an older similarly named prototype for the evaluated version. If repository access fails, say precisely what could not be checked; do not claim to have inspected it.

Do not start experiments, edit source code/configurations, alter recorded metrics, overwrite old archives, or push changes to the repository as part of this thesis-editing request. Produce thesis revisions and an evidence-based change report. Preserve the original thesis, bibliography and figures. Do not rewrite acknowledgments, dedication, unrelated literature, or approved mathematics unnecessarily.

## 2. Critical findings in the latest thesis source

The latest `.tex` differs materially from the rendered PDF. Its active FedSharp+ section contains only an introductory paragraph. The detailed explanation, equations, diagram and Algorithm 2 are in a large **commented-out** block. Do not assume Algorithm 2 currently exists in the compiled source. Do not uncomment the entire old block: it also contains a second methodology chapter, duplicate labels and older Algorithm 1 material.

The following locations refer to the ORIGINAL supplied `thesis.tex`; line numbers will move after editing. Use labels and quoted anchors as the durable references.

| Original location | Required action |
|---|---|
| Lines 70-72, `\abstract{...}` | Keep both method names; revise scope and explanation. The file's opening comment intentionally avoids numerical comparisons in the abstract; retain a nonnumerical abstract unless I request otherwise. |
| Lines 198-216, `sec:contributions` | Align contributions and study scope with the current evidence; do not claim universal superiority. |
| Lines 571-744, `sec:method_fedsharp`, `alg:fedsharp` | Audit against the evaluated survivor guard and realized-pool check. Several descriptions currently state a full-window count of 33 and omit runtime safeguards. Flag mathematical discrepancies for approval. |
| Lines 747-748, `sec:cart` | Expand the active FedSharp+ explanation using Sections 7-8 of this brief. |
| Lines 917-1080, commented FedSharp+ text/diagram/equations | Existing material is useful reference, not active mathematics. Propose restoration selectively; approval is required before activating equations. |
| Lines 1111-1134, commented `alg:fedsharpplus` | Prepare the clarified Algorithm 2 here for review, then place the approved version after the expanded active FedSharp+ explanation and before the chapter summary. |
| Lines 1142-1167, `sec:design_scope`, `sec:conditions`, `tab:matrix` | Replace the 84-run/100-round/Hidden-only study description with the current matrix and distinct clean policy. |
| Lines 1174-1179, `sec:architecture` | Models/batches/schedule largely remain relevant; distinguish impaired FedSharp's μ=1 from both new clean controls' μ=0 in both partitions. |
| Lines 1181-1192, `sec:metrics`, `eq:primary` | Primary result now uses rounds 491-500; the existing equation explicitly sums 91-100. Show the exact bound change for approval. Do not silently alter it. |
| Lines 1197-1198, `sec:reproducibility` | Describe 150 selected runs, 500 rounds and the saved verification evidence; distinguish current selection from 186 historical run records. |
| Lines 1203-1204, `sec:principal_results` | Change figure/endpoints/axis/inset descriptions. Remove the assertion that clean and attack-only traces must match before round 21: their current transfer policies differ. |
| Lines 1206-1231, `sec:mnist_iid_results`, `tab:mnist_iid_endpoints` | Replace old figure with two attack-specific figures; include the shared MNIST noise-only control; update results/table. |
| Lines 1233-1260, `sec:mnist_non_iid_results`, `tab:mnist_non_iid_endpoints` | Replace old figure with two; include FedSharp+ and ACDS only in joint conditions, and the noise-only BASIL control. |
| Lines 1262-1287, `sec:cifar_iid_results`, `tab:cifar_iid_endpoints` | Replace old figure with two; new clean accuracy is about 61.75%, above the joint FedSharp curve. |
| Lines 1289-1318, `sec:cifar_non_iid_results`, `tab:cifar_non_iid_endpoints` | Replace old figure with two; separate Gaussian and Hidden findings; FedSharp+ now exceeds ACDS in this comparison. |
| Lines 1320-1370, `sec:mechanisms`, `tab:mech_sharp`, `tab:mech_plus` | Recompute descriptions from the 500-round mechanism summaries below; don't carry forward 21-100 percentages as 21-500. |
| Lines 1372-1392, `sec:diagnostics`, `tab:diagnostics` | Use final round 500 and both attacks. Distinguish per-seed worst-client and worst-class statistics from population retention guarantees. |
| Lines 1394-1405, `sec:interpretation` | Rewrite all four discussion subsections around the new clean controls and mixed extension results. |
| Lines 1407-1410, `sec:historical_results` | Earlier 66-run component evidence belongs to a different revision; retain only with a clear historical boundary. It is not a current FedSharp+ ablation. |
| Lines 1412-1417, `sec:limitations` | Update scope to two attacks, longer horizon, current clean policy, finite reused probes, three seeds, and absent controls. |
| Lines 1419-1425, `sec:conclusion` | Rewrite evidence and future work; preserve conditional mathematical claims, not universal guarantees. |
| Lines 1428-1459, Appendix A | Preserve equations/proofs. Flag any dependency on inactive or mismatched equations for approval. |
| Lines 1461-1468, Appendix B | Explain original versus evaluated guarded rules, superseded clean controls and the current 150-run selection. |

A static label check of active source found these unresolved references (not a full LaTeX compilation): `alg:fedsharpplus`, `ch:method`, `eq:cart_utility`, `eq:channel`, `eq:gradient`, `eq:hidden`, `eq:lambda`, `eq:prior_ce`, `eq:qselected`, `eq:stationary`, `fig:cart`, `sec:basil_baseline`, `sec:noise_analysis`, `sec:sharp`. Some point into the commented block. Review each, distinguishing a harmless reference repair from a mathematical change. Do not fabricate missing definitions or point a reference at a different equation just to remove a warning.

All active citation keys found by the local static scan exist in `references.bib`. This is a key-presence check, **not** verification of bibliographic metadata or every cited scientific claim. Preserve the existing `IEEEtranN` bibliography style and valid keys. Relevant keys include `basil`, `fedprox`, `logit_adjustment`, `chicco_mcc_kappa`. Do not invent sources or treat a related paper as a proof of this implementation.

## 3. Repository evidence and study accounting

The repository result paths below are the authoritative exported measurements. Each export contains `plan.json`, `SHA256.json`, `independent_validation.json`, frozen `source_snapshots/`, and per-run `run.json`, `metrics.npz` and `mechanisms.json`. Continuations include `continuation.json`; optimized runs additionally include execution evidence. Run paths in the appendix expand as `docs/results/<archive-folder>/<runId>/`.

| Code | Path under `docs/results/` | All / current configs | Manifest files | Revision |
|---|---|---|---|---|
| E1 | `hal9k_fedsharp-fleet-500r-20261103/` | 35 / 32 | 241 | `basil_sharp_thesis_20261029_v1` |
| E2 | `themachine_fedsharp-fleet-500r-20261103/` | 115 / 94 | 546 | `basil_sharp_thesis_20261029_v1` |
| E3 | `hal9k_fedsharp-clean-500r-20261104/` | 12 / 0 | 123 | `fedsharp_clean_mu_by_partition_20261103_v1` |
| E4 | `themachine_fedsharp-plain-clean-500r-20261104/` | 12 / 12 | 124 | `fedsharp_plain_clean_ring_20261104_v1` |
| E5 | `themachine_basil-plain-clean-500r-20261104/` | 12 / 12 | 128 | `basil_plain_clean_ring_20261104_v1` |

The **current comparison contains 150 selected run records, 50 condition groups, three seeds per group**. It comprises 126 impaired/noise-only runs retained from E1/E2 plus 12 fresh FedSharp clean controls from E4 and 12 fresh BASIL clean controls from E5. E1/E2 originally contained 150 runs, including 24 old clean controls. E3 added 12 intermediate clean controls. Thus the five exports together contain **186 distinct historical 500-round run records**, of which **36 are superseded clean controls**. They are not extra independent replications of the current result.

Current logical budget: each run has 100 clients × 500 rounds = 50,000 activations and 250,000 SGD steps. The current 150-run selection represents 7,500,000 activations and 37,500,000 SGD steps. E1/E2 continued saved 100-round runs; the latest E4/E5 clean controls began from initialization. The saved first 100 rounds in a continuation are not a separate independent trial. Equal SGD counts do not imply equal compute: selection, calibration, repair and commit checks add forward/CPU work.

Matrix by dataset and partition:

| Dataset/partition | Current groups | Runs | Included conditions |
|---|---:|---:|---|
| MNIST IID | 11 | 33 | BASIL and FedSharp clean; each method attack-only and joint under Gaussian and Hidden; BASIL noise-only |
| MNIST non-IID | 15 | 45 | IID-style groups plus BASIL+ACDS joint and FedSharp+ joint under both attacks |
| CIFAR-10 IID | 10 | 30 | BASIL and FedSharp clean; each method attack-only and joint under both attacks |
| CIFAR-10 non-IID | 14 | 42 | IID-style groups plus BASIL+ACDS joint and FedSharp+ joint under both attacks |

No FedSharp+ clean, IID, attack-only or noise-only cases were run. No FedSharp noise-only controls or CIFAR-10 noise-only controls are available. Reusing the same clean/noise-only trace in Gaussian and Hidden panels does not create new runs. There is no FedSharp+ clean method to describe.

Local review verified all five export manifests and file membership, completed per-run budgets, saved source/config identities and CPU inference receipt coverage. The saved CPU inference receipts re-evaluated the final models, with at most one holdout example difference per model (0.001 accuracy fraction); this review did not repeat inference from private HPC checkpoints locally. The selected current set covers 15,000 final client models. E4/E5 also include direct-transfer routing audits. Do not call a receipt a replay of training or an independent test evaluation.

E1/E2 include ten runs with separately pinned execution optimizations. These are execution changes qualified against the original numerical path, not a new scientific treatment. One completed E1 CIFAR-10/non-IID/2025/BASIL+ACDS run has an explicit outbound-checkpoint recovery receipt: all 100 trained-state and 100 reconstructed outgoing hashes matched, and zero SGD updates were added. Keep this provenance; do not infer a hardware cause or claim corrupted metrics were accepted.

## 4. Experimental details that the revised thesis must state accurately

- The ring has 100 sequentially activated logical clients. One round visits all clients. Each activation performs **five mini-batch SGD steps**, not five epochs.
- MNIST: original TRAIN set of 60,000; remove a fixed stratified development holdout of 1,000 (100 per class), leaving 59,000 fitting examples. CIFAR-10: original TRAIN set of 50,000; the analogous holdout leaves 49,000 fitting examples. **Official TEST partitions are not loaded.** Use "development-holdout accuracy" or clearly defined validation accuracy, not benchmark test accuracy.
- The holdout is common across conditions and seeds, constructed with a fixed key. Fitting allocation varies with seeds 2025, 2026, 2027, and is matched across methods. IID allocation balances each class across clients. Non-IID uses Dirichlet concentration **0.5**. The three seed-level runs, not the 100 linked client states or repeated rounds, are the replicates.
- MNIST model: 784 input, ReLU hidden widths 64 and 32, 10 logits, 52,650 parameters. CIFAR-10: the small BASIL-type CNN with 117,706 parameters described in the supplied source. Inspect `sharp_worker.py` and the actual instantiated model, not another generic model in the repository.
- Batch sizes: 14 MNIST and 80 CIFAR-10, capped by local fitting-set size. Learning rate: the existing schedule in the thesis, evaluated for zero-based rounds 0-499. No momentum, optimizer weight decay, gradient clipping, augmentation or replacement architecture.
- Attack-present conditions configure 33 of 100 identities. Corruption starts at zero-based round 20, displayed round 21. Attack corrupts the stored **outgoing parameters**; a configured attacker's retained local trained model still undergoes training. Thus the primary all-client metric includes retained models of configured attackers before outgoing corruption. It is not an ensemble or a single global model.
- Gaussian attack is **replacement of outgoing parameter coordinates by N(0,1)**. It is distinct from communication noise and is not the same as adding σ=0.01 noise to an honest model. Hidden is the implemented `attacks.hiddenAttack` path, with default blend 0.55 and strength 1.4 in this experiment; inspect the exact call and frozen function. Do not replace its definition by an idealized single-coordinate attack merely because peak screening is motivated that way.
- Link noise has absolute independent Gaussian coordinate SD **0.01** (variance 0.0001), keyed to receiver/sender/version. Re-reading a link version reconstructs the same observation. It is not relative to each layer's weight scale. Noisy conditions have this noise even before attack onset.
- Under impairment, BASIL evaluates up to 34 predecessor copies on the receiver's batch and selects **one** with lowest CE; it does not average 34 or 3 losses/models. Its ordinary objective has μ=0. FedSharp uses a window up to 66, screen/survivor guard/ranking/realized-pool check and a variable realized count, at most 33. It averages parameter vectors, not the 33 losses to form the SGD loss. It can select one model. Its impaired-run proximal μ is **1 even for IID**; changing that would require new experiments, not a thesis relabeling. The noise-related coefficient vanishes in attack-only runs.
- The latest **BASIL clean and FedSharp clean are matched defense-disabled direct-transfer controls**: no attack, no link noise, μ=0 for both IID and non-IID; immediate predecessor accepted without screening, ranking, multi-model averaging, prior adjustment, gradient-norm penalty or class-aware checking. The first activation uses initialization; later ones use the immediate predecessor, including ring wrap. Describe this in Methods even though "plain ring" was removed from visible legend text at my request.
- All 12 clean BASIL/FedSharp pairs have exactly equal full 500×100 accuracy, 500×100×10 per-class accuracy and 500 honest-mean arrays. They are two separately identified matched control sets, **not six independent seeds per curve**. Do not claim full terminal parameter byte equality: checkpoints are not published in these exports. Both curves overlap; keep both labels without artificial offsets.
- Comparing impaired runs with clean now changes both environment and defensive policy. Clean is a useful matched reference, not a mathematically guaranteed upper bound or a controlled isolation of the effect of noise alone. Comparing old and new clean controls changes selection, and in some FedSharp cases μ; do not assign all gains to one component.
- ACDS is only a non-IID joint comparator. Clients share 5% of fitting examples in five cyclic batches, eventually supplying other clients' contributed union in addition to their own data. This changes information access. No cryptographic anonymity is implemented; do not claim a privacy guarantee. FedSharp+ shares no raw examples or class-performance registry. Local-only evidence does not by itself prove differential privacy or resistance to model inversion.

## 5. Metric definitions and numerical results

For each seed and round, first average the recorded accuracy across the 100 retained client models. The plot line averages those three seed means; bands are **sample SD across the three seeds**, unsmoothed. The primary endpoint averages displayed rounds **491-500 within each seed**, then reports mean ± sample SD across seeds. The secondary endpoint is displayed round **500**, summarized the same way. Values below are percentages; differences are percentage points unless explicitly called relative changes. Round-100 summaries use 91-100 from the same trajectories, not an independent run.

**Approval needed:** the source's `eq:primary` currently defines the 91-100 endpoint. The exact original is:

```latex
M^{(10)}_{s,c}=\frac1{10}\sum_{r=91}^{100}m_{r,s,c},
```

The needed proposed replacement is:

```latex
M^{(10)}_{s,c}=\frac1{10}\sum_{r=491}^{500}m_{r,s,c},
```

This is a mathematical bound change, so **show it to me and get approval before applying**. Likewise identify any other explicit 100-round mathematical budgets or bounds; do not use global search-and-replace on all occurrences of "100" (there are still 100 clients, 100 holdout examples per class and historical 100-round analyses).

### 5.1 All 50 current condition summaries

"100" = mean rounds 91-100; "500 primary" = mean rounds 491-500; "500 final" = round 500. Each cell is mean ± sample SD over seeds 2025-2027. Joint = attack + link noise. Attack names are separate treatments. Clean/noise-only have no active attack.

| Dataset | Split | Method | Environment | Attack | 100 | 500 primary | 500 final |
|---|---|---|---|---|---|---|---|
| MNIST | IID | BASIL | attack only | Gaussian | 92.178 ± 0.249 | 94.088 ± 0.203 | 94.069 ± 0.182 |
| MNIST | IID | BASIL | attack only | Hidden | 92.178 ± 0.249 | 94.088 ± 0.203 | 94.069 ± 0.182 |
| MNIST | IID | BASIL | clean |  -  | 97.224 ± 0.372 | 97.602 ± 0.068 | 97.619 ± 0.087 |
| MNIST | IID | BASIL | joint | Gaussian | 82.630 ± 1.056 | 74.182 ± 2.021 | 74.053 ± 2.408 |
| MNIST | IID | BASIL | joint | Hidden | 82.606 ± 1.072 | 73.374 ± 0.898 | 73.092 ± 0.434 |
| MNIST | IID | BASIL | noise only |  -  | 83.351 ± 0.298 | 73.645 ± 0.252 | 73.616 ± 0.454 |
| MNIST | IID | FedSharp | attack only | Gaussian | 91.082 ± 0.237 | 92.600 ± 0.373 | 92.644 ± 0.401 |
| MNIST | IID | FedSharp | attack only | Hidden | 91.003 ± 0.221 | 92.562 ± 0.374 | 92.602 ± 0.304 |
| MNIST | IID | FedSharp | clean |  -  | 97.224 ± 0.372 | 97.602 ± 0.068 | 97.619 ± 0.087 |
| MNIST | IID | FedSharp | joint | Gaussian | 88.787 ± 0.364 | 87.818 ± 0.650 | 87.742 ± 0.732 |
| MNIST | IID | FedSharp | joint | Hidden | 88.787 ± 0.364 | 87.814 ± 0.810 | 87.645 ± 0.649 |
| MNIST | non-IID | BASIL | attack only | Gaussian | 89.824 ± 0.242 | 92.550 ± 0.156 | 92.628 ± 0.211 |
| MNIST | non-IID | BASIL | attack only | Hidden | 89.824 ± 0.242 | 92.536 ± 0.132 | 92.596 ± 0.160 |
| MNIST | non-IID | BASIL | clean |  -  | 97.136 ± 0.608 | 97.628 ± 0.326 | 97.686 ± 0.339 |
| MNIST | non-IID | BASIL | joint | Gaussian | 79.132 ± 1.038 | 70.344 ± 1.764 | 70.426 ± 1.928 |
| MNIST | non-IID | BASIL | joint | Hidden | 79.132 ± 1.038 | 69.320 ± 1.016 | 69.668 ± 1.157 |
| MNIST | non-IID | BASIL | noise only |  -  | 79.097 ± 0.532 | 71.269 ± 0.634 | 70.931 ± 0.725 |
| MNIST | non-IID | BASIL + ACDS | joint | Gaussian | 82.254 ± 0.609 | 74.704 ± 1.608 | 74.678 ± 1.246 |
| MNIST | non-IID | BASIL + ACDS | joint | Hidden | 82.228 ± 0.639 | 74.701 ± 0.662 | 74.474 ± 0.457 |
| MNIST | non-IID | FedSharp | attack only | Gaussian | 86.843 ± 0.137 | 90.076 ± 0.333 | 90.080 ± 0.414 |
| MNIST | non-IID | FedSharp | attack only | Hidden | 86.843 ± 0.137 | 90.048 ± 0.179 | 90.076 ± 0.444 |
| MNIST | non-IID | FedSharp | clean |  -  | 97.136 ± 0.608 | 97.628 ± 0.326 | 97.686 ± 0.339 |
| MNIST | non-IID | FedSharp | joint | Gaussian | 83.965 ± 0.845 | 84.902 ± 1.173 | 85.038 ± 0.888 |
| MNIST | non-IID | FedSharp | joint | Hidden | 83.965 ± 0.845 | 84.841 ± 0.618 | 84.969 ± 0.240 |
| MNIST | non-IID | FedSharp+ | joint | Gaussian | 84.662 ± 0.488 | 84.264 ± 0.352 | 84.253 ± 0.327 |
| MNIST | non-IID | FedSharp+ | joint | Hidden | 84.638 ± 0.446 | 84.296 ± 0.395 | 84.290 ± 0.376 |
| CIFAR-10 | IID | BASIL | attack only | Gaussian | 32.386 ± 0.503 | 41.488 ± 1.361 | 41.517 ± 1.316 |
| CIFAR-10 | IID | BASIL | attack only | Hidden | 32.386 ± 0.503 | 41.488 ± 1.361 | 41.517 ± 1.316 |
| CIFAR-10 | IID | BASIL | clean |  -  | 62.912 ± 0.996 | 61.754 ± 0.328 | 61.676 ± 0.333 |
| CIFAR-10 | IID | BASIL | joint | Gaussian | 29.910 ± 0.578 | 15.960 ± 3.206 | 15.544 ± 2.895 |
| CIFAR-10 | IID | BASIL | joint | Hidden | 29.910 ± 0.578 | 16.201 ± 2.835 | 15.916 ± 2.696 |
| CIFAR-10 | IID | FedSharp | attack only | Gaussian | 29.664 ± 0.916 | 34.855 ± 0.468 | 34.849 ± 0.447 |
| CIFAR-10 | IID | FedSharp | attack only | Hidden | 29.664 ± 0.916 | 34.855 ± 0.468 | 34.849 ± 0.447 |
| CIFAR-10 | IID | FedSharp | clean |  -  | 62.912 ± 0.996 | 61.754 ± 0.328 | 61.676 ± 0.333 |
| CIFAR-10 | IID | FedSharp | joint | Gaussian | 36.315 ± 0.420 | 44.757 ± 1.086 | 44.784 ± 1.196 |
| CIFAR-10 | IID | FedSharp | joint | Hidden | 36.315 ± 0.420 | 44.757 ± 1.086 | 44.784 ± 1.196 |
| CIFAR-10 | non-IID | BASIL | attack only | Gaussian | 16.995 ± 0.246 | 24.812 ± 1.018 | 24.868 ± 1.112 |
| CIFAR-10 | non-IID | BASIL | attack only | Hidden | 16.808 ± 0.490 | 25.025 ± 0.978 | 25.242 ± 0.824 |
| CIFAR-10 | non-IID | BASIL | clean |  -  | 56.174 ± 0.461 | 59.980 ± 0.488 | 59.936 ± 0.474 |
| CIFAR-10 | non-IID | BASIL | joint | Gaussian | 20.907 ± 0.304 | 11.488 ± 2.556 | 11.556 ± 2.635 |
| CIFAR-10 | non-IID | BASIL | joint | Hidden | 20.694 ± 0.187 | 14.770 ± 2.877 | 15.017 ± 2.427 |
| CIFAR-10 | non-IID | BASIL + ACDS | joint | Gaussian | 29.759 ± 0.419 | 19.081 ± 4.407 | 18.421 ± 4.552 |
| CIFAR-10 | non-IID | BASIL + ACDS | joint | Hidden | 29.759 ± 0.419 | 18.475 ± 3.730 | 18.279 ± 3.612 |
| CIFAR-10 | non-IID | FedSharp | attack only | Gaussian | 12.387 ± 1.186 | 18.527 ± 0.586 | 18.526 ± 0.477 |
| CIFAR-10 | non-IID | FedSharp | attack only | Hidden | 12.448 ± 1.405 | 18.317 ± 0.614 | 18.568 ± 0.738 |
| CIFAR-10 | non-IID | FedSharp | clean |  -  | 56.174 ± 0.461 | 59.980 ± 0.488 | 59.936 ± 0.474 |
| CIFAR-10 | non-IID | FedSharp | joint | Gaussian | 18.257 ± 0.362 | 24.680 ± 1.343 | 24.760 ± 1.560 |
| CIFAR-10 | non-IID | FedSharp | joint | Hidden | 19.048 ± 0.508 | 14.626 ± 0.672 | 14.582 ± 0.538 |
| CIFAR-10 | non-IID | FedSharp+ | joint | Gaussian | 27.746 ± 0.647 | 28.983 ± 0.561 | 29.001 ± 0.557 |
| CIFAR-10 | non-IID | FedSharp+ | joint | Hidden | 27.746 ± 0.647 | 28.983 ± 0.561 | 29.001 ± 0.557 |

### 5.2 Evidence-based interpretation by setting

1. **MNIST IID:** both clean controls reach 97.602%. Attack-only BASIL is about 94.09%, above FedSharp's 92.56-92.60%. Under joint impairment FedSharp reaches 87.81-87.82%, compared with BASIL's 73.37-74.18%. BASIL noise-only is 73.645%; this supports describing substantial degradation in this particular noisy regime, but missing FedSharp noise-only controls prevent isolating its noise response.
2. **MNIST non-IID:** matched clean reaches 97.628%. Joint FedSharp is 84.902% Gaussian and 84.841% Hidden; FedSharp+ is 84.264% and 84.296%. Thus the extension is lower by about 0.638 and 0.545 points at this endpoint. Do not claim it improves MNIST mean accuracy over base FedSharp. Both exceed BASIL joint and BASIL+ACDS joint. ACDS reaches 74.704% and 74.701%. BASIL noise-only is 71.269%.
3. **CIFAR-10 IID:** clean is 61.754%; BASIL attack-only is 41.488%, FedSharp attack-only 34.855%, BASIL joint about 15.96-16.20%, and FedSharp joint 44.757%. The old "noisy FedSharp beats clean" result **does not hold with the latest matched clean controls**. Joint FedSharp still exceeds its own attack-only result, but that contrast does not identify a causal benefit of noise because the noise-related objective also changes. Do not assert noise is beneficial or infer a mechanism from the ordering alone.
4. **CIFAR-10 non-IID:** clean reaches 59.980%. Under Gaussian joint impairment FedSharp reaches 24.680%, above BASIL 11.488%; under Hidden, FedSharp 14.626% and BASIL 14.770% have very similar means. FedSharp+ reaches 28.983% under each attack: about 4.302 points above base FedSharp for Gaussian and 14.357 for Hidden. It also exceeds ACDS at 19.081% Gaussian and 18.475% Hidden. These are the clearest mean-accuracy benefits of the extension in this study, while remaining far below the clean control.

The equality of some Gaussian/Hidden endpoint means must be retained, not perturbed to make curves look different. Equality of summary values alone does not establish full trajectory or parameter equality. Check exported arrays/mechanisms before explaining overlaps.

Longer training is not uniformly beneficial. FedSharp CIFAR-10/non-IID joint changes from 18.257 to 24.680% under Gaussian but from 19.048 to 14.626% under Hidden. FedSharp+ CIFAR-10 rises from about 27.746 to 28.983%, while its MNIST mean slips slightly from its 100-round level. Several noisy BASIL/ACDS cases decline. Discuss sustained training empirically; do not infer monotone convergence or that 500 rounds fixes a weak algorithm.

With only three seeds, do not use "significant" unless a valid explicitly reported test supports it. Do not treat SD as a confidence interval, count 100 clients as independent replications, infer unseen attack performance, or use a final worst-class summary to claim retention of every class throughout training.

### 5.3 Updated final-client and class diagnostics

The table below covers all 16 non-IID joint groups. Honest mean uses the 67 non-attacker identities. Worst client = minimum of the 100 client accuracies within each seed at round 500. Client P10 = within-seed 10th percentile. Worst class = minimum across the ten class recalls after averaging that class's recall across the 100 clients. Then summarize each statistic over three seeds. This is not the worst client-class pair and not a cross-seed minimum.

| Dataset | Method | Attack | Honest final | Worst client | Client P10 | Worst class |
|---|---|---|---|---|---|---|
| MNIST | BASIL | Gaussian | 70.496 ± 1.936 | 66.567 ± 1.848 | 68.657 ± 1.594 | 51.927 ± 8.028 |
| MNIST | BASIL | Hidden | 69.746 ± 1.071 | 46.033 ± 33.301 | 67.697 ± 1.837 | 53.290 ± 1.065 |
| MNIST | BASIL + ACDS | Gaussian | 74.644 ± 1.272 | 72.700 ± 1.852 | 73.400 ± 1.480 | 58.177 ± 2.548 |
| MNIST | BASIL + ACDS | Hidden | 74.249 ± 0.553 | 27.367 ± 39.232 | 73.897 ± 0.355 | 57.440 ± 1.771 |
| MNIST | FedSharp | Gaussian | 84.974 ± 0.868 | 80.933 ± 1.290 | 82.593 ± 1.173 | 75.553 ± 1.964 |
| MNIST | FedSharp | Hidden | 84.877 ± 0.256 | 80.000 ± 1.345 | 82.563 ± 0.686 | 73.347 ± 1.685 |
| MNIST | FedSharp+ | Gaussian | 84.302 ± 0.387 | 72.200 ± 3.816 | 81.123 ± 0.830 | 73.607 ± 0.638 |
| MNIST | FedSharp+ | Hidden | 84.334 ± 0.433 | 72.200 ± 3.816 | 81.130 ± 0.838 | 73.723 ± 0.848 |
| CIFAR-10 | BASIL | Gaussian | 11.572 ± 2.662 | 9.633 ± 0.379 | 10.363 ± 0.629 | 7.293 ± 2.465 |
| CIFAR-10 | BASIL | Hidden | 15.031 ± 2.689 | 9.600 ± 0.693 | 11.853 ± 1.609 | 7.247 ± 2.709 |
| CIFAR-10 | BASIL + ACDS | Gaussian | 18.341 ± 4.566 | 12.067 ± 2.248 | 15.493 ± 4.971 | 10.083 ± 1.721 |
| CIFAR-10 | BASIL + ACDS | Hidden | 18.120 ± 3.666 | 12.000 ± 4.158 | 15.310 ± 4.338 | 9.750 ± 2.762 |
| CIFAR-10 | FedSharp | Gaussian | 24.577 ± 1.454 | 14.667 ± 2.658 | 19.787 ± 1.072 | 11.900 ± 2.746 |
| CIFAR-10 | FedSharp | Hidden | 14.893 ± 0.286 | 9.433 ± 0.723 | 10.667 ± 0.582 | 9.903 ± 0.846 |
| CIFAR-10 | FedSharp+ | Gaussian | 29.044 ± 0.524 | 22.367 ± 1.002 | 26.350 ± 0.650 | 15.517 ± 1.622 |
| CIFAR-10 | FedSharp+ | Hidden | 29.044 ± 0.524 | 22.367 ± 1.002 | 26.350 ± 0.650 | 15.517 ± 1.622 |

The extension's benefits are not uniform across client metrics. For CIFAR-10 the final worst-client mean is about 22.367% with FedSharp+, compared with 14.667% Gaussian or 9.433% Hidden for base FedSharp. For MNIST, FedSharp+'s worst-client mean is **72.200%**, lower than base FedSharp's 80.933%/80.000%; its client P10 and some class statistics also do not uniformly improve. State this tradeoff explicitly if discussing fairness or class retention. These are descriptive outcomes, not a demonstrated fairness guarantee.

### 5.4 Updated mechanism evidence

Mechanism values are activation-weighted within a seed/window, then averaged across three seeds. The summaries below use rounds **21-500**, 48,000 activations per run. Screen survivor count means raw passing candidates, not proof they are honest. Realized q records what actually entered the model; FedSharp+ may admit its own model or one peer. A large pass rate does not prove the screen identifies attackers.

| Dataset | Split | Environment | Attack | Raw survivors /66 | Guard fallback % | All-fail % | Realized q | Mean λ |
|---|---|---|---|---|---|---|---|---|
| MNIST | IID | attack only | Gaussian | 43.599 ± 7.644 | 20.785 ± 22.303 | 0.001 ± 0.002 | 1.184 ± 0.016 | 0.000000 ± 0.000000 |
| MNIST | IID | attack only | Hidden | 36.206 ± 26.303 | 49.969 ± 45.370 | 6.984 ± 12.093 | 1.072 ± 0.064 | 0.000000 ± 0.000000 |
| MNIST | IID | joint | Gaussian | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.960 ± 0.019 | 0.014206 ± 0.000012 |
| MNIST | IID | joint | Hidden | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.942 ± 0.021 | 0.014219 ± 0.000015 |
| MNIST | non-IID | attack only | Gaussian | 21.886 ± 0.122 | 99.922 ± 0.099 | 0.004 ± 0.002 | 1.000 ± 0.000 | 0.000000 ± 0.000000 |
| MNIST | non-IID | attack only | Hidden | 21.790 ± 0.028 | 99.999 ± 0.002 | 0.004 ± 0.002 | 1.000 ± 0.000 | 0.000000 ± 0.000000 |
| MNIST | non-IID | joint | Gaussian | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.132 ± 0.032 | 0.014150 ± 0.000019 |
| MNIST | non-IID | joint | Hidden | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.124 ± 0.040 | 0.014154 ± 0.000021 |
| CIFAR-10 | IID | attack only | Gaussian | 21.764 ± 0.001 | 100.000 ± 0.000 | 0.004 ± 0.002 | 1.000 ± 0.000 | 0.000000 ± 0.000000 |
| CIFAR-10 | IID | attack only | Hidden | 20.957 ± 1.398 | 100.000 ± 0.000 | 0.009 ± 0.010 | 1.000 ± 0.000 | 0.000000 ± 0.000000 |
| CIFAR-10 | IID | joint | Gaussian | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9.751 ± 0.229 | 0.010925 ± 0.000123 |
| CIFAR-10 | IID | joint | Hidden | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9.751 ± 0.229 | 0.010925 ± 0.000123 |
| CIFAR-10 | non-IID | attack only | Gaussian | 21.764 ± 0.001 | 100.000 ± 0.000 | 0.004 ± 0.002 | 1.000 ± 0.000 | 0.000000 ± 0.000000 |
| CIFAR-10 | non-IID | attack only | Hidden | 22.058 ± 0.018 | 100.000 ± 0.000 | 0.004 ± 0.002 | 1.000 ± 0.000 | 0.000000 ± 0.000000 |
| CIFAR-10 | non-IID | joint | Gaussian | 65.996 ± 0.004 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3.563 ± 0.262 | 0.013369 ± 0.000143 |
| CIFAR-10 | non-IID | joint | Hidden | 66.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.033 ± 0.009 | 0.014623 ± 0.000006 |

FedSharp's joint mean q is much smaller than 33. It is about 9.751 on CIFAR-10 IID, 3.563 on CIFAR-10 non-IID Gaussian and 1.033 on non-IID Hidden, and around 2 on MNIST. Raw survivors are almost 66 and survivor-guard fallbacks are near zero in these windows. Thus a full set passing the peak screen does **not** mean 33 models are always averaged; the realized-pool check can make the actual choice much smaller. Do not replace this with "mean of the 33 lowest losses."

| Dataset | Attack | Own % | Peer % | Pool % | Active gate % | Qualified classes | Full % | Half % | Quarter % | Eighth % | Zero % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | Gaussian | 94.758 ± 0.456 | 3.933 ± 0.454 | 1.310 ± 0.138 | 100.000 ± 0.000 | 6.027 ± 0.065 | 68.751 ± 0.701 | 8.571 ± 0.395 | 5.724 ± 0.278 | 4.294 ± 0.239 | 12.660 ± 0.178 |
| MNIST | Hidden | 95.103 ± 0.221 | 3.669 ± 0.276 | 1.228 ± 0.146 | 100.000 ± 0.000 | 6.027 ± 0.065 | 68.589 ± 0.930 | 8.505 ± 0.306 | 5.674 ± 0.201 | 4.294 ± 0.237 | 12.938 ± 0.458 |
| CIFAR-10 | Gaussian | 99.619 ± 0.364 | 0.248 ± 0.241 | 0.133 ± 0.123 | 100.000 ± 0.000 | 5.281 ± 0.276 | 30.085 ± 1.519 | 16.904 ± 0.402 | 13.378 ± 0.402 | 10.723 ± 0.570 | 28.909 ± 1.297 |
| CIFAR-10 | Hidden | 99.619 ± 0.364 | 0.248 ± 0.241 | 0.133 ± 0.123 | 100.000 ± 0.000 | 5.281 ± 0.276 | 30.085 ± 1.519 | 16.904 ± 0.402 | 13.378 ± 0.402 | 10.723 ± 0.570 | 28.909 ± 1.297 |

"Full/half/quarter/eighth/zero" are percentages **among active commit checks**, not necessarily all activations in other windows. The informative gate is active for essentially all rounds 21-500 in the current extension runs. CIFAR-10 admits its own carrier about 99.619% of these activations, versus about 94.758-95.103% on MNIST. CIFAR-10 uses full commits about 30.085% and zero commits about 28.909%; the remaining checks accept partial updates. This is evidence of conservative local continuity, not proof that incoming models are always malicious or that rejecting updates causes the accuracy gain. Predominant owned-model routing also limits collaboration and helps explain why identical attacks can have close outcomes, without proving independence from communication.

To update the original early/postattack mechanism table, use the separate **1-20 and 21-500** summaries next. Don't label 21-100 figures as 21-500. The "uninformative" period and calibration overhead are part of the method, not silent exclusions.

| Dataset | Attack | Rounds | Own % | Peer % | Pool % | Active gate % | Full % of active | Zero % of active |
|---|---|---|---|---|---|---|---|---|
| MNIST | Gaussian | 1-20 | 64.550 ± 0.568 | 31.400 ± 0.910 | 4.050 ± 0.350 | 94.933 ± 0.076 | 35.779 ± 1.630 | 15.011 ± 1.017 |
| MNIST | Gaussian | 21-500 | 94.758 ± 0.456 | 3.933 ± 0.454 | 1.310 ± 0.138 | 100.000 ± 0.000 | 68.751 ± 0.701 | 12.660 ± 0.178 |
| MNIST | Hidden | 1-20 | 64.550 ± 0.568 | 31.400 ± 0.910 | 4.050 ± 0.350 | 94.933 ± 0.076 | 35.779 ± 1.630 | 15.011 ± 1.017 |
| MNIST | Hidden | 21-500 | 95.103 ± 0.221 | 3.669 ± 0.276 | 1.228 ± 0.146 | 100.000 ± 0.000 | 68.589 ± 0.930 | 12.938 ± 0.458 |
| CIFAR-10 | Gaussian | 1-20 | 74.700 ± 0.755 | 22.733 ± 0.621 | 2.567 ± 0.284 | 92.583 ± 0.076 | 21.260 ± 2.326 | 29.811 ± 1.328 |
| CIFAR-10 | Gaussian | 21-500 | 99.619 ± 0.364 | 0.248 ± 0.241 | 0.133 ± 0.123 | 100.000 ± 0.000 | 30.085 ± 1.519 | 28.909 ± 1.297 |
| CIFAR-10 | Hidden | 1-20 | 74.700 ± 0.755 | 22.733 ± 0.621 | 2.567 ± 0.284 | 92.583 ± 0.076 | 21.260 ± 2.326 | 29.811 ± 1.328 |
| CIFAR-10 | Hidden | 21-500 | 99.619 ± 0.364 | 0.248 ± 0.241 | 0.133 ± 0.123 | 100.000 ± 0.000 | 30.085 ± 1.519 | 28.909 ± 1.297 |

## 6. Abstract and chapter revision direction

The current abstract already names both methods. Make the distinction clearer without claiming a benefit in every setting. A possible **nonnumerical** replacement, subject to checking the approved mathematics and final chapter text, is:

> Decentralized federated learning supports collaborative training without a central parameter server, but sequential model exchange can propagate Byzantine corruption and communication noise across statistically heterogeneous clients. This thesis studies FedSharp, a directed-ring method that screens predecessor models, ranks them using receiver-local losses, checks the quality of a selected parameter average, and trains with a noise-aware proximal objective. Its extension, FedSharp+, uses private class-recognition evidence to choose a complete incoming model before training and to check the complete model update before commitment. The extension also adjusts the training loss using the receiver's class prior and includes a local classifier-repair stage. A controlled study evaluates these methods on IID and non-IID MNIST and CIFAR-10 allocations over 500 ring rounds and three seeds, using matched clean controls, two Byzantine attacks, and a common development holdout. FedSharp improves joint attack-and-noise performance over BASIL in several settings, but its advantage depends on the dataset and attack. FedSharp+ gives its clearest gains on non-IID CIFAR-10 and does not uniformly improve MNIST accuracy or client-level outcomes. The results distinguish the contribution of guarded model exchange from the information advantage of a fitting-data-sharing comparator and identify the limits of finite local evidence and a small repeated-seed evaluation.

Do not claim a theorem was derived in active thesis content while its statement/proof references are commented or unresolved. After approved mathematical restoration, a concise appropriately qualified theoretical contribution can be retained in the abstract. Use "Basil" or "BASIL" consistently with the thesis style; do not change citation keys.

Rewrite Discussion around (a) setting-dependent guarded selection, (b) corrected matched clean controls and noise/objective interaction, (c) the extension's CIFAR benefit and MNIST/client tradeoffs, and (d) comparison with ACDS's extra fitting information. The conclusion should answer the existing research questions using these observations. Future work can include independent test evaluation, more seeds and architectures, a current-method ablation, matched noise-only controls, runtime/communication cost measurements and evaluation of broader attacks. Do not present these as completed experiments.

## 7. Explain FedSharp+ clearly and faithfully

Use a short overview first: **before training the receiver chooses a complete model; after training it checks how much of the complete update to keep.** Both decisions use its own fitting data. The local class prior affects training, while candidate evaluation and final prediction use raw logits. Explain each state and decision before introducing the approved mathematics. Avoid claiming the entire method is merely a classifier correction, a class registry, or a per-class splice of several models.

The evaluated scientific entry point is `basil_core/basil_sharp_thesis.py`, especially `configure_worker` and `intervention`. Its installed context composes wrappers; reading one older prototype in isolation is insufficient. Relevant frozen/current modules:

| Stage | Files/functions to inspect |
|---|---|
| Base selection, survivor guard, score batching | `cart_batched_scoring.py:select_input`, `sharp_math.py:peak_screen`, `ranked_indices`, `hop_coefficient` |
| Realized-pool check | `sharp_checked_pool.py:selector` |
| Coherent candidate admission | `cart_prior_neutral_routing.py:selector`, `cart_peer_candidates.py:eligible_senders` |
| Class evidence and utility | `cart_class_skill.py:class_skill`, imported `cart_quality_routing_candidate.py:utility` |
| Training-time prior and local update | `cart_local_prior.py`, `cart_admitted_continuity_candidate.py`, `cart_continuity_candidate.py` |
| Fixed local probes | `sharp_cart.py:probes` |
| Inherited head repair and stored reference | `sharp_cart_v2.py:fit_correction`, `transport` |
| Complete-update check and reference rollback | `cart_prior_neutral_commit.py:selector`, `commit`, `transport` |
| Actual activation order/outgoing corruption | `sharp_cart_v2_protocol.py:run`, `sharp_protocol.py:link_vector`, `research_protocol.py`, `attacks.py` |
| Latest clean overrides | `fedsharp_plain_clean.py`, `basil_plain_clean.py`; older `fedsharp_clean_controls.py` is superseded for the current clean curves |

All short filenames in this table are under `basil_core/`, except inspect the actual imported location for attacks. Use the copied `source_snapshots/` versions in E1/E2 to confirm historical behavior. Symbol names that mention CART, F1 or "candidate" are internal implementation history, not necessarily the algorithm or metric named by the thesis. In particular the admission code aliases `class_skill` as `class_f1`, but the actual current recognition score is **one-versus-rest MCC, not F1**.

Explain the stages in this order:

1. **Build complete candidates.** Begin with the checked base FedSharp model, up to eight eligible received peer copies, and the receiver's own retained complete model if it has trained before. Peer eligibility follows the base screen; when the guard falls back the eligible window is restored. Candidate enumeration takes newest versions, then nearest among equal versions. This is different from the later admission priority, which favors nearest predecessor, then pooled candidate, then owned candidate. No raw examples or class registry are transmitted.
2. **Build two fixed private probes.** They are disjoint subsets of the receiver's fitting data, not the common development holdout. Each uses up to 16 examples per locally supported class, with the fitting probe capped by the half-count rule in `probes`. Use only classes present in both probes. The same probes are reused, so they are adaptive local evidence, not fresh tests.
3. **Measure recognition.** Use raw-logit one-versus-rest MCC on each probe; set a numerically undefined denominator to zero. The owned model supplies conservative class targets. A class qualifies only with at least eight examples in each probe and target at least 0.2. The whole gate becomes informative when the smaller of the two owned-model mean MCCs exceeds 0.02; individual class qualification is a different condition.
4. **Allow for fresh link perturbation only when needed.** If the gate is informative, there is a qualified class, σ is nonzero and some candidate falls below its target, draw 19 keyed perturbations of the owned model. Their largest observed MCC drop supplies the empirical allowance. It is not a confidence or conformal guarantee. This allowance is distinct from the cross-entropy allowance later stored for classifier repair.
5. **Admit one model.** With fewer than two supported classes, use the checked base model and no quality gate. With adequate support but an uninformative owned model, use the priority rule without the quality gate; do not say "choose maximum utility" in this branch. Otherwise take the highest-priority candidate within utility tolerance 0.01 of the best. Utility combines conservative mean MCC and penalties for qualified-class deficits beyond the allowance, normalized over all supported classes. The candidate is an entire coherent model, not a mixture of class-specific layers.
6. **Train from the admitted model.** It is both initialization and proximal anchor. Retain the base candidate's realized q if it wins; a single peer or owned model has q=1. There is no new link-noise coefficient when the owned model is retained or no incoming copy exists; this does not remove historical error. Use the receiver's full fitting-label counts, floored at one per class, for unit-strength training logit adjustment. Exactly five SGD steps operate on the adjusted objective. In impaired FedSharp+ μ=1. Raw logits remain the basis for ranking, probes, repair checks and evaluation.
7. **Apply the inherited repair.** This substage modifies only the final affine head in the trained proposal's current feature representation. A stored receiver-local repair reference is distinct from the owned carrier used at admission. Use the implemented class CE/correct-count gates on both probes, reference-correct/proposal-wrong residuals, class-balanced ridge fit and checked head scales. Reference admission has its own CE/correct-count conditions and keyed allowance. Do not suggest the reference is the previous peer's model or that every activation necessarily repairs the head.
8. **Check the complete update and commit.** When the admission gate was informative, try complete-parameter scales 1, 1/2, 1/4, 1/8, 0 in that order. Use the same admission evidence and tolerance to accept the first qualifying update. Zero returns the admitted model, not necessarily the receiver's original model. No additional SGD steps are added. This can change the backbone as well as the classifier. If a partial/zero scale is chosen, retain the previous repair reference, because the full proposal's newly evaluated reference would not describe the committed model. When the gate is inactive, commit the trained/repaired proposal without this scale check.
9. **Store and transmit.** Save the committed retained model; apply the specified outgoing attack only to configured Byzantine identities after onset. Link noise is reconstructed separately for each receiver/sender/version observation.

If restoring the finite-probe single-activation utility inequality, keep its assumptions explicit. It concerns a deterministic finite utility at one activation. It does not establish population accuracy, every-class retention, monotonic improvement over rounds, convergence, adaptive-attack robustness, or privacy. Do not call this empirical calibration a probabilistic certificate.

## 8. Algorithm 2 revision request and mathematical approval issues

Prepare a readable algorithm with inputs (receiver, local fitting data/probes, owned model, checked FedSharp candidate and provenance, previous repair reference, hyperparameters), outputs (committed complete model, reference state, outgoing version), and the nine stages above. Separate the three admission branches, the five training steps, head repair, and whole-model commitment. Use existing notation consistently; do not add equations solely to shorten pseudocode.

The commented source already contains the intended mathematical definitions under these labels: `eq:mcc`, `eq:qualifier`, `eq:informative`, `eq:allowance`, `eq:cart_utility`, `eq:prior`, `eq:prior_ce`, `eq:plus_objective`, `eq:repair_gate`, `eq:ridge`, `eq:commit_scales`, `eq:commit_gate`, `eq:two_gate`. In the older rendered PDF these appeared as Eqs. 3.13-3.25, but **do not hard-code those numbers in the revised source**. The active chapter's different equation count and commented state mean numbering is not currently reliable. Use labels only after their restoration is approved and checked.

For Algorithm 2 show an original-to-proposed change map: state/provenance clarified; eligibility versus priority distinguished; all three admission branches explicit; conditional 19-draw calibration explicit; q and anchor tied to the admitted source; training-only prior distinguished from raw-logit evaluation; repair reference distinguished from owned model; head-only repair distinguished from whole-model commit; atomic reference rollback explicit; no extra SGD steps; outbound attack and receiver noise separated. Where this only clarifies an existing operation, say so. Where a line's formula/behavior changes, mark it pending approval.

At minimum present these pending mathematical/source consistency issues for review:

- **Endpoint equation:** the exact 91-100 → 491-500 proposal in Section 5; do not apply without approval.
- **Commented definitions and Algorithm 2:** show precisely which original commented blocks would become active and which wording changes are proposed. Restoration adds active mathematical content and is approval-gated. Keep duplicated methodology chapters and old Algorithm 1 commented.
- **Base selected count and coefficient:** active `eq:method_selection_budget` and the active Algorithm 1 calculate the count from the available window and state a full-window q=33, while the evaluated method uses post-screen survivor guards and checks the realized parameter average. In particular code uses one candidate if the passed set is no larger than the design budget; otherwise it targets at most the number of survivors minus that budget, capped by the budget; the pool check can then reduce actual q to one. The actual coefficient uses that realized q. Show exact source versus proposed mathematical lines and obtain approval; do not conceal the difference in a caption or change experimental q to 33.
- **Formal versus experimental controls:** active formal algorithms do not directly describe the latest defense-disabled clean controls. Explain the configured clean branch in prose and a method/configuration table. If a formal algorithm must change to include it, propose the change for approval. Likewise avoid silently replacing a Hidden-only formal transmit step by a Gaussian/Hidden switch.
- **Broken proof references:** Appendix A/B refer to labels currently commented out; restoring, renaming or changing referenced mathematics requires explicit review of their meaning. Do not claim an old proof covers a different selection rule or adaptive setting.

Until approvals arrive, supply ready-to-use prose/results/figures and a separate candidate Algorithm 2, clearly marked **DRAFT  -  mathematical restoration/change pending approval**. Do not label the full thesis compile-ready when necessary mathematical decisions are unresolved.

## 9. Exact Overleaf plot placement and captions

Use the eight individual thesis figures. They contain FedSharp+ in the four non-IID panels; no FedSharp+ IID data exist. The two combined research-paper strips (`mnist_accuracy`, `cifar10_accuracy`) deliberately **exclude FedSharp+** and should not replace the thesis's extension evidence. They may be optional overview figures if clearly described.

Every image reference is relative to **`Images/`**, not a local Windows/WSL path. Existing source already loads `svg`, sets `\svgpath{{Images/}}`, and defines `\studyplot` at lines 36-46. It currently includes SVG through that macro at about 0.93 text width. Reuse it if it compiles in this project; do not add a duplicate macro/package. The SVG workflow uses Inkscape/shell escape as noted in the source. If unavailable, use the matching supplied PNG via `graphicx`, not a made-up PDF filename. SVGs have outlined text; PNGs are 600 DPI. Keep each plot close to full text width so the large legend below remains readable. Do not crop the legend or put two plots side by side at half width.

Replace the existing single figure environment in each result section with the following two in Gaussian-then-Hidden order. Use the original section labels as anchors; do not promise exact PDF page numbers after floats and numbering change. Replace outdated prose rather than pasting new claims after contradictory old paragraphs. Keep table labels if appropriate and update all figure cross-references, list of figures, captions and discussion references.

### 1. MNIST IID, Gaussian

In the section labeled `sec:mnist_iid_results`, replace the old figure environment labeled `fig:mnist_iid_accuracy` with the pair specified here, this one first. Use **`Images/mnist_iid_gaussian_accuracy.svg`**; matching fallback: `Images/mnist_iid_gaussian_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{mnist_iid_gaussian_accuracy}{MNIST IID, Gaussian attack}
\caption{MNIST IID development-holdout accuracy over 500 ring rounds under the Gaussian attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap. The BASIL noise-only control is shared with the other attack panel.}
\label{fig:mnist-iid-gaussian-500r}
\end{figure}
```

### 2. MNIST IID, Hidden

In the section labeled `sec:mnist_iid_results`, replace the old figure environment labeled `fig:mnist_iid_accuracy` with the pair specified here, this one second. Use **`Images/mnist_iid_hidden_accuracy.svg`**; matching fallback: `Images/mnist_iid_hidden_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{mnist_iid_hidden_accuracy}{MNIST IID, Hidden attack}
\caption{MNIST IID development-holdout accuracy over 500 ring rounds under the Hidden attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap. The BASIL noise-only control is shared with the other attack panel.}
\label{fig:mnist-iid-hidden-500r}
\end{figure}
```

### 3. MNIST non-IID, Gaussian

In the section labeled `sec:mnist_non_iid_results`, replace the old figure environment labeled `fig:mnist_non_iid_accuracy` with the pair specified here, this one first. Use **`Images/mnist_non_iid_gaussian_accuracy.svg`**; matching fallback: `Images/mnist_non_iid_gaussian_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{mnist_non_iid_gaussian_accuracy}{MNIST non-IID, Gaussian attack}
\caption{MNIST non-IID development-holdout accuracy over 500 ring rounds under the Gaussian attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap. The BASIL noise-only control is shared with the other attack panel. BASIL with ACDS and FedSharp+ are shown under joint attack and noise only.}
\label{fig:mnist-non-iid-gaussian-500r}
\end{figure}
```

### 4. MNIST non-IID, Hidden

In the section labeled `sec:mnist_non_iid_results`, replace the old figure environment labeled `fig:mnist_non_iid_accuracy` with the pair specified here, this one second. Use **`Images/mnist_non_iid_hidden_accuracy.svg`**; matching fallback: `Images/mnist_non_iid_hidden_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{mnist_non_iid_hidden_accuracy}{MNIST non-IID, Hidden attack}
\caption{MNIST non-IID development-holdout accuracy over 500 ring rounds under the Hidden attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap. The BASIL noise-only control is shared with the other attack panel. BASIL with ACDS and FedSharp+ are shown under joint attack and noise only.}
\label{fig:mnist-non-iid-hidden-500r}
\end{figure}
```

### 5. CIFAR-10 IID, Gaussian

In the section labeled `sec:cifar_iid_results`, replace the old figure environment labeled `fig:cifar_iid_accuracy` with the pair specified here, this one first. Use **`Images/cifar10_iid_gaussian_accuracy.svg`**; matching fallback: `Images/cifar10_iid_gaussian_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{cifar10_iid_gaussian_accuracy}{CIFAR-10 IID, Gaussian attack}
\caption{CIFAR-10 IID development-holdout accuracy over 500 ring rounds under the Gaussian attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap.}
\label{fig:cifar10-iid-gaussian-500r}
\end{figure}
```

### 6. CIFAR-10 IID, Hidden

In the section labeled `sec:cifar_iid_results`, replace the old figure environment labeled `fig:cifar_iid_accuracy` with the pair specified here, this one second. Use **`Images/cifar10_iid_hidden_accuracy.svg`**; matching fallback: `Images/cifar10_iid_hidden_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{cifar10_iid_hidden_accuracy}{CIFAR-10 IID, Hidden attack}
\caption{CIFAR-10 IID development-holdout accuracy over 500 ring rounds under the Hidden attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap.}
\label{fig:cifar10-iid-hidden-500r}
\end{figure}
```

### 7. CIFAR-10 non-IID, Gaussian

In the section labeled `sec:cifar_non_iid_results`, replace the old figure environment labeled `fig:cifar_non_iid_accuracy` with the pair specified here, this one first. Use **`Images/cifar10_non_iid_gaussian_accuracy.svg`**; matching fallback: `Images/cifar10_non_iid_gaussian_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{cifar10_non_iid_gaussian_accuracy}{CIFAR-10 non-IID, Gaussian attack}
\caption{CIFAR-10 non-IID development-holdout accuracy over 500 ring rounds under the Gaussian attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap. BASIL with ACDS and FedSharp+ are shown under joint attack and noise only.}
\label{fig:cifar10-non-iid-gaussian-500r}
\end{figure}
```

### 8. CIFAR-10 non-IID, Hidden

In the section labeled `sec:cifar_non_iid_results`, replace the old figure environment labeled `fig:cifar_non_iid_accuracy` with the pair specified here, this one second. Use **`Images/cifar10_non_iid_hidden_accuracy.svg`**; matching fallback: `Images/cifar10_non_iid_hidden_accuracy.png`.

```latex
\begin{figure}[htbp]
\centering
\studyplot{cifar10_non_iid_hidden_accuracy}{CIFAR-10 non-IID, Hidden attack}
\caption{CIFAR-10 non-IID development-holdout accuracy over 500 ring rounds under the Hidden attack. Lines show unsmoothed means across 100 retained client models and three seeds; bands show sample SD across the three seed means. The vertical dotted line marks attack onset at displayed round 21; the inset enlarges rounds 401--500. BASIL and FedSharp clean controls use matched direct-transfer training and overlap. BASIL with ACDS and FedSharp+ are shown under joint attack and noise only.}
\label{fig:cifar10-non-iid-hidden-500r}
\end{figure}
```

For a PNG fallback, replace only the `\studyplot` line by the following form, using that figure's actual stem:

```latex
\includegraphics[width=0.93\linewidth,height=0.53\textheight,keepaspectratio]{Images/mnist_iid_gaussian_accuracy.png}
```

Do not paste the MNIST example filename into every figure. Keep each matching caption/label. Do not reference Desktop folder names in Overleaf; the images are already in `Images/`.


Each caption must say: development-holdout accuracy; unsmoothed 100-client means and three-seed sample-SD bands; 500 rounds; attack onset displayed round 21; inset rounds 401-500. State the new clean controls use identical direct-transfer training and overlap. MNIST includes the same BASIL noise-only control in both attack-specific panels. Non-IID panels include BASIL+ACDS and FedSharp+ under joint impairment only. Do not call a sample-SD band a confidence interval. The main axes now span 0-100% for both datasets, including CIFAR-10.

Leave unrelated system diagrams and earlier literature figures in place. If changing the method diagram to explain FedSharp+, use the code-faithful stages above; any mathematical claims added inside it follow the same approval rule.

## 10. Deliverables I want from Claude

1. Confirm which commit/files were inspected; distinguish current code, frozen experimental sources and this packet's local analysis.
2. Give an evidence-based audit of outdated/contradictory claims before rewriting. Specifically address matched clean controls, variable q, 500-round scope, both attacks, MCC rather than F1, missing comparisons, the extension's MNIST tradeoff, and the commented Algorithm 2.
3. Provide replacement LaTeX for Abstract, relevant introduction/contributions, Experimental Design, all four result sections and their tables, mechanism/client diagnostics, Discussion, Limitations, Conclusions and Appendix B study boundary. Keep equations intact until approvals are granted.
4. Provide a clear FedSharp+ narrative and the candidate Algorithm 2, with before/after mathematical proposals isolated. Explain the method without inflated claims or implementation jargon in the thesis prose.
5. Provide the exact eight figure placements and captions using existing `Images/` filenames. Do not ask me to infer placement from a generic instruction such as "put figures in Results."
6. Provide a change log with source labels/anchors and original line locations, plus an unresolved-items list. Preserve bibliography keys; identify missing support instead of inventing a citation. Inspect source labels/references and compile if the class/assets/toolchain are actually available; otherwise state that compilation was not performed.
7. If you can edit the supplied source, return a revised `.tex` and concise diff/change map while preserving the original. If you only have pasted text, provide complete replacement blocks and exact anchors; do not pretend to have directly updated Overleaf.

Use precise, restrained master's-level prose: explain mechanisms in connected paragraphs, define acronyms, distinguish observation from explanation, state numerical units, and acknowledge limitations without repetitive disclaimers. Do not describe every internal wrapper/module in the thesis; use code paths as audit evidence in your response. Avoid promotional wording and generic filler. Do not write that FedSharp/FedSharp+ is "best" in every environment.

## Appendix A. Every current run, with provenance and endpoints

There are exactly 150 rows below. `M100` is the 91-100 mean, `M500` is the 491-500 mean, `F500` is the final-round all-client mean. These are per-run percentages, not across-seed averages. Archive codes expand in Section 3; exact IDs permit inspection of all detailed run/config/source/metric records. `O` marks a recorded qualified execution optimization; `R` marks an explicit checkpoint recovery; ` - ` means neither flag. Flags do not identify a new algorithm or omit the run.

| ID | Exact runId | Archive | Dataset | Split | Method | Environment | Attack | Seed | M100 | M500 | F500 | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R001 | `mnist-iid-2025-basil_attack_only-3e77f7996281` | E2 | MNIST | IID | BASIL | attack only | Gaussian | 2025 | 92.265 | 94.318 | 94.235 |  -  |
| R002 | `mnist-iid-2026-basil_attack_only-96bf634251ae` | E2 | MNIST | IID | BASIL | attack only | Gaussian | 2026 | 91.897 | 94.014 | 94.098 |  -  |
| R003 | `mnist-iid-2027-basil_attack_only-2b9444d108d5` | E2 | MNIST | IID | BASIL | attack only | Gaussian | 2027 | 92.372 | 93.933 | 93.875 |  -  |
| R004 | `mnist-iid-2025-basil_attack_only-9a60c38babb4` | E2 | MNIST | IID | BASIL | attack only | Hidden | 2025 | 92.265 | 94.318 | 94.235 |  -  |
| R005 | `mnist-iid-2026-basil_attack_only-624ec192cb5b` | E2 | MNIST | IID | BASIL | attack only | Hidden | 2026 | 91.897 | 94.014 | 94.098 |  -  |
| R006 | `mnist-iid-2027-basil_attack_only-2f2a426ff45a` | E2 | MNIST | IID | BASIL | attack only | Hidden | 2027 | 92.372 | 93.933 | 93.875 |  -  |
| R007 | `mnist-iid-2025-basil-plain-clean-5dfd67d4d361` | E5 | MNIST | IID | BASIL | clean |  -  | 2025 | 97.646 | 97.552 | 97.647 |  -  |
| R008 | `mnist-iid-2026-basil-plain-clean-7bf396ec98ec` | E5 | MNIST | IID | BASIL | clean |  -  | 2026 | 96.942 | 97.680 | 97.688 |  -  |
| R009 | `mnist-iid-2027-basil-plain-clean-01bc6a998b40` | E5 | MNIST | IID | BASIL | clean |  -  | 2027 | 97.084 | 97.575 | 97.521 |  -  |
| R010 | `mnist-iid-2025-basil_joint-d3e05c638aef` | E2 | MNIST | IID | BASIL | joint | Gaussian | 2025 | 81.866 | 73.468 | 73.123 |  -  |
| R011 | `mnist-iid-2026-basil_joint-fb4883ee8528` | E2 | MNIST | IID | BASIL | joint | Gaussian | 2026 | 82.190 | 72.615 | 72.249 |  -  |
| R012 | `mnist-iid-2027-basil_joint-bf494206aac0` | E1 | MNIST | IID | BASIL | joint | Gaussian | 2027 | 83.836 | 76.463 | 76.788 |  -  |
| R013 | `mnist-iid-2025-basil_joint-82564c8ab558` | E2 | MNIST | IID | BASIL | joint | Hidden | 2025 | 81.866 | 73.214 | 72.843 |  -  |
| R014 | `mnist-iid-2026-basil_joint-d50e51afeedd` | E2 | MNIST | IID | BASIL | joint | Hidden | 2026 | 82.117 | 72.567 | 72.841 |  -  |
| R015 | `mnist-iid-2027-basil_joint-a63aaea6b6cc` | E1 | MNIST | IID | BASIL | joint | Hidden | 2027 | 83.836 | 74.341 | 73.593 |  -  |
| R016 | `mnist-iid-2025-basil_noise_only-45f64b3d6f5a` | E1 | MNIST | IID | BASIL | noise only |  -  | 2025 | 83.158 | 73.515 | 73.623 |  -  |
| R017 | `mnist-iid-2026-basil_noise_only-b51a75104faf` | E1 | MNIST | IID | BASIL | noise only |  -  | 2026 | 83.202 | 73.484 | 73.159 |  -  |
| R018 | `mnist-iid-2027-basil_noise_only-1d74d8b314ef` | E1 | MNIST | IID | BASIL | noise only |  -  | 2027 | 83.694 | 73.936 | 74.066 |  -  |
| R019 | `mnist-iid-2025-sharp_attack_only-fe7848005398` | E2 | MNIST | IID | FedSharp | attack only | Gaussian | 2025 | 91.108 | 92.431 | 92.498 |  -  |
| R020 | `mnist-iid-2026-sharp_attack_only-3244fea34a17` | E2 | MNIST | IID | FedSharp | attack only | Gaussian | 2026 | 90.833 | 92.341 | 92.336 |  -  |
| R021 | `mnist-iid-2027-sharp_attack_only-49ec08c28623` | E1 | MNIST | IID | FedSharp | attack only | Gaussian | 2027 | 91.305 | 93.027 | 93.097 | O |
| R022 | `mnist-iid-2025-sharp_attack_only-6730e58d13c6` | E1 | MNIST | IID | FedSharp | attack only | Hidden | 2025 | 90.815 | 92.431 | 92.534 |  -  |
| R023 | `mnist-iid-2026-sharp_attack_only-b3011cd0e0b7` | E1 | MNIST | IID | FedSharp | attack only | Hidden | 2026 | 90.948 | 92.271 | 92.338 |  -  |
| R024 | `mnist-iid-2027-sharp_attack_only-0ae28af69dbd` | E1 | MNIST | IID | FedSharp | attack only | Hidden | 2027 | 91.246 | 92.984 | 92.934 |  -  |
| R025 | `mnist-iid-2025-fedsharp-plain-clean-190d77eb09e3` | E4 | MNIST | IID | FedSharp | clean |  -  | 2025 | 97.646 | 97.552 | 97.647 |  -  |
| R026 | `mnist-iid-2026-fedsharp-plain-clean-1f759edf2ce9` | E4 | MNIST | IID | FedSharp | clean |  -  | 2026 | 96.942 | 97.680 | 97.688 |  -  |
| R027 | `mnist-iid-2027-fedsharp-plain-clean-396921a6e0d7` | E4 | MNIST | IID | FedSharp | clean |  -  | 2027 | 97.084 | 97.575 | 97.521 |  -  |
| R028 | `mnist-iid-2025-sharp_joint-729dbb48f17a` | E2 | MNIST | IID | FedSharp | joint | Gaussian | 2025 | 88.425 | 87.102 | 86.921 |  -  |
| R029 | `mnist-iid-2026-sharp_joint-dd4aa4dd8e0b` | E2 | MNIST | IID | FedSharp | joint | Gaussian | 2026 | 89.153 | 87.980 | 87.979 |  -  |
| R030 | `mnist-iid-2027-sharp_joint-952fd178a24b` | E2 | MNIST | IID | FedSharp | joint | Gaussian | 2027 | 88.783 | 88.372 | 88.326 |  -  |
| R031 | `mnist-iid-2025-sharp_joint-7067629b3946` | E2 | MNIST | IID | FedSharp | joint | Hidden | 2025 | 88.425 | 86.879 | 86.899 |  -  |
| R032 | `mnist-iid-2026-sharp_joint-3bc5f082d187` | E2 | MNIST | IID | FedSharp | joint | Hidden | 2026 | 89.153 | 88.280 | 88.082 |  -  |
| R033 | `mnist-iid-2027-sharp_joint-44748a0b2f8d` | E2 | MNIST | IID | FedSharp | joint | Hidden | 2027 | 88.783 | 88.284 | 87.953 |  -  |
| R034 | `mnist-dirichlet-2025-basil_attack_only-d0ab0386c7b7` | E2 | MNIST | non-IID | BASIL | attack only | Gaussian | 2025 | 89.728 | 92.497 | 92.591 |  -  |
| R035 | `mnist-dirichlet-2026-basil_attack_only-ff9f08be368d` | E2 | MNIST | non-IID | BASIL | attack only | Gaussian | 2026 | 90.100 | 92.428 | 92.438 |  -  |
| R036 | `mnist-dirichlet-2027-basil_attack_only-b02a0a349633` | E2 | MNIST | non-IID | BASIL | attack only | Gaussian | 2027 | 89.645 | 92.727 | 92.856 |  -  |
| R037 | `mnist-dirichlet-2025-basil_attack_only-6f6ebae4cca2` | E2 | MNIST | non-IID | BASIL | attack only | Hidden | 2025 | 89.728 | 92.497 | 92.591 |  -  |
| R038 | `mnist-dirichlet-2026-basil_attack_only-00fc1bd5181c` | E2 | MNIST | non-IID | BASIL | attack only | Hidden | 2026 | 90.100 | 92.428 | 92.438 |  -  |
| R039 | `mnist-dirichlet-2027-basil_attack_only-efc31974e42d` | E2 | MNIST | non-IID | BASIL | attack only | Hidden | 2027 | 89.645 | 92.683 | 92.758 |  -  |
| R040 | `mnist-dirichlet-2025-basil-plain-clean-e688c03a3c0c` | E5 | MNIST | non-IID | BASIL | clean |  -  | 2025 | 97.772 | 97.964 | 98.076 |  -  |
| R041 | `mnist-dirichlet-2026-basil-plain-clean-f8aa0924c255` | E5 | MNIST | non-IID | BASIL | clean |  -  | 2026 | 96.560 | 97.314 | 97.455 |  -  |
| R042 | `mnist-dirichlet-2027-basil-plain-clean-575f068884d8` | E5 | MNIST | non-IID | BASIL | clean |  -  | 2027 | 97.076 | 97.604 | 97.528 |  -  |
| R043 | `mnist-dirichlet-2025-basil_joint-f2d847ebc796` | E2 | MNIST | non-IID | BASIL | joint | Gaussian | 2025 | 78.835 | 69.642 | 70.014 |  -  |
| R044 | `mnist-dirichlet-2026-basil_joint-1c7d48b14b8e` | E2 | MNIST | non-IID | BASIL | joint | Gaussian | 2026 | 78.275 | 69.038 | 68.737 |  -  |
| R045 | `mnist-dirichlet-2027-basil_joint-38e746f2d805` | E2 | MNIST | non-IID | BASIL | joint | Gaussian | 2027 | 80.286 | 72.350 | 72.526 |  -  |
| R046 | `mnist-dirichlet-2025-basil_joint-249d7c7aa4cd` | E1 | MNIST | non-IID | BASIL | joint | Hidden | 2025 | 78.835 | 69.231 | 69.104 |  -  |
| R047 | `mnist-dirichlet-2026-basil_joint-acdd0828d04e` | E1 | MNIST | non-IID | BASIL | joint | Hidden | 2026 | 78.275 | 70.377 | 70.999 |  -  |
| R048 | `mnist-dirichlet-2027-basil_joint-9e00afbd8071` | E1 | MNIST | non-IID | BASIL | joint | Hidden | 2027 | 80.286 | 68.351 | 68.900 |  -  |
| R049 | `mnist-dirichlet-2025-basil_noise_only-78103a320365` | E2 | MNIST | non-IID | BASIL | noise only |  -  | 2025 | 78.893 | 71.013 | 70.116 |  -  |
| R050 | `mnist-dirichlet-2026-basil_noise_only-59bb2ee15a4e` | E2 | MNIST | non-IID | BASIL | noise only |  -  | 2026 | 78.696 | 70.803 | 71.170 |  -  |
| R051 | `mnist-dirichlet-2027-basil_noise_only-70cac9d5626b` | E1 | MNIST | non-IID | BASIL | noise only |  -  | 2027 | 79.701 | 71.992 | 71.506 |  -  |
| R052 | `mnist-dirichlet-2025-basil_acds-0e30e6423b50` | E2 | MNIST | non-IID | BASIL + ACDS | joint | Gaussian | 2025 | 82.012 | 76.030 | 75.461 |  -  |
| R053 | `mnist-dirichlet-2026-basil_acds-c6eacbf07ac0` | E2 | MNIST | non-IID | BASIL + ACDS | joint | Gaussian | 2026 | 82.947 | 72.916 | 73.241 |  -  |
| R054 | `mnist-dirichlet-2027-basil_acds-a178c78865ce` | E2 | MNIST | non-IID | BASIL + ACDS | joint | Gaussian | 2027 | 81.804 | 75.167 | 75.331 |  -  |
| R055 | `mnist-dirichlet-2025-basil_acds-1d6da256baf5` | E2 | MNIST | non-IID | BASIL + ACDS | joint | Hidden | 2025 | 82.012 | 74.256 | 73.953 |  -  |
| R056 | `mnist-dirichlet-2026-basil_acds-cc61de1048a5` | E2 | MNIST | non-IID | BASIL + ACDS | joint | Hidden | 2026 | 82.947 | 75.462 | 74.809 |  -  |
| R057 | `mnist-dirichlet-2027-basil_acds-bb52572cbacc` | E2 | MNIST | non-IID | BASIL + ACDS | joint | Hidden | 2027 | 81.726 | 74.386 | 74.659 |  -  |
| R058 | `mnist-dirichlet-2025-sharp_attack_only-ef8471df62ab` | E1 | MNIST | non-IID | FedSharp | attack only | Gaussian | 2025 | 86.771 | 90.145 | 90.237 |  -  |
| R059 | `mnist-dirichlet-2026-sharp_attack_only-5ab74c6469c3` | E1 | MNIST | non-IID | FedSharp | attack only | Gaussian | 2026 | 86.756 | 89.714 | 89.610 |  -  |
| R060 | `mnist-dirichlet-2027-sharp_attack_only-1c7253dd9fda` | E1 | MNIST | non-IID | FedSharp | attack only | Gaussian | 2027 | 87.001 | 90.369 | 90.393 |  -  |
| R061 | `mnist-dirichlet-2025-sharp_attack_only-b43327cb1792` | E2 | MNIST | non-IID | FedSharp | attack only | Hidden | 2025 | 86.771 | 90.092 | 90.361 |  -  |
| R062 | `mnist-dirichlet-2026-sharp_attack_only-1b21926f396c` | E2 | MNIST | non-IID | FedSharp | attack only | Hidden | 2026 | 86.756 | 89.850 | 89.564 |  -  |
| R063 | `mnist-dirichlet-2027-sharp_attack_only-befb3ee45aee` | E1 | MNIST | non-IID | FedSharp | attack only | Hidden | 2027 | 87.001 | 90.200 | 90.302 | O |
| R064 | `mnist-dirichlet-2025-fedsharp-plain-clean-54e5acdb21af` | E4 | MNIST | non-IID | FedSharp | clean |  -  | 2025 | 97.772 | 97.964 | 98.076 |  -  |
| R065 | `mnist-dirichlet-2026-fedsharp-plain-clean-110102f4ecd2` | E4 | MNIST | non-IID | FedSharp | clean |  -  | 2026 | 96.560 | 97.314 | 97.455 |  -  |
| R066 | `mnist-dirichlet-2027-fedsharp-plain-clean-311670834800` | E4 | MNIST | non-IID | FedSharp | clean |  -  | 2027 | 97.076 | 97.604 | 97.528 |  -  |
| R067 | `mnist-dirichlet-2025-sharp_joint-437f943cc08c` | E2 | MNIST | non-IID | FedSharp | joint | Gaussian | 2025 | 82.990 | 83.547 | 84.133 |  -  |
| R068 | `mnist-dirichlet-2026-sharp_joint-6651e115c8aa` | E2 | MNIST | non-IID | FedSharp | joint | Gaussian | 2026 | 84.483 | 85.588 | 85.074 |  -  |
| R069 | `mnist-dirichlet-2027-sharp_joint-3c9fc9feb477` | E2 | MNIST | non-IID | FedSharp | joint | Gaussian | 2027 | 84.423 | 85.571 | 85.907 |  -  |
| R070 | `mnist-dirichlet-2025-sharp_joint-905dcf7b651b` | E2 | MNIST | non-IID | FedSharp | joint | Hidden | 2025 | 82.990 | 84.153 | 84.713 |  -  |
| R071 | `mnist-dirichlet-2026-sharp_joint-16887a33d4e5` | E2 | MNIST | non-IID | FedSharp | joint | Hidden | 2026 | 84.483 | 85.023 | 85.004 |  -  |
| R072 | `mnist-dirichlet-2027-sharp_joint-625879bcafe5` | E2 | MNIST | non-IID | FedSharp | joint | Hidden | 2027 | 84.423 | 85.349 | 85.190 |  -  |
| R073 | `mnist-dirichlet-2025-sharp_cart-c2ebde895be4` | E2 | MNIST | non-IID | FedSharp+ | joint | Gaussian | 2025 | 84.356 | 83.888 | 83.896 |  -  |
| R074 | `mnist-dirichlet-2026-sharp_cart-463f588e731a` | E2 | MNIST | non-IID | FedSharp+ | joint | Gaussian | 2026 | 84.406 | 84.320 | 84.324 |  -  |
| R075 | `mnist-dirichlet-2027-sharp_cart-e7978c9fd75f` | E1 | MNIST | non-IID | FedSharp+ | joint | Gaussian | 2027 | 85.225 | 84.585 | 84.538 | O |
| R076 | `mnist-dirichlet-2025-sharp_cart-46236275f264` | E2 | MNIST | non-IID | FedSharp+ | joint | Hidden | 2025 | 84.356 | 83.888 | 83.896 |  -  |
| R077 | `mnist-dirichlet-2026-sharp_cart-3956705aa057` | E2 | MNIST | non-IID | FedSharp+ | joint | Hidden | 2026 | 84.406 | 84.324 | 84.328 |  -  |
| R078 | `mnist-dirichlet-2027-sharp_cart-d6b3a68e8db8` | E1 | MNIST | non-IID | FedSharp+ | joint | Hidden | 2027 | 85.152 | 84.677 | 84.646 | O |
| R079 | `cifar10-iid-2025-basil_attack_only-5e423a7f97fb` | E2 | CIFAR-10 | IID | BASIL | attack only | Gaussian | 2025 | 32.568 | 42.292 | 42.269 |  -  |
| R080 | `cifar10-iid-2026-basil_attack_only-9fcf07708c6f` | E2 | CIFAR-10 | IID | BASIL | attack only | Gaussian | 2026 | 31.818 | 39.917 | 39.997 |  -  |
| R081 | `cifar10-iid-2027-basil_attack_only-dba625bbc1f8` | E2 | CIFAR-10 | IID | BASIL | attack only | Gaussian | 2027 | 32.774 | 42.256 | 42.284 |  -  |
| R082 | `cifar10-iid-2025-basil_attack_only-9e366b4082f9` | E2 | CIFAR-10 | IID | BASIL | attack only | Hidden | 2025 | 32.568 | 42.292 | 42.269 |  -  |
| R083 | `cifar10-iid-2026-basil_attack_only-949177cd5bd9` | E2 | CIFAR-10 | IID | BASIL | attack only | Hidden | 2026 | 31.818 | 39.917 | 39.997 |  -  |
| R084 | `cifar10-iid-2027-basil_attack_only-610427ef63d2` | E2 | CIFAR-10 | IID | BASIL | attack only | Hidden | 2027 | 32.774 | 42.256 | 42.284 |  -  |
| R085 | `cifar10-iid-2025-basil-plain-clean-3536df6132de` | E5 | CIFAR-10 | IID | BASIL | clean |  -  | 2025 | 62.048 | 61.835 | 61.812 |  -  |
| R086 | `cifar10-iid-2026-basil-plain-clean-e89a1ca4919a` | E5 | CIFAR-10 | IID | BASIL | clean |  -  | 2026 | 64.002 | 61.393 | 61.297 |  -  |
| R087 | `cifar10-iid-2027-basil-plain-clean-d0872a506e5f` | E5 | CIFAR-10 | IID | BASIL | clean |  -  | 2027 | 62.687 | 62.035 | 61.919 |  -  |
| R088 | `cifar10-iid-2025-basil_joint-bb2ebc388b55` | E2 | CIFAR-10 | IID | BASIL | joint | Gaussian | 2025 | 30.273 | 15.597 | 15.234 |  -  |
| R089 | `cifar10-iid-2026-basil_joint-06a59373b6fa` | E2 | CIFAR-10 | IID | BASIL | joint | Gaussian | 2026 | 30.214 | 12.952 | 12.816 |  -  |
| R090 | `cifar10-iid-2027-basil_joint-87d279c4c54d` | E2 | CIFAR-10 | IID | BASIL | joint | Gaussian | 2027 | 29.243 | 19.332 | 18.582 |  -  |
| R091 | `cifar10-iid-2025-basil_joint-ed43ae2b6883` | E2 | CIFAR-10 | IID | BASIL | joint | Hidden | 2025 | 30.273 | 17.475 | 17.720 |  -  |
| R092 | `cifar10-iid-2026-basil_joint-586f983a5cc1` | E2 | CIFAR-10 | IID | BASIL | joint | Hidden | 2026 | 30.214 | 12.952 | 12.816 |  -  |
| R093 | `cifar10-iid-2027-basil_joint-37d30d3ffa6e` | E1 | CIFAR-10 | IID | BASIL | joint | Hidden | 2027 | 29.243 | 18.176 | 17.211 |  -  |
| R094 | `cifar10-iid-2025-sharp_attack_only-0b40043fbf1f` | E2 | CIFAR-10 | IID | FedSharp | attack only | Gaussian | 2025 | 30.082 | 35.081 | 34.984 |  -  |
| R095 | `cifar10-iid-2026-sharp_attack_only-13cf8b213724` | E2 | CIFAR-10 | IID | FedSharp | attack only | Gaussian | 2026 | 30.298 | 35.166 | 35.213 |  -  |
| R096 | `cifar10-iid-2027-sharp_attack_only-86a1177a4432` | E1 | CIFAR-10 | IID | FedSharp | attack only | Gaussian | 2027 | 28.614 | 34.317 | 34.350 | O |
| R097 | `cifar10-iid-2025-sharp_attack_only-a524f2f898b2` | E2 | CIFAR-10 | IID | FedSharp | attack only | Hidden | 2025 | 30.082 | 35.081 | 34.984 |  -  |
| R098 | `cifar10-iid-2026-sharp_attack_only-e6c676753413` | E2 | CIFAR-10 | IID | FedSharp | attack only | Hidden | 2026 | 30.298 | 35.166 | 35.213 |  -  |
| R099 | `cifar10-iid-2027-sharp_attack_only-123694f80d87` | E1 | CIFAR-10 | IID | FedSharp | attack only | Hidden | 2027 | 28.614 | 34.317 | 34.350 | O |
| R100 | `cifar10-iid-2025-fedsharp-plain-clean-ac9d6a1c6c7f` | E4 | CIFAR-10 | IID | FedSharp | clean |  -  | 2025 | 62.048 | 61.835 | 61.812 |  -  |
| R101 | `cifar10-iid-2026-fedsharp-plain-clean-9b6daab5739d` | E4 | CIFAR-10 | IID | FedSharp | clean |  -  | 2026 | 64.002 | 61.393 | 61.297 |  -  |
| R102 | `cifar10-iid-2027-fedsharp-plain-clean-5a2aa38b9d73` | E4 | CIFAR-10 | IID | FedSharp | clean |  -  | 2027 | 62.687 | 62.035 | 61.919 |  -  |
| R103 | `cifar10-iid-2025-sharp_joint-e657704d2975` | E2 | CIFAR-10 | IID | FedSharp | joint | Gaussian | 2025 | 35.895 | 43.594 | 43.609 |  -  |
| R104 | `cifar10-iid-2026-sharp_joint-eea5e642f1f7` | E2 | CIFAR-10 | IID | FedSharp | joint | Gaussian | 2026 | 36.315 | 44.933 | 44.743 |  -  |
| R105 | `cifar10-iid-2027-sharp_joint-30fb1232951b` | E2 | CIFAR-10 | IID | FedSharp | joint | Gaussian | 2027 | 36.735 | 45.744 | 46.000 |  -  |
| R106 | `cifar10-iid-2025-sharp_joint-83314cb1974f` | E2 | CIFAR-10 | IID | FedSharp | joint | Hidden | 2025 | 35.895 | 43.594 | 43.609 |  -  |
| R107 | `cifar10-iid-2026-sharp_joint-fc105cb1d157` | E2 | CIFAR-10 | IID | FedSharp | joint | Hidden | 2026 | 36.315 | 44.933 | 44.743 |  -  |
| R108 | `cifar10-iid-2027-sharp_joint-f02c4443e0a3` | E2 | CIFAR-10 | IID | FedSharp | joint | Hidden | 2027 | 36.735 | 45.744 | 46.000 |  -  |
| R109 | `cifar10-dirichlet-2025-basil_attack_only-cdc1922edb79` | E2 | CIFAR-10 | non-IID | BASIL | attack only | Gaussian | 2025 | 16.720 | 24.042 | 23.750 |  -  |
| R110 | `cifar10-dirichlet-2026-basil_attack_only-add6f052f68f` | E2 | CIFAR-10 | non-IID | BASIL | attack only | Gaussian | 2026 | 17.196 | 24.428 | 24.881 |  -  |
| R111 | `cifar10-dirichlet-2027-basil_attack_only-8c20383d2eab` | E2 | CIFAR-10 | non-IID | BASIL | attack only | Gaussian | 2027 | 17.068 | 25.966 | 25.973 |  -  |
| R112 | `cifar10-dirichlet-2025-basil_attack_only-825d8f059711` | E2 | CIFAR-10 | non-IID | BASIL | attack only | Hidden | 2025 | 16.242 | 24.317 | 24.816 |  -  |
| R113 | `cifar10-dirichlet-2026-basil_attack_only-7370a8d9ae93` | E2 | CIFAR-10 | non-IID | BASIL | attack only | Hidden | 2026 | 17.094 | 24.617 | 24.718 |  -  |
| R114 | `cifar10-dirichlet-2027-basil_attack_only-1402438ab62e` | E2 | CIFAR-10 | non-IID | BASIL | attack only | Hidden | 2027 | 17.087 | 26.141 | 26.191 |  -  |
| R115 | `cifar10-dirichlet-2025-basil-plain-clean-cdb6a973c27d` | E5 | CIFAR-10 | non-IID | BASIL | clean |  -  | 2025 | 55.806 | 59.419 | 59.390 |  -  |
| R116 | `cifar10-dirichlet-2026-basil-plain-clean-8344b051aa27` | E5 | CIFAR-10 | non-IID | BASIL | clean |  -  | 2026 | 56.025 | 60.301 | 60.178 |  -  |
| R117 | `cifar10-dirichlet-2027-basil-plain-clean-c4af1a254791` | E5 | CIFAR-10 | non-IID | BASIL | clean |  -  | 2027 | 56.690 | 60.220 | 60.240 |  -  |
| R118 | `cifar10-dirichlet-2025-basil_joint-ed7459e13a9c` | E2 | CIFAR-10 | non-IID | BASIL | joint | Gaussian | 2025 | 20.659 | 10.004 | 10.016 |  -  |
| R119 | `cifar10-dirichlet-2026-basil_joint-5abca61bb399` | E2 | CIFAR-10 | non-IID | BASIL | joint | Gaussian | 2026 | 20.816 | 14.440 | 14.599 |  -  |
| R120 | `cifar10-dirichlet-2027-basil_joint-9f70fb7d3268` | E2 | CIFAR-10 | non-IID | BASIL | joint | Gaussian | 2027 | 21.246 | 10.021 | 10.053 |  -  |
| R121 | `cifar10-dirichlet-2025-basil_joint-494158af6ffc` | E2 | CIFAR-10 | non-IID | BASIL | joint | Hidden | 2025 | 20.517 | 11.454 | 12.249 |  -  |
| R122 | `cifar10-dirichlet-2026-basil_joint-4933bcaf248c` | E2 | CIFAR-10 | non-IID | BASIL | joint | Hidden | 2026 | 20.890 | 16.254 | 16.026 |  -  |
| R123 | `cifar10-dirichlet-2027-basil_joint-6698383ba3dc` | E2 | CIFAR-10 | non-IID | BASIL | joint | Hidden | 2027 | 20.675 | 16.603 | 16.777 |  -  |
| R124 | `cifar10-dirichlet-2025-basil_acds-c201b2e8ec9b` | E2 | CIFAR-10 | non-IID | BASIL + ACDS | joint | Gaussian | 2025 | 29.675 | 20.125 | 19.829 |  -  |
| R125 | `cifar10-dirichlet-2026-basil_acds-41514f3d489e` | E2 | CIFAR-10 | non-IID | BASIL + ACDS | joint | Gaussian | 2026 | 30.214 | 14.246 | 13.332 |  -  |
| R126 | `cifar10-dirichlet-2027-basil_acds-eba0e98d5416` | E2 | CIFAR-10 | non-IID | BASIL + ACDS | joint | Gaussian | 2027 | 29.389 | 22.873 | 22.103 |  -  |
| R127 | `cifar10-dirichlet-2025-basil_acds-03863eb17a93` | E1 | CIFAR-10 | non-IID | BASIL + ACDS | joint | Hidden | 2025 | 29.675 | 18.717 | 18.452 | R |
| R128 | `cifar10-dirichlet-2026-basil_acds-fba99c36cab0` | E1 | CIFAR-10 | non-IID | BASIL + ACDS | joint | Hidden | 2026 | 30.214 | 14.630 | 14.583 |  -  |
| R129 | `cifar10-dirichlet-2027-basil_acds-a194fa471509` | E1 | CIFAR-10 | non-IID | BASIL + ACDS | joint | Hidden | 2027 | 29.389 | 22.079 | 21.801 |  -  |
| R130 | `cifar10-dirichlet-2025-sharp_attack_only-9bb2726dfeb3` | E2 | CIFAR-10 | non-IID | FedSharp | attack only | Gaussian | 2025 | 13.757 | 19.148 | 19.022 |  -  |
| R131 | `cifar10-dirichlet-2026-sharp_attack_only-bb6d96ebb52e` | E2 | CIFAR-10 | non-IID | FedSharp | attack only | Gaussian | 2026 | 11.685 | 17.982 | 18.071 |  -  |
| R132 | `cifar10-dirichlet-2027-sharp_attack_only-4cb46c167778` | E1 | CIFAR-10 | non-IID | FedSharp | attack only | Gaussian | 2027 | 11.719 | 18.450 | 18.486 | O |
| R133 | `cifar10-dirichlet-2025-sharp_attack_only-3823a3f93373` | E2 | CIFAR-10 | non-IID | FedSharp | attack only | Hidden | 2025 | 14.052 | 18.902 | 19.258 |  -  |
| R134 | `cifar10-dirichlet-2026-sharp_attack_only-17360344e1da` | E2 | CIFAR-10 | non-IID | FedSharp | attack only | Hidden | 2026 | 11.431 | 17.678 | 17.790 |  -  |
| R135 | `cifar10-dirichlet-2027-sharp_attack_only-47d64264fad7` | E1 | CIFAR-10 | non-IID | FedSharp | attack only | Hidden | 2027 | 11.862 | 18.371 | 18.655 | O |
| R136 | `cifar10-dirichlet-2025-fedsharp-plain-clean-94399710cb00` | E4 | CIFAR-10 | non-IID | FedSharp | clean |  -  | 2025 | 55.806 | 59.419 | 59.390 |  -  |
| R137 | `cifar10-dirichlet-2026-fedsharp-plain-clean-fedf208b30ac` | E4 | CIFAR-10 | non-IID | FedSharp | clean |  -  | 2026 | 56.025 | 60.301 | 60.178 |  -  |
| R138 | `cifar10-dirichlet-2027-fedsharp-plain-clean-5c7ee74aa9d5` | E4 | CIFAR-10 | non-IID | FedSharp | clean |  -  | 2027 | 56.690 | 60.220 | 60.240 |  -  |
| R139 | `cifar10-dirichlet-2025-sharp_joint-62246b854ecc` | E1 | CIFAR-10 | non-IID | FedSharp | joint | Gaussian | 2025 | 18.247 | 24.939 | 25.106 |  -  |
| R140 | `cifar10-dirichlet-2026-sharp_joint-01111976c0cc` | E1 | CIFAR-10 | non-IID | FedSharp | joint | Gaussian | 2026 | 18.625 | 25.875 | 26.117 |  -  |
| R141 | `cifar10-dirichlet-2027-sharp_joint-ca6fc78b497f` | E1 | CIFAR-10 | non-IID | FedSharp | joint | Gaussian | 2027 | 17.901 | 23.227 | 23.056 |  -  |
| R142 | `cifar10-dirichlet-2025-sharp_joint-ef8b6adcfc43` | E2 | CIFAR-10 | non-IID | FedSharp | joint | Hidden | 2025 | 18.505 | 14.002 | 14.433 |  -  |
| R143 | `cifar10-dirichlet-2026-sharp_joint-b78bd7d90771` | E2 | CIFAR-10 | non-IID | FedSharp | joint | Hidden | 2026 | 19.130 | 15.337 | 15.178 |  -  |
| R144 | `cifar10-dirichlet-2027-sharp_joint-6c403f2aec67` | E2 | CIFAR-10 | non-IID | FedSharp | joint | Hidden | 2027 | 19.511 | 14.539 | 14.134 |  -  |
| R145 | `cifar10-dirichlet-2025-sharp_cart-c7e2293c56c6` | E2 | CIFAR-10 | non-IID | FedSharp+ | joint | Gaussian | 2025 | 27.506 | 29.325 | 29.352 |  -  |
| R146 | `cifar10-dirichlet-2026-sharp_cart-0e7065a49271` | E2 | CIFAR-10 | non-IID | FedSharp+ | joint | Gaussian | 2026 | 27.254 | 28.335 | 28.359 |  -  |
| R147 | `cifar10-dirichlet-2027-sharp_cart-190f7e67755e` | E1 | CIFAR-10 | non-IID | FedSharp+ | joint | Gaussian | 2027 | 28.479 | 29.288 | 29.293 | O |
| R148 | `cifar10-dirichlet-2025-sharp_cart-6aea2e473335` | E2 | CIFAR-10 | non-IID | FedSharp+ | joint | Hidden | 2025 | 27.506 | 29.325 | 29.352 |  -  |
| R149 | `cifar10-dirichlet-2026-sharp_cart-ac9462314989` | E2 | CIFAR-10 | non-IID | FedSharp+ | joint | Hidden | 2026 | 27.254 | 28.335 | 28.359 |  -  |
| R150 | `cifar10-dirichlet-2027-sharp_cart-d6fd385aa051` | E1 | CIFAR-10 | non-IID | FedSharp+ | joint | Hidden | 2027 | 28.479 | 29.288 | 29.293 | O |

### Per-run final diagnostics and postattack mechanisms

Join to the preceding table by ID. Diagnostic accuracies are percentages at round 500; q, λ, own-route and gate/commit statistics cover rounds 21-500. Zero-commit percentages use active checks as denominator; " - " means no such checks, not missing accuracy data. `Cal. draws` is the sum of the recorded `calibrationDraws` mechanism field in that window, not a full count of all forward evaluations or every other calibration field. Clean all-client and honest means coincide because there are no active attackers.

| ID | Honest final | Worst client | Client P10 | Worst class | μ | Window | q | λ | Own % | Gate % | Zero % active | Cal. draws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R001 | 94.212 | 93.700 | 93.900 | 88.340 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R002 | 94.087 | 93.500 | 93.700 | 89.290 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R003 | 93.875 | 93.200 | 93.490 | 89.330 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R004 | 94.212 | 93.700 | 93.900 | 88.340 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R005 | 94.087 | 93.500 | 93.700 | 89.290 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R006 | 93.875 | 93.200 | 93.490 | 89.330 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R007 | 97.647 | 97.500 | 97.600 | 93.560 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R008 | 97.688 | 97.600 | 97.600 | 94.320 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R009 | 97.521 | 97.500 | 97.500 | 94.430 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R010 | 72.884 | 69.200 | 70.090 | 61.070 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R011 | 72.275 | 69.700 | 70.900 | 54.570 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R012 | 76.836 | 73.400 | 74.590 | 59.920 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R013 | 72.916 | 68.800 | 70.000 | 48.170 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R014 | 72.842 | 70.300 | 71.500 | 57.040 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R015 | 73.500 | 71.500 | 72.090 | 58.940 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R016 | 73.623 | 70.700 | 72.400 | 58.840 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R017 | 73.159 | 70.500 | 70.990 | 60.470 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R018 | 74.066 | 72.000 | 72.400 | 60.040 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R019 | 92.516 | 92.000 | 92.100 | 87.040 | 1 | 66 | 1.197 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R020 | 92.342 | 91.500 | 91.800 | 85.230 | 1 | 66 | 1.190 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R021 | 93.063 | 92.400 | 92.780 | 88.690 | 1 | 66 | 1.167 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R022 | 92.554 | 92.000 | 92.200 | 86.360 | 1 | 66 | 1.093 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R023 | 92.303 | 91.600 | 92.090 | 86.390 | 1 | 66 | 1.123 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R024 | 92.915 | 92.200 | 92.400 | 88.520 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R025 | 97.647 | 97.500 | 97.600 | 93.560 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R026 | 97.688 | 97.600 | 97.600 | 94.320 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R027 | 97.521 | 97.500 | 97.500 | 94.430 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R028 | 86.937 | 84.600 | 85.590 | 78.110 | 1 | 66 | 1.938 | 0.014220 | 0.000 | 0.000 |  -  | 0 |
| R029 | 87.972 | 85.900 | 86.400 | 82.540 | 1 | 66 | 1.970 | 0.014202 | 0.000 | 0.000 |  -  | 0 |
| R030 | 88.428 | 85.500 | 86.490 | 81.630 | 1 | 66 | 1.971 | 0.014198 | 0.000 | 0.000 |  -  | 0 |
| R031 | 86.807 | 83.400 | 85.800 | 75.010 | 1 | 66 | 1.937 | 0.014219 | 0.000 | 0.000 |  -  | 0 |
| R032 | 88.078 | 84.300 | 85.770 | 83.510 | 1 | 66 | 1.923 | 0.014233 | 0.000 | 0.000 |  -  | 0 |
| R033 | 88.027 | 86.000 | 87.090 | 81.360 | 1 | 66 | 1.964 | 0.014204 | 0.000 | 0.000 |  -  | 0 |
| R034 | 92.522 | 91.300 | 91.590 | 86.740 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R035 | 92.382 | 91.000 | 91.800 | 86.800 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R036 | 92.797 | 91.600 | 92.100 | 86.710 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R037 | 92.522 | 91.300 | 91.590 | 86.740 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R038 | 92.382 | 91.000 | 91.800 | 86.800 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R039 | 92.766 | 91.900 | 92.390 | 87.190 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R040 | 98.076 | 97.900 | 98.000 | 93.190 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R041 | 97.455 | 97.200 | 97.300 | 94.430 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R042 | 97.528 | 97.500 | 97.500 | 93.200 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R043 | 70.121 | 65.500 | 68.790 | 52.190 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R044 | 68.775 | 65.500 | 67.000 | 43.770 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R045 | 72.591 | 68.700 | 70.180 | 59.820 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R046 | 69.546 | 7.600 | 67.300 | 52.690 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R047 | 70.903 | 66.300 | 69.700 | 54.520 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R048 | 68.788 | 64.200 | 66.090 | 52.660 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R049 | 70.116 | 66.100 | 68.100 | 53.320 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R050 | 71.170 | 65.800 | 69.690 | 44.830 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R051 | 71.506 | 68.400 | 70.080 | 57.740 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R052 | 75.434 | 74.100 | 74.400 | 61.020 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R053 | 73.176 | 70.600 | 71.700 | 57.410 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R054 | 75.321 | 73.400 | 74.100 | 56.100 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R055 | 73.636 | 6.900 | 73.600 | 55.510 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R056 | 74.401 | 2.600 | 74.290 | 57.820 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R057 | 74.710 | 72.600 | 73.800 | 58.990 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R058 | 90.197 | 88.100 | 89.290 | 83.480 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R059 | 89.533 | 86.600 | 88.080 | 81.990 | 1 | 66 | 1.001 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R060 | 90.394 | 88.200 | 89.290 | 83.750 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R061 | 90.245 | 87.200 | 89.090 | 82.770 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R062 | 89.522 | 87.300 | 88.290 | 78.970 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R063 | 90.324 | 88.800 | 89.290 | 83.170 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R064 | 98.076 | 97.900 | 98.000 | 93.190 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R065 | 97.455 | 97.200 | 97.300 | 94.430 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R066 | 97.528 | 97.500 | 97.500 | 93.200 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R067 | 84.124 | 79.500 | 81.300 | 74.230 | 1 | 66 | 2.096 | 0.014168 | 0.000 | 0.000 |  -  | 0 |
| R068 | 84.939 | 81.300 | 82.890 | 74.620 | 1 | 66 | 2.140 | 0.014152 | 0.000 | 0.000 |  -  | 0 |
| R069 | 85.858 | 82.000 | 83.590 | 77.810 | 1 | 66 | 2.159 | 0.014129 | 0.000 | 0.000 |  -  | 0 |
| R070 | 84.794 | 81.100 | 83.100 | 75.030 | 1 | 66 | 2.078 | 0.014178 | 0.000 | 0.000 |  -  | 0 |
| R071 | 84.672 | 80.400 | 81.790 | 71.660 | 1 | 66 | 2.153 | 0.014142 | 0.000 | 0.000 |  -  | 0 |
| R072 | 85.164 | 78.500 | 82.800 | 73.350 | 1 | 66 | 2.142 | 0.014141 | 0.000 | 0.000 |  -  | 0 |
| R073 | 83.901 | 69.800 | 80.190 | 73.300 | 1 | 66 | 1.010 | 0.000617 | 94.869 | 100.000 | 12.865 | 912000 |
| R074 | 84.333 | 76.600 | 81.400 | 73.180 | 1 | 66 | 1.007 | 0.000574 | 95.148 | 100.000 | 12.535 | 912000 |
| R075 | 84.673 | 70.200 | 81.780 | 74.340 | 1 | 66 | 1.005 | 0.000780 | 94.256 | 100.000 | 12.581 | 912000 |
| R076 | 83.901 | 69.800 | 80.190 | 73.300 | 1 | 66 | 1.010 | 0.000617 | 94.867 | 100.000 | 12.865 | 912000 |
| R077 | 84.333 | 76.600 | 81.400 | 73.170 | 1 | 66 | 1.007 | 0.000574 | 95.140 | 100.000 | 12.521 | 912000 |
| R078 | 84.767 | 70.200 | 81.800 | 74.700 | 1 | 66 | 1.006 | 0.000565 | 95.304 | 100.000 | 13.427 | 912000 |
| R079 | 42.212 | 40.000 | 41.390 | 18.530 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R080 | 39.943 | 38.600 | 39.300 | 10.280 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R081 | 42.218 | 40.800 | 41.500 | 16.480 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R082 | 42.212 | 40.000 | 41.390 | 18.530 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R083 | 39.943 | 38.600 | 39.300 | 10.280 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R084 | 42.218 | 40.800 | 41.500 | 16.480 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R085 | 61.812 | 60.300 | 61.100 | 43.390 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R086 | 61.297 | 60.100 | 60.600 | 41.990 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R087 | 61.919 | 60.300 | 61.100 | 45.580 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R088 | 15.301 | 6.900 | 9.500 | 11.270 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R089 | 12.584 | 9.300 | 10.600 | 8.010 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R090 | 18.776 | 12.400 | 16.100 | 9.790 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R091 | 17.328 | 10.200 | 13.900 | 4.900 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R092 | 12.584 | 9.300 | 10.600 | 8.010 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R093 | 16.934 | 10.500 | 13.390 | 10.600 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R094 | 34.925 | 33.500 | 34.000 | 11.940 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R095 | 35.221 | 33.900 | 34.400 | 9.220 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R096 | 34.306 | 33.000 | 33.700 | 8.490 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R097 | 34.925 | 33.500 | 34.000 | 11.940 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R098 | 35.221 | 33.900 | 34.400 | 9.220 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R099 | 34.306 | 33.000 | 33.700 | 8.490 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R100 | 61.812 | 60.300 | 61.100 | 43.390 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R101 | 61.297 | 60.100 | 60.600 | 41.990 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R102 | 61.919 | 60.300 | 61.100 | 45.580 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R103 | 43.699 | 41.100 | 42.100 | 27.010 | 1 | 66 | 9.594 | 0.011001 | 0.000 | 0.000 |  -  | 0 |
| R104 | 44.810 | 41.600 | 43.100 | 28.380 | 1 | 66 | 9.645 | 0.010991 | 0.000 | 0.000 |  -  | 0 |
| R105 | 45.955 | 42.200 | 44.780 | 30.960 | 1 | 66 | 10.013 | 0.010783 | 0.000 | 0.000 |  -  | 0 |
| R106 | 43.699 | 41.100 | 42.100 | 27.010 | 1 | 66 | 9.594 | 0.011001 | 0.000 | 0.000 |  -  | 0 |
| R107 | 44.810 | 41.600 | 43.100 | 28.380 | 1 | 66 | 9.645 | 0.010991 | 0.000 | 0.000 |  -  | 0 |
| R108 | 45.955 | 42.200 | 44.780 | 30.960 | 1 | 66 | 10.013 | 0.010783 | 0.000 | 0.000 |  -  | 0 |
| R109 | 23.460 | 14.500 | 21.000 | 5.570 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R110 | 25.057 | 17.800 | 21.200 | 13.900 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R111 | 26.028 | 17.500 | 22.790 | 6.470 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R112 | 24.490 | 15.500 | 20.600 | 12.810 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R113 | 24.757 | 18.300 | 20.780 | 11.170 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R114 | 26.215 | 18.200 | 23.200 | 5.830 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R115 | 59.390 | 56.500 | 57.500 | 41.330 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R116 | 60.178 | 55.900 | 58.590 | 42.980 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R117 | 60.240 | 56.600 | 58.800 | 47.950 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R118 | 10.009 | 9.800 | 10.000 | 6.980 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R119 | 14.645 | 9.200 | 11.090 | 9.900 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R120 | 10.061 | 9.900 | 10.000 | 5.000 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R121 | 11.952 | 8.800 | 10.000 | 4.340 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R122 | 16.227 | 10.000 | 12.890 | 9.700 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R123 | 16.915 | 10.000 | 12.670 | 7.700 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R124 | 20.030 | 14.000 | 16.800 | 9.350 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R125 | 13.172 | 9.600 | 10.000 | 8.850 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R126 | 21.822 | 12.600 | 19.680 | 12.050 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R127 | 18.264 | 9.500 | 14.570 | 10.650 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R128 | 14.384 | 9.700 | 11.390 | 6.650 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R129 | 21.712 | 16.800 | 19.970 | 11.950 | 0 | 34 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R130 | 18.779 | 13.900 | 16.290 | 8.670 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R131 | 17.982 | 10.000 | 13.790 | 9.450 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R132 | 18.425 | 13.700 | 15.100 | 9.320 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R133 | 19.060 | 14.300 | 16.870 | 9.200 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R134 | 17.673 | 10.000 | 13.690 | 7.470 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R135 | 18.627 | 14.100 | 15.690 | 9.370 | 1 | 66 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R136 | 59.390 | 56.500 | 57.500 | 41.330 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R137 | 60.178 | 55.900 | 58.590 | 42.980 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R138 | 60.240 | 56.600 | 58.800 | 47.950 | 0 | 1 | 1.000 | 0.000000 | 0.000 | 0.000 |  -  | 0 |
| R139 | 24.849 | 11.600 | 20.440 | 11.560 | 1 | 66 | 3.865 | 0.013205 | 0.000 | 0.000 |  -  | 0 |
| R140 | 25.876 | 16.300 | 20.370 | 9.340 | 1 | 66 | 3.441 | 0.013435 | 0.000 | 0.000 |  -  | 0 |
| R141 | 23.006 | 16.100 | 18.550 | 14.800 | 1 | 66 | 3.385 | 0.013466 | 0.000 | 0.000 |  -  | 0 |
| R142 | 14.649 | 9.800 | 10.930 | 9.120 | 1 | 66 | 1.025 | 0.014628 | 0.000 | 0.000 |  -  | 0 |
| R143 | 15.207 | 9.900 | 11.070 | 9.790 | 1 | 66 | 1.043 | 0.014617 | 0.000 | 0.000 |  -  | 0 |
| R144 | 14.822 | 8.600 | 10.000 | 10.800 | 1 | 66 | 1.032 | 0.014623 | 0.000 | 0.000 |  -  | 0 |
| R145 | 29.467 | 23.400 | 26.370 | 13.990 | 1 | 66 | 1.001 | 0.000071 | 99.215 | 100.000 | 30.406 | 911259 |
| R146 | 28.458 | 22.300 | 25.690 | 15.340 | 1 | 66 | 1.000 | 0.000004 | 99.921 | 100.000 | 28.171 | 911962 |
| R147 | 29.207 | 21.400 | 26.990 | 17.220 | 1 | 66 | 1.000 | 0.000018 | 99.721 | 100.000 | 28.150 | 911677 |
| R148 | 29.467 | 23.400 | 26.370 | 13.990 | 1 | 66 | 1.001 | 0.000071 | 99.215 | 100.000 | 30.406 | 911259 |
| R149 | 28.458 | 22.300 | 25.690 | 15.340 | 1 | 66 | 1.000 | 0.000004 | 99.921 | 100.000 | 28.171 | 911962 |
| R150 | 29.207 | 21.400 | 26.990 | 17.220 | 1 | 66 | 1.000 | 0.000018 | 99.721 | 100.000 | 28.150 | 911677 |

## Appendix B. Every superseded 500-round clean run  -  historical only

These 36 records are retained for provenance and explaining the control revision. **Do not pool them with current seeds, reuse them for current clean curves, or present them as additional independent confirmation.** E1/E2 entries are original defense-enabled clean controls; E3 entries are intermediate FedSharp clean controls with μ by partition. E4/E5 supply all current clean endpoints. Below "primary" means their own rounds 491-500; "final" means round 500. Old controls may be discussed explicitly as a protocol comparison, with the changed selection/μ policy stated.

| ID | Exact runId | Archive | Dataset | Split | Method | Seed | μ | Window | Primary | Final |
|---|---|---|---|---|---|---|---|---|---|---|
| H001 | `mnist-iid-2025-basil_clean-1b68da0f345d` | E2 | MNIST | IID | BASIL | 2025 | 0 | 34 | 93.937 | 93.893 |
| H002 | `mnist-iid-2026-basil_clean-74cd29604d4b` | E2 | MNIST | IID | BASIL | 2026 | 0 | 34 | 93.788 | 93.849 |
| H003 | `mnist-iid-2027-basil_clean-559cb7a7aa0c` | E2 | MNIST | IID | BASIL | 2027 | 0 | 34 | 94.192 | 94.190 |
| H004 | `mnist-iid-2025-sharp_clean-56ece7fa71f7` | E2 | MNIST | IID | FedSharp | 2025 | 1 | 66 | 92.325 | 92.283 |
| H005 | `mnist-iid-2025-fedsharp-clean-mu0-4ccbaf4f3e53` | E3 | MNIST | IID | FedSharp | 2025 | 0 | 66 | 92.323 | 92.411 |
| H006 | `mnist-iid-2026-sharp_clean-b8b86b1acb54` | E2 | MNIST | IID | FedSharp | 2026 | 1 | 66 | 92.315 | 92.391 |
| H007 | `mnist-iid-2026-fedsharp-clean-mu0-b59aa63f6bb4` | E3 | MNIST | IID | FedSharp | 2026 | 0 | 66 | 92.600 | 92.586 |
| H008 | `mnist-iid-2027-sharp_clean-f1ffc719c6d7` | E2 | MNIST | IID | FedSharp | 2027 | 1 | 66 | 92.749 | 92.716 |
| H009 | `mnist-iid-2027-fedsharp-clean-mu0-1e309a46416d` | E3 | MNIST | IID | FedSharp | 2027 | 0 | 66 | 92.789 | 92.779 |
| H010 | `mnist-dirichlet-2025-basil_clean-71a01903dc00` | E2 | MNIST | non-IID | BASIL | 2025 | 0 | 34 | 92.473 | 92.412 |
| H011 | `mnist-dirichlet-2026-basil_clean-b1a0b04ffaf7` | E2 | MNIST | non-IID | BASIL | 2026 | 0 | 34 | 92.159 | 92.250 |
| H012 | `mnist-dirichlet-2027-basil_clean-bbd297d4e4b3` | E2 | MNIST | non-IID | BASIL | 2027 | 0 | 34 | 92.282 | 92.431 |
| H013 | `mnist-dirichlet-2025-sharp_clean-9bc51ee5b268` | E2 | MNIST | non-IID | FedSharp | 2025 | 1 | 66 | 89.607 | 89.592 |
| H014 | `mnist-dirichlet-2025-fedsharp-clean-mu1-05a6d52e8d51` | E3 | MNIST | non-IID | FedSharp | 2025 | 1 | 66 | 89.607 | 89.592 |
| H015 | `mnist-dirichlet-2026-sharp_clean-f29d655db43d` | E2 | MNIST | non-IID | FedSharp | 2026 | 1 | 66 | 89.972 | 89.884 |
| H016 | `mnist-dirichlet-2026-fedsharp-clean-mu1-76b0f82052b6` | E3 | MNIST | non-IID | FedSharp | 2026 | 1 | 66 | 89.972 | 89.884 |
| H017 | `mnist-dirichlet-2027-sharp_clean-269f14ecbe38` | E2 | MNIST | non-IID | FedSharp | 2027 | 1 | 66 | 89.551 | 89.614 |
| H018 | `mnist-dirichlet-2027-fedsharp-clean-mu1-b09d164dd507` | E3 | MNIST | non-IID | FedSharp | 2027 | 1 | 66 | 89.551 | 89.614 |
| H019 | `cifar10-iid-2025-basil_clean-85a36e465378` | E1 | CIFAR-10 | IID | BASIL | 2025 | 0 | 34 | 42.750 | 42.844 |
| H020 | `cifar10-iid-2026-basil_clean-bdfd945beafa` | E1 | CIFAR-10 | IID | BASIL | 2026 | 0 | 34 | 40.061 | 40.081 |
| H021 | `cifar10-iid-2027-basil_clean-43107d082b1f` | E1 | CIFAR-10 | IID | BASIL | 2027 | 0 | 34 | 42.598 | 42.458 |
| H022 | `cifar10-iid-2025-sharp_clean-069e5c0e6515` | E2 | CIFAR-10 | IID | FedSharp | 2025 | 1 | 66 | 35.751 | 35.752 |
| H023 | `cifar10-iid-2025-fedsharp-clean-mu0-f71169373628` | E3 | CIFAR-10 | IID | FedSharp | 2025 | 0 | 66 | 35.671 | 35.782 |
| H024 | `cifar10-iid-2026-sharp_clean-75d68376fbe2` | E2 | CIFAR-10 | IID | FedSharp | 2026 | 1 | 66 | 35.194 | 35.206 |
| H025 | `cifar10-iid-2026-fedsharp-clean-mu0-5095374b1427` | E3 | CIFAR-10 | IID | FedSharp | 2026 | 0 | 66 | 34.804 | 34.951 |
| H026 | `cifar10-iid-2027-sharp_clean-451cce053559` | E2 | CIFAR-10 | IID | FedSharp | 2027 | 1 | 66 | 35.107 | 35.195 |
| H027 | `cifar10-iid-2027-fedsharp-clean-mu0-7a49bddbadf0` | E3 | CIFAR-10 | IID | FedSharp | 2027 | 0 | 66 | 36.306 | 36.432 |
| H028 | `cifar10-dirichlet-2025-basil_clean-5be969223bf2` | E2 | CIFAR-10 | non-IID | BASIL | 2025 | 0 | 34 | 25.249 | 25.131 |
| H029 | `cifar10-dirichlet-2026-basil_clean-c157910a3cee` | E2 | CIFAR-10 | non-IID | BASIL | 2026 | 0 | 34 | 24.028 | 24.049 |
| H030 | `cifar10-dirichlet-2027-basil_clean-cf3a9f945f23` | E2 | CIFAR-10 | non-IID | BASIL | 2027 | 0 | 34 | 25.361 | 25.216 |
| H031 | `cifar10-dirichlet-2025-sharp_clean-60dd37a4e8c8` | E2 | CIFAR-10 | non-IID | FedSharp | 2025 | 1 | 66 | 18.740 | 18.777 |
| H032 | `cifar10-dirichlet-2025-fedsharp-clean-mu1-2312f4bd73e5` | E3 | CIFAR-10 | non-IID | FedSharp | 2025 | 1 | 66 | 18.740 | 18.777 |
| H033 | `cifar10-dirichlet-2026-sharp_clean-5be2df4d5a92` | E2 | CIFAR-10 | non-IID | FedSharp | 2026 | 1 | 66 | 17.387 | 17.352 |
| H034 | `cifar10-dirichlet-2026-fedsharp-clean-mu1-9a23d1ad7435` | E3 | CIFAR-10 | non-IID | FedSharp | 2026 | 1 | 66 | 17.387 | 17.352 |
| H035 | `cifar10-dirichlet-2027-sharp_clean-a216dc211e92` | E2 | CIFAR-10 | non-IID | FedSharp | 2027 | 1 | 66 | 18.383 | 18.475 |
| H036 | `cifar10-dirichlet-2027-fedsharp-clean-mu1-2e6f755ca4f8` | E3 | CIFAR-10 | non-IID | FedSharp | 2027 | 1 | 66 | 18.383 | 18.475 |

## Appendix C. Source identity of the supplied thesis files

| Input | Original supplied location | Bytes | SHA256 |
|---|---|---|---|
| thesis.tex | `/mnt/c/Users/ojasw/Downloads/thesis.tex` | 164471 | `53d338e768d7e05f83371e6228e1a8fd9ffd9feee5192146756508d094efb293` |
| references.bib | `/mnt/c/Users/ojasw/Downloads/references.bib` | 31006 | `b937e4bf8cb626f4fc5adf5ff599051d54fa40e1df1ec67214eaa680afdeef59` |
| thesis.pdf | `/mnt/c/Users/ojasw/Desktop/THESIS/OSinha_MSCS_Thesis_Robust_DFL_2026.pdf` | 7065514 | `8008f6eb18ed4cd4bd941b08947fc728a5743c34573a31d6887db361a0a6b701` |

Export manifest identities, checked before summarizing:

| Archive | SHA256 of `SHA256.json` |
|---|---|
| E1 | `c63c881dc40eb103f2b9450286a8fc8c37c272e136e213c6a177908416be796a` |
| E2 | `483ce127bfc48064cea94c1af81e7d251764d8ed95338b70016aed45e45bbbfc` |
| E3 | `64331cae4ecfeb1bac2900a2f184ca660088916f2c62ff0539e5c7434581e33e` |
| E4 | `ef3a080b2ca0c48704d9e135175fe5aea6b4dc1f62bd3c20fb2bf5e3f4576644` |
| E5 | `0f86831270e1721ceb087c58b47d98dc2a32c20a4d50184ad9a2a83f4202368e` |

This packet's private analysis verified exported data and summarized it; it is not itself a new public experiment archive. Repository paths in Section 3 are sufficient for an independent recheck. Full raw 500-round metrics, mechanism records, class arrays and source hashes remain in those exports; do not estimate values by digitizing the SVGs. If any statement in this packet conflicts with verified archived records, report the conflict and resolve it from the records instead of silently changing results.
