# Review of the matched FedSharp thesis results

All 84 configurations in the hal9k export completed 100 ring rounds with 100 clients and five mini-batch optimizer steps per activation. All 356 original exported files were verified on import. Removing four duplicate plot PDFs and adding the packaging record leaves 353 checksummed publication files. The audit checked those file hashes, source-linked configurations, metric shapes and ranges, three-seed coverage for every curve, and the independent-inference receipt. Total work is 840,000 activations and 4,200,000 optimizer steps.

The saved HPC receipt covers fresh CPU inference for 8,400 final client models. Its largest client discrepancy is 0.001 (one example out of 1,000), within the recorded tolerance. This local review checks that receipt and recomputes statistics from raw arrays; the compact archive excludes checkpoints, so this review does not claim another local checkpoint inference.

## Figure assessment

Four figures contain exactly six IID and eight non-IID curves per dataset, using the professor-approved FedSharp and FedSharp+ names. All 100 rounds are plotted without smoothing. Curves average client accuracies within each run, then average across three seeds; shading is sample SD across seeds. The axis correctly reads validation accuracy. The holdout consists of original-TRAIN examples, not benchmark TEST examples.

The figures use serif typography, condition colors, distinct family line styles, staggered markers, external legends and pure vector SVG output with outlined fonts. MNIST includes a labelled late-round detail inset. CIFAR axes are consistently 0–50%; MNIST main axes are 0–100%. The dotted line marks attack onset at round 21. The four PNG previews were inspected for clipping, legend overlap, curve visibility and label readability. Clean and attack-only curves nearly coincide; markers preserve their distinction without perturbing the observations.

## Scientific interpretation

The measured outcomes are mixed. FedSharp+ improves CIFAR-10 non-IID joint-condition final mean accuracy from FedSharp’s 19.319% to 27.816%; BASIL + ACDS reaches 29.531% under that joint condition. On MNIST non-IID, FedSharp+ reaches 84.642%, compared with FedSharp’s 84.010% and BASIL + ACDS’s 82.084%. These descriptive three-seed comparisons do not establish universal superiority or statistical significance.

Clean and attack-only trajectories are identical before configured attack onset in all eight paired dataset/partition/family checks. They remain close after onset, indicating a small measured attack effect for these configurations. Some CIFAR clean curves learn weakly and the noisy FedSharp conditions exceed their clean counterparts. Noise-dependent regularization also changes across these conditions, so the curves alone do not isolate a beneficial causal effect of channel noise. The thesis must report these outcomes faithfully and distinguish this matrix from older high-clean-accuracy protocols.

ACDS changes fitting-data access by sharing 5% of fitting examples; FedSharp+ retains the full declared local-prior/class-aware pipeline. The comparison is not a single-component causal isolation.

## Recomputed endpoints

Values are percentages, mean ± sample SD across seeds 2025, 2026 and 2027. The primary endpoint averages rounds 91–100 within each seed. Final-round values are descriptive secondary endpoints.

### MNIST — IID

| Method | Condition | Rounds 91–100 | Round 100 |
| --- | --- | ---: | ---: |
| BASIL | attack only | 92.178 ± 0.249 | 92.295 ± 0.301 |
| BASIL | clean | 92.266 ± 0.141 | 92.386 ± 0.116 |
| BASIL | attack + noise | 82.606 ± 1.072 | 82.212 ± 1.123 |
| FedSharp | attack only | 91.003 ± 0.221 | 91.103 ± 0.172 |
| FedSharp | clean | 90.980 ± 0.183 | 91.063 ± 0.120 |
| FedSharp | attack + noise | 88.787 ± 0.364 | 88.539 ± 0.570 |

### MNIST — non-IID

| Method | Condition | Rounds 91–100 | Round 100 |
| --- | --- | ---: | ---: |
| BASIL + ACDS | attack + noise | 82.228 ± 0.639 | 82.084 ± 0.485 |
| BASIL | attack only | 89.824 ± 0.242 | 90.124 ± 0.096 |
| BASIL | clean | 89.562 ± 0.200 | 89.674 ± 0.223 |
| BASIL | attack + noise | 79.132 ± 1.038 | 78.979 ± 1.190 |
| FedSharp | attack only | 86.843 ± 0.137 | 87.180 ± 0.372 |
| FedSharp+ | attack + noise | 84.638 ± 0.446 | 84.642 ± 0.469 |
| FedSharp | clean | 86.864 ± 0.473 | 87.057 ± 0.462 |
| FedSharp | attack + noise | 83.965 ± 0.845 | 84.010 ± 0.861 |

### CIFAR-10 — IID

| Method | Condition | Rounds 91–100 | Round 100 |
| --- | --- | ---: | ---: |
| BASIL | attack only | 32.386 ± 0.503 | 32.588 ± 0.543 |
| BASIL | clean | 32.616 ± 0.742 | 32.819 ± 0.844 |
| BASIL | attack + noise | 29.910 ± 0.578 | 29.820 ± 0.856 |
| FedSharp | attack only | 29.664 ± 0.916 | 29.739 ± 0.949 |
| FedSharp | clean | 29.917 ± 0.556 | 30.006 ± 0.655 |
| FedSharp | attack + noise | 36.315 ± 0.420 | 36.474 ± 0.593 |

### CIFAR-10 — non-IID

| Method | Condition | Rounds 91–100 | Round 100 |
| --- | --- | ---: | ---: |
| BASIL + ACDS | attack + noise | 29.759 ± 0.419 | 29.531 ± 0.859 |
| BASIL | attack only | 16.808 ± 0.490 | 16.898 ± 0.742 |
| BASIL | clean | 16.795 ± 0.103 | 16.986 ± 0.166 |
| BASIL | attack + noise | 20.694 ± 0.187 | 20.496 ± 0.555 |
| FedSharp | attack only | 12.448 ± 1.405 | 12.704 ± 1.463 |
| FedSharp+ | attack + noise | 27.746 ± 0.647 | 27.816 ± 0.623 |
| FedSharp | clean | 12.593 ± 1.455 | 12.671 ± 1.419 |
| FedSharp | attack + noise | 19.048 ± 0.508 | 19.319 ± 0.782 |

## Reproduction

```bash
MPLCONFIGDIR=/tmp/papermerge-fedsharp-mpl OPENBLAS_NUM_THREADS=1 \
  environment/basil-noise-env/bin/python scripts/review_fedsharp_thesis_results.py \
  --output docs/results/hal9k_basil-sharp-thesis-100r-20261030 \
  --destination /tmp/fedsharp-thesis-review-new
```

The reviewer is outside the frozen producer fingerprint. Original scientific records and source snapshots remain unchanged; duplicate plot PDFs were removed with the packaging change recorded and checksummed. Only the reviewed SVG deliverables are intended for the thesis Images folder; PNG, CSV and audit files stay with this review. Plot PDFs are not produced.
