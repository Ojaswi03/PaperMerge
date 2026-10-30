# FedSharp thesis comparison and figure contract

The current verified study has 84 configurations: MNIST/CIFAR-10, IID/non-IID,
three seeds (2025–2027), 100 clients and 100 complete ring rounds, with five
mini-batch SGD steps per activation. Each dataset has six IID curves and eight
non-IID curves. The duplicated impairment condition was intentionally removed.

| Method | IID conditions | Additional non-IID condition |
| --- | --- | --- |
| BASIL | Clean; attack only; attack + noise | BASIL + ACDS under attack + noise |
| FedSharp | Clean; attack only; attack + noise | FedSharp+ under attack + noise |

BASIL performs single-model loss selection over up to 34 predecessors.
FedSharp uses the 66-predecessor guarded selector and a check of the realized
pooled model. Clean conditions retain the respective algorithm's selection.
Noise is absolute Gaussian coordinate noise with standard deviation 0.01 in
noisy conditions; the implemented Hidden attack begins at displayed round 21.
Non-IID class allocation uses concentration 0.5. ACDS shares 5% of fitting
examples; FedSharp+ uses the full declared local-prior/class-aware extension.

## Verified evidence and publication

- [Current verified export](results/hal9k_basil-sharp-thesis-100r-20261030/).
- [Review and all numerical endpoints](FEDSHARP_RESULTS_REVIEW_20261030.md).
- [Reviewed SVGs and numeric tables](thesis-figures/fedsharp-hal9k-20261030/).
- [Current launch, monitor, verification and export commands](../README.md).

The four thesis files are `Images/mnist_iid_accuracy.svg`,
`Images/mnist_non_iid_accuracy.svg`, `Images/cifar10_iid_accuracy.svg`, and
`Images/cifar10_non_iid_accuracy.svg`. These four figures contain 28 condition
curves averaged across 84 runs. No bars or smoothed trajectories are used.
The axis is validation accuracy because evaluation uses a fixed original-TRAIN
holdout. The shaded band is sample SD across seeds after averaging clients.
The raw arrays retain per-class, honest-client and worst-client diagnostics.

Source producer files and their snapshots remain unchanged. FedSharp and
FedSharp+ are presentation names; internal SHARP/CART identifiers are preserved
for provenance. The separate earlier 66-run ablations remain reference evidence,
not interchangeable curves for this matrix.
