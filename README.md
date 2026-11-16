# FedSharp and FedSharp+ | PaperMerge

Research code and results for robust decentralized federated learning under
Byzantine attacks and noisy model exchange. Author: **Ojaswi Sinha**, MS Computer
Science, Ohio University. Advisor: **Dr. Animesh Yadav**.

**Start with the [research index](docs/RESEARCH_INDEX.md).** It identifies the
current experiments, code, figures, model downloads, and material for revising
the thesis. Historical exports remain available and are identified separately.

## Current 500-round validation results

The current study contains **168 runs in 56 three-seed groups**: MNIST and CIFAR-10, IID and non-IID, with seeds 2025, 2026 and 2027. The latest 18 noise-only runs are complete and independently verified on the producing server. The dataset includes 24 clean, 48 attack-only, 72 attack-and-noise and 24 noise-only runs.

**Each table cell reports validation accuracy (%) and local-training cross-entropy (CE), both as mean +/- sample SD across three seeds.** Each seed endpoint averages rounds 491 to 500 and all 100 clients. These are development-holdout accuracies, not official test results.

**Loss scope:** the exports do not contain validation loss for every run. CE below is the saved local-training loss (`mechanisms.json`, `meanLocalTrainCE`), averaged over five SGD steps per activation, then clients and rounds. It excludes the added gradient-norm and proximal penalties. FedSharp+ CE uses prior-adjusted training logits; other methods use their recorded training logits. These local losses are diagnostics, not a common raw-logit validation/test loss and should not be used alone to rank the methods. The separate official-test table reports raw-logit test CE where available.

Gaussian and Hidden identify the Byzantine attack. Attack conditions use 33 attackers starting in round 21. Noise conditions use channel sigma 0.01; noise-only has zero attackers. Clean BASIL and FedSharp accept the immediate predecessor without defenses and use mu zero. Their twelve paired accuracy and training-CE trajectories match exactly.

[All 168 per-run values](docs/figures/postdefense_results/per_run.csv) | [All 56 group summaries](docs/figures/postdefense_results/group_summary.csv) | [Source and figure checks](docs/figures/postdefense_results/audit.json)

### MNIST IID

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 97.602 +/- 0.068%<br>CE 0.0102 +/- 0.0003 | Not run | 97.602 +/- 0.068%<br>CE 0.0102 +/- 0.0003 | Not run |
| Noise only | 73.645 +/- 0.252%<br>CE 1.0352 +/- 0.0086 | Not run | 88.793 +/- 0.240%<br>CE 0.3555 +/- 0.0116 | Not run |
| Attack only: Gaussian | 94.088 +/- 0.203%<br>CE 0.1901 +/- 0.0052 | Not run | 92.600 +/- 0.373%<br>CE 0.2428 +/- 0.0043 | Not run |
| Attack only: Hidden | 94.088 +/- 0.203%<br>CE 0.1901 +/- 0.0052 | Not run | 92.562 +/- 0.374%<br>CE 0.2417 +/- 0.0047 | Not run |
| Attack + noise: Gaussian | 74.182 +/- 2.021%<br>CE 1.0492 +/- 0.0596 | Not run | 87.818 +/- 0.650%<br>CE 0.3894 +/- 0.0208 | Not run |
| Attack + noise: Hidden | 73.374 +/- 0.898%<br>CE 1.0858 +/- 0.0394 | Not run | 87.814 +/- 0.810%<br>CE 0.3855 +/- 0.0065 | Not run |

### MNIST non-IID

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 97.628 +/- 0.326%<br>CE 0.0110 +/- 0.0005 | Not run | 97.628 +/- 0.326%<br>CE 0.0110 +/- 0.0005 | Not run |
| Noise only | 71.269 +/- 0.634%<br>CE 1.0388 +/- 0.0351 | Not run | 85.935 +/- 0.504%<br>CE 0.3376 +/- 0.0112 | 84.442 +/- 0.508%<br>CE 0.1469 +/- 0.0008 |
| Attack only: Gaussian | 92.550 +/- 0.156%<br>CE 0.1951 +/- 0.0016 | Not run | 90.076 +/- 0.333%<br>CE 0.2439 +/- 0.0045 | Not run |
| Attack only: Hidden | 92.536 +/- 0.132%<br>CE 0.1947 +/- 0.0010 | Not run | 90.048 +/- 0.179%<br>CE 0.2407 +/- 0.0041 | Not run |
| Attack + noise: Gaussian | 70.344 +/- 1.764%<br>CE 1.0879 +/- 0.0665 | 74.704 +/- 1.608%<br>CE 1.0532 +/- 0.0428 | 84.902 +/- 1.173%<br>CE 0.3741 +/- 0.0121 | 84.264 +/- 0.352%<br>CE 0.1331 +/- 0.0062 |
| Attack + noise: Hidden | 69.320 +/- 1.016%<br>CE 1.0274 +/- 0.0987 | 74.701 +/- 0.662%<br>CE 1.0242 +/- 0.0507 | 84.841 +/- 0.618%<br>CE 0.3774 +/- 0.0090 | 84.296 +/- 0.395%<br>CE 0.1337 +/- 0.0052 |

### CIFAR-10 IID

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 61.754 +/- 0.328%<br>CE 0.3569 +/- 0.0150 | Not run | 61.754 +/- 0.328%<br>CE 0.3569 +/- 0.0150 | Not run |
| Noise only | Not run | Not run | 46.456 +/- 1.023%<br>CE 1.5116 +/- 0.0188 | Not run |
| Attack only: Gaussian | 41.488 +/- 1.361%<br>CE 1.5924 +/- 0.0329 | Not run | 34.855 +/- 0.468%<br>CE 1.7558 +/- 0.0076 | Not run |
| Attack only: Hidden | 41.488 +/- 1.361%<br>CE 1.5924 +/- 0.0329 | Not run | 34.855 +/- 0.468%<br>CE 1.7558 +/- 0.0076 | Not run |
| Attack + noise: Gaussian | 15.960 +/- 3.206%<br>CE 2.4566 +/- 0.3574 | Not run | 44.757 +/- 1.086%<br>CE 1.5401 +/- 0.0137 | Not run |
| Attack + noise: Hidden | 16.201 +/- 2.835%<br>CE 2.3470 +/- 0.0159 | Not run | 44.757 +/- 1.086%<br>CE 1.5401 +/- 0.0137 | Not run |

### CIFAR-10 non-IID

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 59.980 +/- 0.488%<br>CE 0.3617 +/- 0.0034 | Not run | 59.980 +/- 0.488%<br>CE 0.3617 +/- 0.0034 | Not run |
| Noise only | Not run | Not run | 26.723 +/- 1.419%<br>CE 1.3942 +/- 0.0370 | 29.062 +/- 0.450%<br>CE 0.9094 +/- 0.0927 |
| Attack only: Gaussian | 24.812 +/- 1.018%<br>CE 1.5310 +/- 0.0172 | Not run | 18.527 +/- 0.586%<br>CE 1.6686 +/- 0.0138 | Not run |
| Attack only: Hidden | 25.025 +/- 0.978%<br>CE 1.5271 +/- 0.0181 | Not run | 18.317 +/- 0.614%<br>CE 1.6631 +/- 0.0240 | Not run |
| Attack + noise: Gaussian | 11.488 +/- 2.556%<br>CE 1.9714 +/- 0.0892 | 19.081 +/- 4.407%<br>CE 2.2066 +/- 0.0913 | 24.680 +/- 1.343%<br>CE 1.5032 +/- 0.0354 | 28.983 +/- 0.561%<br>CE 0.9018 +/- 0.0848 |
| Attack + noise: Hidden | 14.770 +/- 2.877%<br>CE 1.8429 +/- 0.1145 | 18.475 +/- 3.730%<br>CE 2.2282 +/- 0.0737 | 14.626 +/- 0.672%<br>CE 1.8150 +/- 0.0356 | 28.983 +/- 0.561%<br>CE 0.9018 +/- 0.0848 |

`Not run` means no matching case in the current matrix, not zero accuracy. BASIL + ACDS has only non-IID attack-and-noise cases. BASIL noise-only covers MNIST only. FedSharp+ has non-IID attack-and-noise and noise-only cases; there is no FedSharp+ clean or IID case.

BASIL performs better in the measured attack-only cases. FedSharp generally performs better when attack and noise occur together; CIFAR-10 non-IID Hidden is the near-equal exception for the base methods. FedSharp+ performs best on CIFAR-10 non-IID attack-and-noise and noise-only, while base FedSharp is slightly higher on MNIST non-IID. These are descriptive comparisons, not significance claims or universal guarantees. ACDS uses additional fitting-data access. Clean-versus-noisy comparisons also change the defensive policy, so their gaps cannot be attributed only to the environment.

The latest noise-only export is [here](docs/results/themachine_fedsharp-noise-only-500r-20261114/). One CIFAR-10 FedSharp+ run resumed from activation 41,523 after a CUDA timeout; it completed and passed fresh CPU verification. Timing values in the per-run CSV exclude recovery downtime. Historical timing comparisons reflect different shared-server loads. Use the [automatic launcher](docs/FEDSHARP_NOISE_ONLY_AUTO.md) for reproduction.

## Updated postdefense plots

All curves show the three-seed mean and sample SD, with all 500 unsmoothed rounds. Legends are below each plot; the right panel shows rounds 401 to 500 without covering the full trajectory. Clean and noise-only references appear in both attack contexts. The context label applies only to the attacked curves. Training-CE figures use the loss definition above. PNG is 600 DPI and SVG is native vector.

### MNIST IID | Gaussian attack conditions

![MNIST IID | Gaussian attack conditions: validation accuracy](docs/figures/postdefense_results/mnist_iid_gaussian_accuracy.png)

![MNIST IID | Gaussian attack conditions: local training CE](docs/figures/postdefense_results/mnist_iid_gaussian_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/mnist_iid_gaussian_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/mnist_iid_gaussian_training_loss.svg)

### MNIST IID | Hidden attack conditions

![MNIST IID | Hidden attack conditions: validation accuracy](docs/figures/postdefense_results/mnist_iid_hidden_accuracy.png)

![MNIST IID | Hidden attack conditions: local training CE](docs/figures/postdefense_results/mnist_iid_hidden_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/mnist_iid_hidden_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/mnist_iid_hidden_training_loss.svg)

### MNIST non-IID | Gaussian attack conditions

![MNIST non-IID | Gaussian attack conditions: validation accuracy](docs/figures/postdefense_results/mnist_non_iid_gaussian_accuracy.png)

![MNIST non-IID | Gaussian attack conditions: local training CE](docs/figures/postdefense_results/mnist_non_iid_gaussian_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/mnist_non_iid_gaussian_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/mnist_non_iid_gaussian_training_loss.svg)

### MNIST non-IID | Hidden attack conditions

![MNIST non-IID | Hidden attack conditions: validation accuracy](docs/figures/postdefense_results/mnist_non_iid_hidden_accuracy.png)

![MNIST non-IID | Hidden attack conditions: local training CE](docs/figures/postdefense_results/mnist_non_iid_hidden_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/mnist_non_iid_hidden_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/mnist_non_iid_hidden_training_loss.svg)

### CIFAR-10 IID | Gaussian attack conditions

![CIFAR-10 IID | Gaussian attack conditions: validation accuracy](docs/figures/postdefense_results/cifar10_iid_gaussian_accuracy.png)

![CIFAR-10 IID | Gaussian attack conditions: local training CE](docs/figures/postdefense_results/cifar10_iid_gaussian_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/cifar10_iid_gaussian_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/cifar10_iid_gaussian_training_loss.svg)

### CIFAR-10 IID | Hidden attack conditions

![CIFAR-10 IID | Hidden attack conditions: validation accuracy](docs/figures/postdefense_results/cifar10_iid_hidden_accuracy.png)

![CIFAR-10 IID | Hidden attack conditions: local training CE](docs/figures/postdefense_results/cifar10_iid_hidden_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/cifar10_iid_hidden_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/cifar10_iid_hidden_training_loss.svg)

### CIFAR-10 non-IID | Gaussian attack conditions

![CIFAR-10 non-IID | Gaussian attack conditions: validation accuracy](docs/figures/postdefense_results/cifar10_non_iid_gaussian_accuracy.png)

![CIFAR-10 non-IID | Gaussian attack conditions: local training CE](docs/figures/postdefense_results/cifar10_non_iid_gaussian_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/cifar10_non_iid_gaussian_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/cifar10_non_iid_gaussian_training_loss.svg)

### CIFAR-10 non-IID | Hidden attack conditions

![CIFAR-10 non-IID | Hidden attack conditions: validation accuracy](docs/figures/postdefense_results/cifar10_non_iid_hidden_accuracy.png)

![CIFAR-10 non-IID | Hidden attack conditions: local training CE](docs/figures/postdefense_results/cifar10_non_iid_hidden_training_loss.png)

[Accuracy SVG](docs/figures/postdefense_results/cifar10_non_iid_hidden_accuracy.svg) | [Training CE SVG](docs/figures/postdefense_results/cifar10_non_iid_hidden_training_loss.svg)

## Official test evaluation

Eight frozen final round-500 runs were evaluated locally: four methods across
two datasets, **non-IID, Hidden attack + noise, seed 2025**. Each of 100 retained
client models classified all 10,000 official test images per dataset. This gives
8,000,000 client-image predictions across 20,000 distinct images.

| Dataset / metric | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| MNIST test accuracy | 71.474% | 74.857% | 85.797% | 84.883% |
| MNIST raw-logit test CE | 1.167073 | 1.576609 | 0.491761 | 0.491335 |
| CIFAR-10 test accuracy | 12.259% | 18.437% | 14.523% | 29.086% |
| CIFAR-10 raw-logit test CE | 2.609257 | 2.780031 | 2.634462 | 2.003305 |

These are means over separately evaluated client models, with no parameter
averaging, voting, test-time adjustment, training or tuning. ACDS has extra
fitting-data access. One seed and one attack setting do not support significance
claims or universal superiority. Test image 0 was demonstrated earlier; the
models were unchanged afterward. The other 9,999-image subset is also reported.
CPU TensorFlow 2.18 differs from training TensorFlow 2.21, as recorded in receipts.

- [Test accuracy, loss, class recall and PNG/SVG figures](docs/results/official-test-20261110/four-method-official-test-20261110/report.md)
- [Saved predictions and verification evidence](docs/results/official-test-20261110/)
- [EuroSAT diagnostic](docs/results/official-test-20261110/eurosat-ood-20261110/report.md): different labels, so confidence diagnostics only, not classification accuracy.

## Trained models

The [model index](models/README.md) links all eight locally downloaded HPC model
bundles, containing 800 retained client models. Original checkpoints and compact
inference copies are distributed as [GitHub Release assets](https://github.com/Ojaswi03/PaperMerge/releases/tag/trained-models-500r-20261111).
Every file has a size and SHA-256 in [the manifest](models/manifest.json).

```bash
python3 scripts/download_research_models.py
```

The default downloads compact models into `.papermerge/models/final-500r` and
checks SHA-256 before installation. `--variant original` retrieves the exact
files downloaded from the servers. See [official-test instructions](docs/FEDSHARP_OFFICIAL_TEST.md)
and [the research index](docs/RESEARCH_INDEX.md) for evaluation and HPC guides.
Frozen source snapshots establish which implementation produced each result.

## Figures and thesis material

- [Postdefense figures and tables](docs/figures/postdefense_results/): current 168-run accuracy and training-CE figures, PNG/SVG.

- [Research paper plots](docs/figures/Research_Paper_Plots/): two combined figures, PNG/SVG, excluding FedSharp+.
- [Thesis plots](docs/figures/Thesis_Plots/): eight individual figures, PNG/SVG, including FedSharp+ in four non-IID panels.
- [Official test plots](docs/results/official-test-20261110/four-method-official-test-20261110/): four figures, PNG/SVG.
- [Thesis revision brief](docs/thesis/Thesis_Results_Revision_Brief.md): evidence, placement and equation-approval requirements.
- [Presentation draft](docs/presentation/defense-draft-20261104/): editable PPTX, PDF preview and speaker notes. This November 4 draft predates official testing.

PNG figures use 600 DPI; SVG figures retain vector paths. The original thesis
source, bibliography and thesis PDF supplied separately are not changed by this
publication. Private environments, credentials and live runtime state are excluded.
