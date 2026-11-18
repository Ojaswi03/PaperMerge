# Postdefense results

Current 168 runs, 56 three-seed groups, 500 rounds. Accuracy is development-holdout accuracy. Loss is local training CE, not held-out validation/test CE or the full regularized objective. FedSharp+ training CE uses prior-adjusted logits. Endpoints average rounds 491 to 500.

Each context has separate validation-accuracy and training-CE figures in 600-DPI PNG and native-vector SVG. Each image contains one full-range plot, without a zoom panel. All legends are below. Every curve has a distinct color and marker. Coincident clean controls retain their measured values, with staggered markers and different line styles. Clean and noise-only references are repeated across the two attack contexts.

`group_summary.csv` contains all 56 group summaries; `per_run.csv` contains all 168 individual values and timings. `plot_data.json` records the exact plotted means/SDs and run identities. `audit.json` links the five source manifests and records figure checks.

### MNIST IID

**Validation accuracy (%)**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | **97.602 ± 0.068** | -- | **97.602 ± 0.068** | -- |
| Noise only | 73.645 ± 0.252 | -- | **88.793 ± 0.240** | -- |
| Attack only: Gaussian | **94.088 ± 0.203** | -- | 92.600 ± 0.373 | -- |
| Attack only: Hidden | **94.088 ± 0.203** | -- | 92.562 ± 0.374 | -- |
| Attack + noise: Gaussian | 74.182 ± 2.021 | -- | **87.818 ± 0.650** | -- |
| Attack + noise: Hidden | 73.374 ± 0.898 | -- | **87.814 ± 0.810** | -- |

**Local training CE**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.0102 ± 0.0003 | -- | 0.0102 ± 0.0003 | -- |
| Noise only | 1.0352 ± 0.0086 | -- | 0.3555 ± 0.0116 | -- |
| Attack only: Gaussian | 0.1901 ± 0.0052 | -- | 0.2428 ± 0.0043 | -- |
| Attack only: Hidden | 0.1901 ± 0.0052 | -- | 0.2417 ± 0.0047 | -- |
| Attack + noise: Gaussian | 1.0492 ± 0.0596 | -- | 0.3894 ± 0.0208 | -- |
| Attack + noise: Hidden | 1.0858 ± 0.0394 | -- | 0.3855 ± 0.0065 | -- |

### MNIST non-IID

**Validation accuracy (%)**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | **97.628 ± 0.326** | -- | **97.628 ± 0.326** | -- |
| Noise only | 71.269 ± 0.634 | -- | **85.935 ± 0.504** | 84.442 ± 0.508 |
| Attack only: Gaussian | **92.550 ± 0.156** | -- | 90.076 ± 0.333 | -- |
| Attack only: Hidden | **92.536 ± 0.132** | -- | 90.048 ± 0.179 | -- |
| Attack + noise: Gaussian | 70.344 ± 1.764 | 74.704 ± 1.608 | **84.902 ± 1.173** | 84.264 ± 0.352 |
| Attack + noise: Hidden | 69.320 ± 1.016 | 74.701 ± 0.662 | **84.841 ± 0.618** | 84.296 ± 0.395 |

**Local training CE**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.0110 ± 0.0005 | -- | 0.0110 ± 0.0005 | -- |
| Noise only | 1.0388 ± 0.0351 | -- | 0.3376 ± 0.0112 | 0.1469 ± 0.0008 |
| Attack only: Gaussian | 0.1951 ± 0.0016 | -- | 0.2439 ± 0.0045 | -- |
| Attack only: Hidden | 0.1947 ± 0.0010 | -- | 0.2407 ± 0.0041 | -- |
| Attack + noise: Gaussian | 1.0879 ± 0.0665 | 1.0532 ± 0.0428 | 0.3741 ± 0.0121 | 0.1331 ± 0.0062 |
| Attack + noise: Hidden | 1.0274 ± 0.0987 | 1.0242 ± 0.0507 | 0.3774 ± 0.0090 | 0.1337 ± 0.0052 |

### CIFAR-10 IID

**Validation accuracy (%)**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | **61.754 ± 0.328** | -- | **61.754 ± 0.328** | -- |
| Noise only | -- | -- | 46.456 ± 1.023 | -- |
| Attack only: Gaussian | **41.488 ± 1.361** | -- | 34.855 ± 0.468 | -- |
| Attack only: Hidden | **41.488 ± 1.361** | -- | 34.855 ± 0.468 | -- |
| Attack + noise: Gaussian | 15.960 ± 3.206 | -- | **44.757 ± 1.086** | -- |
| Attack + noise: Hidden | 16.201 ± 2.835 | -- | **44.757 ± 1.086** | -- |

**Local training CE**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.3569 ± 0.0150 | -- | 0.3569 ± 0.0150 | -- |
| Noise only | -- | -- | 1.5116 ± 0.0188 | -- |
| Attack only: Gaussian | 1.5924 ± 0.0329 | -- | 1.7558 ± 0.0076 | -- |
| Attack only: Hidden | 1.5924 ± 0.0329 | -- | 1.7558 ± 0.0076 | -- |
| Attack + noise: Gaussian | 2.4566 ± 0.3574 | -- | 1.5401 ± 0.0137 | -- |
| Attack + noise: Hidden | 2.3470 ± 0.0159 | -- | 1.5401 ± 0.0137 | -- |

### CIFAR-10 non-IID

**Validation accuracy (%)**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | **59.980 ± 0.488** | -- | **59.980 ± 0.488** | -- |
| Noise only | -- | -- | 26.723 ± 1.419 | **29.062 ± 0.450** |
| Attack only: Gaussian | **24.812 ± 1.018** | -- | 18.527 ± 0.586 | -- |
| Attack only: Hidden | **25.025 ± 0.978** | -- | 18.317 ± 0.614 | -- |
| Attack + noise: Gaussian | 11.488 ± 2.556 | 19.081 ± 4.407 | 24.680 ± 1.343 | **28.983 ± 0.561** |
| Attack + noise: Hidden | 14.770 ± 2.877 | 18.475 ± 3.730 | 14.626 ± 0.672 | **28.983 ± 0.561** |

**Local training CE**

| Condition | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.3617 ± 0.0034 | -- | 0.3617 ± 0.0034 | -- |
| Noise only | -- | -- | 1.3942 ± 0.0370 | 0.9094 ± 0.0927 |
| Attack only: Gaussian | 1.5310 ± 0.0172 | -- | 1.6686 ± 0.0138 | -- |
| Attack only: Hidden | 1.5271 ± 0.0181 | -- | 1.6631 ± 0.0240 | -- |
| Attack + noise: Gaussian | 1.9714 ± 0.0892 | 2.2066 ± 0.0913 | 1.5032 ± 0.0354 | 0.9018 ± 0.0848 |
| Attack + noise: Hidden | 1.8429 ± 0.1145 | 2.2282 ± 0.0737 | 1.8150 ± 0.0356 | 0.9018 ± 0.0848 |
