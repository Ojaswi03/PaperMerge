# Postdefense results

Current 168 runs, 56 three-seed groups, 500 rounds. Accuracy is development-holdout accuracy. Loss is local training CE, not held-out validation/test CE or the full regularized objective. FedSharp+ training CE uses prior-adjusted logits. Endpoints average rounds 491 to 500.

Each context has separate validation-accuracy and training-CE figures in 600-DPI PNG and native-vector SVG. The right panel expands rounds 401 to 500. All legends are below. Clean and noise-only references are repeated across the two attack contexts.

`group_summary.csv` contains all 56 group summaries; `per_run.csv` contains all 168 individual values and timings. `plot_data.json` records the exact plotted means/SDs and run identities. `audit.json` links the five source manifests and records figure checks.

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
