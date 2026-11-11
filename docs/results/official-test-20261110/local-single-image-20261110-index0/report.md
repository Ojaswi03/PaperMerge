# Local single-image inference

Settings: final round 500, seed 2025, non-IID, Hidden attack + channel noise. CPU inference on the local PC.

One official test image per dataset: MNIST index 0 (digit 7), CIFAR-10 index 0 (cat). Each image is evaluated separately with all 100 retained client models of each method. No model averaging or ensemble voting.

| Dataset | Method | Correct clients / 100 | Mean cross-entropy | Mean predicted-class confidence | Mean true-class probability |
|---|---|---:|---:|---:|---:|
| mnist | FedSharp | 100/100 | 0.023299 | 97.83% | 97.83% |
| mnist | FedSharp+ | 100/100 | 0.006764 | 99.33% | 99.33% |
| cifar10 | FedSharp | 9/100 | 2.508228 | 33.46% | 11.05% |
| cifar10 | FedSharp+ | 27/100 | 1.772472 | 29.64% | 19.45% |

The fraction of correct clients on one image is not test-set accuracy. The 100 predictions reuse the same image and are not independent test examples. Loss is raw-logit cross-entropy, with no training-only penalties or test-time prior adjustment. These observations do not establish an overall method ranking.

Training used TensorFlow 2.21.0; local inference used TensorFlow 2.18.0. All trained weights were loaded and final parameter hashes were checked. Runtime details are recorded in results.json.

Checkpoint and experiment files were hashed before and after inference; no changes were detected. No optimizer updates were performed.
