# Official test-set evaluation

Frozen final round-500 models; non-IID, Hidden attack + noise, seed 2025. No retraining or test-based tuning.
All 10,000 official test images per dataset, evaluated separately by each of 100 retained client models.

| Dataset | Method | Test accuracy, all clients | Mean test CE | Honest-client accuracy | Round-500 validation |
|---|---|---:|---:|---:|---:|
| mnist | BASIL | 71.474% | 1.167073 | 71.971% | 69.104% |
| mnist | BASIL + ACDS | 74.857% | 1.576609 | 74.541% | 73.953% |
| cifar10 | BASIL | 12.259% | 2.609257 | 12.068% | 12.249% |
| cifar10 | BASIL + ACDS | 18.437% | 2.780031 | 18.263% | 18.452% |

Accuracy is first computed over all test images for each client model, then averaged equally across clients.
Honest-client results use the 67 identities not configured to corrupt outgoing copies during training.
All-client inference uses retained trained parameters, including the trained models of attack-configured identities, not their corrupted outgoing copies.
Loss is ordinary raw-logit cross-entropy in natural-log units, without proximal/gradient penalties or test-time prior adjustment.

## Scope and interpretation

This evaluation covers 4 runs of one seed and one non-IID attack/noise setting, not the full 150-run study or a three-seed result.
The validation comparison is the final round-500 checkpoint, not the mean of rounds 491 to 500 used for the main validation endpoints.
Client variation is not seed variation, a confidence interval, or evidence of statistical significance.
Test image 0 was demonstrated earlier. No model changes were made in response; the JSON also reports the other 9,999 images separately.
The official test data were not used for fitting, admission, repair, checkpoint selection, or threshold tuning in this evaluation.
Local CPU inference uses TensorFlow 2.18.0; training used 2.21.0. All trained parameters and model-source identities were checked; runtimes are recorded.
Original checkpoints and experiment artifacts were hashed before and after evaluation and remained unchanged.
