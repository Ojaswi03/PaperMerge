# Full official test evaluation of downloaded final models

`scripts/evaluate_fedsharp_test.py` evaluates the frozen final round-500 client
models on all 10,000 official test images of MNIST and CIFAR-10. Its scope is
four matched runs by default: FedSharp and FedSharp+, non-IID, Hidden attack +
noise, seed 2025. All four methods can be selected as shown below. The frozen
models are now available from [the published model release](../models/README.md):

```bash
python3 scripts/download_research_models.py
```

```bash
cd /home/ojaswi/PaperMerge
evaluation_output="$PWD/.papermerge/evaluations/full-test-$(date +%Y%m%d-%H%M%S)"
environment/basil-noise-env/bin/python scripts/evaluate_fedsharp_test.py \
  --checkpoint-dir "$PWD/.papermerge/models/final-500r" \
  --destination "$evaluation_output" --batch-size 256 --threads 4
```

The script refuses an existing destination. This preserves earlier results;
do not rerun to choose a preferred outcome. Progress is printed every ten
client models and saved in `progress.json`.

## Frozen evaluation protocol

- Cases and final checkpoints are declared before any test predictions.
- All 100 retained trained/committed client models are scored separately.
  Outgoing attack-corrupted copies are not the classifier parameters.
- There are no optimizer steps, candidate admission, classifier repair,
  test-time prior adjustment, channel simulation, parameter averaging or voting.
- Raw test pixels use the same float32 `/255` normalization as fitting data.
- Model-source snapshots, published experiment identities, strict checkpoint
  contracts, final parameter hashes and final validation arrays are checked.
- Checkpoints and experiment records are hashed before and after evaluation.
- The CPU runtime is recorded, including differences from training versions.

Training learns parameters from fitting data. The original development holdout
was used for validation. This script measures the completed, frozen models on
the separate official test split. Image 0 of each dataset was demonstrated
previously, without subsequent changes to these models. Full-test results and
the remaining 9,999-example subset are both reported. The test results must not
be used to tune thresholds or choose new models and then be presented as an
untouched final evaluation of those changed choices.

## Metrics and outputs

`results.json` records every client's test accuracy, cross-entropy, class
recall and confusion matrix. It contains all-client and honest-client
summaries separately. `per_client.csv` contains convenient numerical rows.
Each run's `predictions.npz` stores labels, predicted classes, per-example loss
and confidence, enabling independent calculation of all reported metrics.
`report.md` gives a readable comparison with final-round validation.
`SHA256.json` checks output integrity.

The primary test accuracy is the unweighted mean of the 100 individual client
accuracies, each evaluated on all 10,000 test images. Honest-client summaries
use the 67 identities not configured to attack outgoing copies during training.
Cross-entropy is ordinary raw-logit classification loss; the training-only
gradient and proximal penalties are excluded. Validation comparisons use the
same final round-500 checkpoint, rather than the rounds-491-to-500 validation
average used elsewhere in the thesis.

The four runs give one seed per dataset/method, not a three-seed comparison.
Client dispersion is not seed uncertainty or a confidence interval. Extending
this evaluation to all thesis conditions requires the corresponding additional
frozen checkpoints and a separately declared case/seed matrix. Never pool
client models as if they were independent training repetitions.

The evaluator writes private local outputs under `.papermerge` and never
publishes automatically. At the author's explicit request, the completed
November 10 evaluation, prediction arrays and figures were subsequently copied
to [the public results package](results/official-test-20261110/). Original
downloaded model files and compact copies are published as release assets.
The original result exports and local source artifacts remain unchanged.

## Add BASIL and BASIL + ACDS

The default release downloader above includes BASIL and BASIL + ACDS.
Alternatively, the original HPC retrieval utility can download compact copies
from hal9k (MNIST BASIL and CIFAR-10 BASIL + ACDS) and themachine (MNIST BASIL +
ACDS and CIFAR-10 BASIL), if those server files remain available:

```bash
cd /home/ojaswi/PaperMerge
environment/basil-noise-env/bin/python scripts/fetch_basil_test_models.py
```

This uses one normal SSH/password connection per server. Its remote exporter
uses Python's standard library, reads completed checkpoint files, verifies the
final trained model hashes, and sends only `nodeParams` plus a compact contract.
It does not use a GPU or perform training, and does not copy the long activation
history or attack-corrupted outbound arrays. Local validation checks the copy
against the published experiment contract before installation as `model.npz`.
Existing local models are skipped. A partial failed transfer does not replace
an installed model; rerunning the downloader retries missing models.

Then evaluate only the missing methods, preserving the prior FedSharp results:

```bash
evaluation_output="$PWD/.papermerge/evaluations/basil-full-test-$(date +%Y%m%d-%H%M%S)"
environment/basil-noise-env/bin/python scripts/evaluate_fedsharp_test.py \
  --checkpoint-dir "$PWD/.papermerge/models/final-500r" \
  --destination "$evaluation_output" --methods basil basil_acds
```

The evaluator also accepts all four method keys in one new run. ACDS retains its
declared additional fitting-data sharing. Its base partition, development holdout
and initialization match the other methods, while its sharing audit is kept
distinct. It should not be described as an equal-data-access comparison.

```bash
environment/basil-noise-env/bin/python scripts/evaluate_fedsharp_test.py \
  --checkpoint-dir "$PWD/.papermerge/models/final-500r" \
  --destination "$PWD/.papermerge/evaluations/new-four-method-evaluation" \
  --methods basil basil_acds fedsharp fedsharp_plus
```

To check the published summaries without rerunning inference:

```bash
MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python \
  scripts/verify_published_test_results.py
```

To regenerate four test figures in PNG and SVG into a fresh directory:

```bash
MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python \
  scripts/render_official_test_figures.py \
  --destination "$PWD/.papermerge/evaluations/regenerated-test-figures"
```

The completed inference used TensorFlow 2.18.0 and NumPy 1.26.4.
Consult the original receipts for the full runtime record before reproducing.
