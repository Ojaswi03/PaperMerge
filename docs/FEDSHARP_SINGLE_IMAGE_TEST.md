# One-image inference from the completed 500-round checkpoints

This demonstration evaluates **official test image 0** independently with all
100 retained client models of FedSharp and all 100 of FedSharp+, for each of
MNIST and CIFAR-10. It uses non-IID, Hidden attack + noise, seed 2025.
All four selected final checkpoints belong to **themachine**. The shared home
directory contains the queue, but `/var/tmp` checkpoints are host-local.

The demonstration does not train, resume a campaign, simulate attacks/noise,
apply a training-time class prior, or change a checkpoint. FedSharp+ inference
uses raw logits from its trained/committed parameters. Corrupted outbound
copies are not the retained classifier models and are not used for inference.

## Run on themachine

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"

python scripts/test_fedsharp_image.py --queue "$shared_run" \
  --datasets mnist cifar10 --attack hidden --seed 2025 --index 0 --describe

demo_output="$HOME/Masters_Research/PaperMerge/.papermerge/evaluations/single-image-$(date +%Y%m%d-%H%M%S)"

python scripts/test_fedsharp_image.py --queue "$shared_run" \
  --datasets mnist cifar10 --attack hidden --seed 2025 --index 0 \
  --destination "$demo_output"
```

CPU inference is sufficient and leaves shared GPUs available. Use the original
`.venv-hpc`; TensorFlow/NumPy versions and required inference sources are checked
against the frozen training records. Every final model is checked against its
recorded hash. The script verifies source snapshots, completion, matched
data/partition/initialization, strict checkpoint structure, and saved metrics.
It hashes the checkpoint and experiment records before and after inference.
Missing checkpoints or changed identities are errors, not reasons to retrain or
substitute a different model. The official test files must already be cached in
the original Keras dataset locations; there are no automatic downloads.

The description command reads the queue/configurations only, not test images
or model parameters. Evaluation selects cases and the image index before seeing
any prediction. Existing report folders are never overwritten. Experiment
outputs are not accepted as report destinations.

## Read the output

- `summary.txt`: correct clients out of 100 and mean cross-entropy for each method.
- `per_client.csv`: predicted class, correct/incorrect, prediction confidence,
  true-class probability, and loss for every retained client model.
- `results.json`: complete predictions/probabilities, honest-client summaries,
  test-image identities, source/checkpoint hashes, and evaluation provenance.
- `mnist_test_image.png`, `cifar10_test_image.png`: the exact images evaluated,
  saved losslessly at their original benchmark resolution.

The loss is ordinary sparse softmax cross-entropy in natural-log units. It does
not include the gradient-norm penalty or proximal term used during training.
For one client and one image, correctness is either 0 or 1. The fraction of
client models that get this image right is **not test-set accuracy** and those
100 predictions are not 100 independent test images. No parameter averaging or
ensemble voting is performed.

This action uses one example from each official test split. Record it as a
demonstration, not an untouched final test result. Do not tune thresholds, choose
seeds/clients, or change the method in response to these predictions and then
describe the same examples as an independent final evaluation. A defensible
test-set performance claim needs a separately specified evaluation across the
official test set and the chosen frozen cases/seeds.

The utility writes local reports under `.papermerge/evaluations` and never
publishes automatically. The author subsequently authorized publication of the
saved demonstration, full test results and downloaded HPC models. See
[the research index](RESEARCH_INDEX.md). Credentials and live queue state stay private.

## Run locally after copying the final weights

The historical HPC results exports contain measurements and source snapshots.
The final weights are now separately available through [the model release](../models/README.md).
Earlier pilot checkpoints are different experiments and must not be substituted.

```bash
python3 scripts/download_research_models.py
```

For direct HPC retrieval instead of release downloads, the original four full
checkpoint paths are listed below. Preserve the `runId/checkpoint.npz` structure:

```text
mnist-dirichlet-2025-sharp_joint-905dcf7b651b/checkpoint.npz
mnist-dirichlet-2025-sharp_cart-46236275f264/checkpoint.npz
cifar10-dirichlet-2025-sharp_joint-ef8b6adcfc43/checkpoint.npz
cifar10-dirichlet-2025-sharp_cart-6aea2e473335/checkpoint.npz
```

The recorded original root for all four is
`/var/tmp/papermerge-1718278706/themachine_fedsharp-all-500r-724feac03d03`.
If the files have since been moved, use the current fleet registry location or
the actual preserved backup; do not use unrelated pilot weights.

Once copied, run from the local checkout:

```bash
cd /home/ojaswi/PaperMerge
models_dir="$PWD/.papermerge/models/final-500r"
report_dir="$PWD/.papermerge/evaluations/local-single-image-$(date +%Y%m%d-%H%M%S)"

environment/basil-noise-env/bin/python scripts/test_fedsharp_image.py \
  --checkpoint-dir "$models_dir" --datasets mnist cifar10 \
  --attack hidden --seed 2025 --index 0 --describe

environment/basil-noise-env/bin/python scripts/test_fedsharp_image.py \
  --checkpoint-dir "$models_dir" --datasets mnist cifar10 \
  --attack hidden --seed 2025 --index 0 --destination "$report_dir"
```

Local mode checks the published manifest, frozen sources, recorded checkpoint
contract, metrics and every final retained parameter hash. It accepts a different
local TensorFlow/NumPy version for forward inference and records that difference
explicitly, including whether the initializer differs. The initializer is never
used for prediction: all trained parameters are loaded. Exact numerical equality
between different runtimes is not promised. The server mode requires matching
training library versions. Both modes use CPU and raw-logit inference only.
