# Trained model downloads

All eight model bundles downloaded locally from the HPC servers are published
in [this GitHub Release](https://github.com/Ojaswi03/PaperMerge/releases/tag/trained-models-500r-20261111).
Each run contains **100 client parameter vectors**, for 800 models in total.
These eight runs are not all 150 study checkpoints.

Each case is non-IID, Hidden attack + noise with sigma 0.01, seed 2025, final
round 500. MNIST and CIFAR-10 each have BASIL, BASIL + ACDS, FedSharp and FedSharp+.

```bash
python3 scripts/download_research_models.py --list
python3 scripts/download_research_models.py
```

The default downloads eight compact `model.npz` files into
`.papermerge/models/final-500r/<runId>/model.npz`. Each includes all 100 retained
client models and a metadata contract. No parameters are averaged or altered.

For the exact files downloaded from the HPC servers:

```bash
python3 scripts/download_research_models.py --variant original
```

This selects four original full FedSharp/FedSharp+ checkpoints and four original
compact BASIL/BASIL + ACDS copies. Full BASIL checkpoints were not downloaded
locally and are not included. Four additional compact FedSharp files have arrays
exactly equal to their full-checkpoint counterparts. Compact copies support
inference; they are not complete training-resume checkpoints.

Use **`nodeParams`**, not attack-corrupted `outbound`, for inference. Compact
files contain only `nodeParams.npy` and `model_metadata.json`.

[manifest.json](manifest.json) records download URLs, sizes, SHA-256, run IDs,
test-result locations and all 100 client parameter hashes per run. Downloads
are verified before installation. Existing matching files are reused; conflicting
files are preserved and rejected. Existing full checkpoints take precedence in
the evaluator when both formats are installed.

See [official-test instructions](../docs/FEDSHARP_OFFICIAL_TEST.md) for local
evaluation. The saved evaluation used CPU TensorFlow 2.18 and NumPy 1.26.4,
versus HPC training TensorFlow 2.21. No training, prior adjustment or ensemble
voting occurs during inference.

Claude can read [the research index](../docs/RESEARCH_INDEX.md) and JSON/CSV
summaries directly. Inspecting weights or reproducing inference requires
downloading the release assets and executing code.
