# FedSharp and FedSharp+ | PaperMerge

Research code and results for robust decentralized federated learning under
Byzantine attacks and noisy model exchange. Author: **Ojaswi Sinha**, MS Computer
Science, Ohio University. Advisor: **Dr. Animesh Yadav**.

**Start with the [research index](docs/RESEARCH_INDEX.md).** It identifies the
current experiments, code, figures, model downloads, and material for revising
the thesis. Historical exports remain available and are identified separately.

## Current validation study

The current matrix contains **150 runs at 500 rounds**: MNIST and CIFAR-10,
IID and non-IID partitions, and seeds 2025, 2026, and 2027. It combines 126
impaired-environment runs with 24 matched clean controls. Both clean methods
accept the immediate predecessor without defenses and use ordinary SGD. Their
twelve paired validation trajectories match exactly.

The thesis endpoint averages rounds 491-500. Curves show three-seed means and
sample SD bands. These are development-holdout results, not official test
results. See [validation evidence](docs/thesis/results-evidence-20261104/) and
[the eight individual plots](docs/figures/Thesis_Plots/).

## Official test evaluation

Eight frozen final round-500 runs were evaluated locally: four methods across
two datasets, **non-IID, Hidden attack + noise, seed 2025**. Each of 100 retained
client models classified all 10,000 official test images per dataset. This gives
8,000,000 client-image predictions across 20,000 distinct images.

| Dataset | BASIL | BASIL + ACDS | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: | ---: |
| MNIST test accuracy | 71.474% | 74.857% | 85.797% | 84.883% |
| CIFAR-10 test accuracy | 12.259% | 18.437% | 14.523% | 29.086% |

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

- [Research paper plots](docs/figures/Research_Paper_Plots/): two combined figures, PNG/SVG, excluding FedSharp+.
- [Thesis plots](docs/figures/Thesis_Plots/): eight individual figures, PNG/SVG, including FedSharp+ in four non-IID panels.
- [Official test plots](docs/results/official-test-20261110/four-method-official-test-20261110/): four figures, PNG/SVG.
- [Thesis revision brief](docs/thesis/Thesis_Results_Revision_Brief.md): evidence, placement and equation-approval requirements.
- [Presentation draft](docs/presentation/defense-draft-20261104/): editable PPTX, PDF preview and speaker notes. This November 4 draft predates official testing.

PNG figures use 600 DPI; SVG figures retain vector paths. The original thesis
source, bibliography and thesis PDF supplied separately are not changed by this
publication. Private environments, credentials and live runtime state are excluded.
