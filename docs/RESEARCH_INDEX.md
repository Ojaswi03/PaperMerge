# Research index for thesis review

Start here when reading this repository with Claude or another reviewer.
Prepared on 11 November 2026. Record the current `main` commit SHA before
proposing thesis changes.

## Reading order and editing rules

1. Read [the current summary](../README.md) and [official test report](results/official-test-20261110/four-method-official-test-20261110/report.md).
2. Read [the thesis revision brief](thesis/Thesis_Results_Revision_Brief.md). Its November 4 validation evidence is current; earlier statements about absent official testing are superseded by the November 10 test reports.
3. Use [per-run validation evidence](thesis/results-evidence-20261104/current_runs.csv), [condition summaries](thesis/results-evidence-20261104/condition_summary.csv), and [mechanism summaries](thesis/results-evidence-20261104/mechanism_summary.csv).
4. Inspect current code AND each export's frozen `source_snapshots`. Current code alone does not establish what produced an older result.
5. Use [the model index](../models/README.md) for trained parameters. Binary models require download and code execution; reading repository text alone cannot inspect weights meaningfully.

Use the author's separately supplied `thesis.tex`, `references.bib`, and thesis
PDF. Explain both FedSharp and FedSharp+ in the abstract and methodology, and
clarify Algorithm 2 from the evaluated implementation. Keep the prose suitable
for a master's thesis. Avoid inflated claims, em dashes and curly quotes.

**Equations require the author's explicit approval before changes.** Before
changing, adding, removing, activating commented mathematics, or changing the
mathematical behavior of Algorithm 2, show the exact original and proposed
LaTeX, the reason and supporting evidence. Wait for approval before applying
the mathematical change. Independent prose and figure work can continue.
Do not change experiment data or run experiments as part of thesis editing.

## Current validation study: 150 runs

| Evidence | Location | Role |
| --- | --- | --- |
| Original 500-round fleet | [hal9k: 35](results/hal9k_fedsharp-fleet-500r-20261103/), [themachine: 115](results/themachine_fedsharp-fleet-500r-20261103/) | Retain 126 impaired cases; old clean controls are superseded. |
| Current FedSharp clean | [12 runs](results/themachine_fedsharp-plain-clean-500r-20261104/) | Matched defense-disabled controls. |
| Current BASIL clean | [12 runs](results/themachine_basil-plain-clean-500r-20261104/) | Matched defense-disabled controls. |
| Selected run IDs | [current_runs.csv](thesis/results-evidence-20261104/current_runs.csv) | Exactly 150 cases; three seeds per curve. |
| Historical controls | [superseded_runs.csv](thesis/results-evidence-20261104/superseded_runs.csv) | Never pool repeated seeds. |

Runs use 100 clients, 500 ring rounds and five SGD steps per activation.
The development holdout has 1,000 examples drawn from original training data.
MNIST has 59,000 fitting examples; CIFAR-10 has 49,000. The thesis endpoint
averages rounds 491-500, with sample SD across three seeds.

Both current clean methods accept the immediate predecessor with mu zero in
both partitions and no selection or defense. All twelve paired 500-round
validation arrays match exactly. This is metric equality, not a claim of
parameter byte equality. Impaired methods keep their declared defense policies;
environment comparisons therefore also change algorithmic policy.

## Official test study: eight runs, one seed

| Material | Location |
| --- | --- |
| Four-method summary, figures and per-client CSV | [Combined report](results/official-test-20261110/four-method-official-test-20261110/report.md) |
| FedSharp/FedSharp+ receipts and prediction arrays | [Full official test](results/official-test-20261110/full-official-test-20261110/) |
| BASIL/BASIL + ACDS receipts and prediction arrays | [BASIL official test](results/official-test-20261110/basil-full-test-20261110/) |
| Earlier one-image demonstration | [Single-image results](results/official-test-20261110/local-single-image-20261110-index0/) |
| Unrelated-data diagnostic | [EuroSAT report](results/official-test-20261110/eurosat-ood-20261110/report.md) |
| Frozen model downloads | [Models](../models/README.md) |

All eight cases are non-IID, Hidden attack + noise, seed 2025, final round 500.
Each of 100 client models sees 10,000 official test images per dataset: 20,000
distinct images and 8,000,000 client-image predictions. This is separate from
the rounds-491-to-500 validation endpoint. No training, voting, averaging or
test-time prior adjustment occurred. The 100-client and 67-honest-client
summaries are separate. Client dispersion is not seed uncertainty. Image 0 was
previously demonstrated, and the other 9,999-image subset is also saved.

EuroSAT uses a fixed 2,000-image sample, 200 per land-cover class, resized from
64x64 to 32x32. CIFAR-10 object outputs do not match land-cover labels. Report
confidence diagnostics, not EuroSAT accuracy or loss against an invented class
mapping. CIFAR-10.1 was downloaded but not evaluated; there is no such result.

## Implementation map

| Component | Files |
| --- | --- |
| Ring activation, attack/channel and checkpoints | [sharp_protocol.py](../basil_core/sharp_protocol.py), [cart_comparison_worker.py](../scripts/cart_comparison_worker.py) |
| Screening and realized-pool calculations | [sharp_math.py](../basil_core/sharp_math.py), [research_protocol.py](../basil_core/research_protocol.py) |
| Study adapters and case selection | [basil_sharp_thesis.py](../basil_core/basil_sharp_thesis.py) |
| Current clean controls | [fedsharp_plain_clean.py](../basil_core/fedsharp_plain_clean.py), [basil_plain_clean.py](../basil_core/basil_plain_clean.py) |
| Official test evaluator | [evaluate_fedsharp_test.py](../scripts/evaluate_fedsharp_test.py) |
| Model checks and single-image inference | [test_fedsharp_image.py](../scripts/test_fedsharp_image.py) |
| Download release models | [download_research_models.py](../scripts/download_research_models.py) |
| Recheck saved official test summaries | [verify_published_test_results.py](../scripts/verify_published_test_results.py) |
| Render official test PNG/SVG | [render_official_test_figures.py](../scripts/render_official_test_figures.py) |
| EuroSAT diagnostic | [evaluate_eurosat_diagnostic.py](../scripts/evaluate_eurosat_diagnostic.py) |

For FedSharp+, follow the evaluated frozen `cart` implementation and installed
adapters. The thesis brief explains probe, admission, prior, repair and commit
stages. Do not substitute an older prototype. FedSharp+ is available only for
non-IID joint attack + noise in this study. There are no FedSharp+ clean or IID
runs. ACDS has extra fitting-data access and must be described accordingly.

## Figures and presentation

The [eight thesis panels](figures/Thesis_Plots/) include FedSharp+ in four
non-IID panels. The [two research-paper strips](figures/Research_Paper_Plots/)
exclude FedSharp+ for the conference version. Both have PNG/SVG copies, legends
below and matched clean controls. Use matching names under `Images/` in
Overleaf. The [figure map](thesis/results-evidence-20261104/figure_map.json) and
revision brief give placement suggestions. The four official-test figures are
additional figures for the separate one-seed test study.

The [presentation draft](presentation/defense-draft-20261104/) contains an
editable PPTX, PDF preview and speaker notes. It predates official testing and
needs a test-results update before it is a final defense deck. The PDF is a
parallel rendering, not an export from PowerPoint.

Original numerical files and evaluation receipts are preserved. Absolute local
paths in old receipts are provenance. Readers do not need to recreate them.
Dataset archives, local environments, credentials and private agent state are
not included. Check SHA-256 manifests when downloading models and results.
