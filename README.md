# FedSharp and FedSharp+ — PaperMerge

Byzantine-resilient decentralized learning under noisy model exchange. **FedSharp** is the guarded, checked-pooling method previously named SHARP. **FedSharp+** is its complete class-aware extension previously named SHARP + CART. Internal code identifiers remain unchanged so saved source fingerprints and checkpoints stay traceable.

## Latest verified study

The [hal9k export](docs/results/hal9k_basil-sharp-thesis-100r-20261030/) contains **84 completed configurations**, 100 clients, 100 full ring rounds and five mini-batch SGD steps per activation. MNIST and CIFAR-10 each have six IID conditions and eight non-IID conditions, repeated for seeds 2025–2027. Total: **840,000 activations and 4,200,000 optimizer steps**.

All **356 original exported checksums** matched on import. The PNG/SVG-only publication now has 353 checksummed files after removing four duplicate plot PDFs and recording that packaging change. The saved fresh-CPU-inference receipt covers **8,400 final client models**; its maximum per-client discrepancy is one example out of the 1,000-example development holdout. The compact Git export excludes trained checkpoints.

Accuracy below is measured on a fixed, stratified holdout drawn from original TRAIN data. It is **validation accuracy**, averaged across separately evaluated client models, then seeds. It is not benchmark TEST accuracy or the accuracy of an averaged global model.

## Results

Joint Byzantine attack + channel noise, **rounds 91–100**, percentage accuracy: mean ± sample SD across three seeds. Noisy cases use absolute coordinate standard deviation 0.01; non-IID allocation uses concentration 0.5.

| Dataset | Partition | BASIL | FedSharp | BASIL + ACDS | FedSharp+ |
| --- | --- | ---: | ---: | ---: | ---: |
| MNIST | IID | 82.606 ± 1.072 | 88.787 ± 0.364 | — | — |
| MNIST | non-IID | 79.132 ± 1.038 | 83.965 ± 0.845 | 82.228 ± 0.639 | 84.638 ± 0.446 |
| CIFAR-10 | IID | 29.910 ± 0.578 | 36.315 ± 0.420 | — | — |
| CIFAR-10 | non-IID | 20.694 ± 0.187 | 19.048 ± 0.508 | 29.759 ± 0.419 | 27.746 ± 0.647 |

ACDS shares 5% of fitting examples and changes client data access. FedSharp+ includes the complete declared training-prior, class-aware incoming-model and full-update checks. The extra methods were evaluated only for non-IID data. Sample SD is not a confidence interval; these results do not establish universal superiority. Clean and attack-only curves nearly coincide, while some noisy FedSharp cases outperform their clean controls. Noise-dependent regularization also changes, so this does not isolate a beneficial effect of noise.

- [Full results review, including all clean/attack/joint conditions](docs/FEDSHARP_RESULTS_REVIEW_20261030.md)
- [All seed-level endpoints](docs/thesis-figures/fedsharp-hal9k-20261030/seed_endpoints.csv)
- [All means and sample SDs](docs/thesis-figures/fedsharp-hal9k-20261030/summary_endpoints.csv)
- [Every recorded round and seed](docs/thesis-figures/fedsharp-hal9k-20261030/accuracy_by_round.csv)

## Thesis figures

The four main **vector SVGs** contain 28 condition curves across all 84 runs. Lines are unsmoothed three-seed means, bands are sample SD, and the dotted line marks attack onset at displayed round 21. MNIST includes a labelled late-round inset. Fonts are outlined for portable SVG rendering. PNG versions use 600 DPI. No plot PDFs are published.

| Dataset | IID | non-IID |
| --- | --- | --- |
| MNIST | [SVG](docs/thesis-figures/fedsharp-hal9k-20261030/mnist_iid_accuracy.svg) | [SVG](docs/thesis-figures/fedsharp-hal9k-20261030/mnist_non_iid_accuracy.svg) |
| CIFAR-10 | [SVG](docs/thesis-figures/fedsharp-hal9k-20261030/cifar10_iid_accuracy.svg) | [SVG](docs/thesis-figures/fedsharp-hal9k-20261030/cifar10_non_iid_accuracy.svg) |

The thesis uses these filenames under `Images/`. Per-class recall, honest-client accuracy and client-level metrics remain in the verified arrays for additional diagnostics. They are not discarded when older generated figures are removed.

## Environment and input checks

For an existing HPC checkout:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
nvidia-smi
free -h
python scripts/check_basil_sharp_inputs.py
```

For a fresh environment, use `bash scripts/setup_hpc.sh`; dependency pins live under `environment/requirements*.txt`. Original MNIST and CIFAR-10 archives are required. MNIST discovery/download and both datasets’ fingerprints are checked by the input preflight. CIFAR-10 uses the five original training batches under the Keras dataset directory. Normalized shared-cache files are not used by this study.

## Prepare and run a new campaign

Use a fresh output directory on the producing server’s local disk. This prepares the same 84-case scientific matrix without starting training:

```bash
thesis_run="/var/tmp/papermerge-$(id -u)/$(hostname -s)_fedsharp-100r-$(date +%Y%m%d-%H%M%S)"

python scripts/run_basil_sharp_thesis.py \
  --gpu-count 2 \
  --ram-mib-per-worker 8192 --ram-headroom-mib 16384 \
  --output "$thesis_run"
```

Use `--gpu-count 4` for four available GPUs on themachine. Hal9k has two GPUs and about 93 GiB host RAM; the conservative reservations normally permit roughly nine total workers across both devices. Device capacity alone does not authorize extra host-RAM allocations.

Start the prepared plan through the process-memory guard. The per-device execution gates run before research workers start:

```bash
nohup python -u scripts/resume_basil_sharp_resilient.py \
  --resume --execute --wait-for-resources \
  --output "$thesis_run" --ntfy-topic papermerge-ojaswi \
  > "$thesis_run/adaptive_supervisor.log" 2>&1 < /dev/null &
echo $! > "$thesis_run/supervisor.pid"
```

After GPU verification passes and workers are running, enable the shared GPU queue on this server:

```bash
python scripts/rebalance_basil_sharp_live.py \
  --output "$thesis_run" --execute --background \
  --ntfy-topic papermerge-ojaswi
tail -f "$thesis_run/live_scheduler.log"
```

The handover adopts active workers without restarting them. New configurations prefer healthy available GPUs under live host/device memory checks. Failed workers are set aside; CUDA OOM/device failures quarantine the affected GPU and notify through ntfy. Existing checkpointed work retains its original resource contract. [Operational details](docs/BASIL_SHARP_LIVE_BALANCING.md).

**Multi-server work stealing is not implemented yet.** Each current campaign is bound to one host. Do not run two supervisors against the same output. The proposed Server 0/themachine and Server 1/hal9k shared queue is future work.

## Monitor

[Permanent live dashboard](https://ojaswi03.github.io/PaperMerge/?feed=c3e99a29-732b-4bcf-a42c-a4a19410d4ee) — one consistent read-only link. The dark dashboard has Running, Pending, Completed, Failed/stopped and All tabs. It polls every second; worker-derived ETA values change when new measurements arrive.

```bash
# Current completed hal9k campaign
thesis_run="/var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030"

python scripts/run_cart_comparison.py --monitor --output "$thesis_run"
```

Ctrl+C closes only the terminal monitor. To publish/select a campaign for the existing browser link:

```bash
python scripts/share_cart_monitor.py \
  --output "$thesis_run" --background --interval 1 \
  --ntfy-topic papermerge-ojaswi
python scripts/share_cart_monitor.py --print-link
```

Publishers and detached workers survive SSH logout. Private feed write keys stay outside Git. [Monitor guide](docs/CART_BROWSER_MONITOR.md).

## Validate, export and pull results

On the producing server, after every configuration completes:

```bash
export_dir="docs/results/hal9k_basil-sharp-thesis-100r-20261030"

python scripts/verify_basil_sharp_thesis_results.py --output "$thesis_run" && \
python scripts/export_basil_sharp_thesis.py \
  --output "$thesis_run" --destination "$PWD/$export_dir" && \
python scripts/package_plot_formats.py --output "$PWD/$export_dir" && \
git add -- "$export_dir" && \
git commit -m "Add verified FedSharp results" && \
git push origin main
```

The current export is already published. Do not export over its packaged files. Use a new explicitly reviewed export name for a genuinely new campaign and update the curated allowlist in `.gitignore`. Fresh CPU inference verifies saved checkpoints; it does not retrain. Preserve original checkpoints on the producing server.

Locally, pull and regenerate reviewed figures into an empty destination:

```bash
git pull --ff-only

python scripts/review_fedsharp_thesis_results.py \
  --output docs/results/hal9k_basil-sharp-thesis-100r-20261030 \
  --destination /tmp/fedsharp-thesis-review-new
```

The reviewer checks the archive, metric coverage and independent-inference receipt before making plots. Inspect the generated PNG previews locally; publish the reviewed PNGs, SVGs and numeric tables. Scientific run records and source snapshots remain unchanged. Duplicate plot PDFs were removed; the packaging change is recorded in each export, with updated checksums.

## Checks and repository contents

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_fedsharp_publication.py \
  tests/test_live_gpu_scheduler.py tests/test_cart_adaptive.py \
  tests/test_stable_monitor.py tests/test_cart_public_monitor.py
node web/cart-monitor-site/scripts/test-worker.mjs
python scripts/check_publication_policy.py
```

Publish only source, tests, dependency specifications, current documentation, reviewed figures and compact verified evidence. Agent/editor settings, environments, datasets, caches, credentials, queue state, logs and checkpoints are local-only. Ignoring a file does not remove an already tracked copy; publication hooks also reject those runtime paths.

The older [66-run ablation evidence](docs/results/cart-contribution-100r-20261029/) and [its review](docs/CART_RESULTS_REVIEW_20261029.md) remain because they support thesis component comparisons. They are a separate protocol version and must not replace the current 84-case results. Superseded pilots, duplicate figures and generated legacy runtime outputs have been removed without creating an archive.

Reference PDFs and [their source index](papers/FIRECRAWL_REFERENCE_INDEX.md), the original algorithm specification, required source dependencies and reusable GUI presets remain for research/reproducibility.
