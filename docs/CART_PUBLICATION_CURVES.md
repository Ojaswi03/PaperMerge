# Thesis and paper learning curves

The new exporter makes line plots only. It reads all configurations from the
completed comparison plan, verifies their source snapshots, completion records,
data identities and metric arrays, and keeps the original experiment files
unchanged. It also works on older frozen exports without requiring their
producer code to match the current checkout.

## Current full comparison

For 24 runs across MNIST/CIFAR-10, IID/Dirichlet partitions, three seeds and two
methods, the output contains five figure layouts, each in SVG, vector PDF and
600-dpi PNG:

| Filename stem | Recommended use |
| --- | --- |
| `comparison_joint_sigma_0.01_accuracy_curves` | Four-panel paper overview |
| `mnist_joint_sigma_0.01_accuracy_curves` | Thesis MNIST results |
| `cifar10_joint_sigma_0.01_accuracy_curves` | Thesis CIFAR-10 results |
| `mnist_joint_sigma_0.01_seed_curves` | Appendix: all MNIST seeds |
| `cifar10_joint_sigma_0.01_seed_curves` | Appendix: all CIFAR-10 seeds |

Curves first average client-model accuracies within a run, then average across
seeds. The shaded region is sample SD across independent seeds, not a confidence
interval and not variation across the 100 correlated client models. Seed panels
show individual trajectories without uncertainty shading. Every recorded round
is included; values are not smoothed or resampled. The attack marker is at the
first attack-active displayed round, 21 for this full comparison. Y limits match
between the IID and non-IID panels for each dataset and start at zero.

The palette distinguishes CART in blue and ACDS in orange; line styles also
differ for monochrome printing. Fonts remain editable in SVG and embedded in
PDF. Plot labels contain method names and scientific settings, without internal
campaign/version labels or administrative footnotes. Extra-data availability,
aggregation and uncertainty are described in the accompanying `captions.txt`.

## Accuracy measurement

These producers record a development-holdout accuracy trajectory. The correct
y-axis label is **Validation accuracy (%)**. Calling those arrays test accuracy
would change the meaning of the measurement. Final-checkpoint test evaluation
can supply a final-round test endpoint; it cannot supply historical test curves
without the corresponding round-level models or recorded test measurements.
This plotting update does not alter the running training/evaluation protocol.

## Generate automatically when the running configurations finish

This is a separate CPU plotting job. It does not stop training, add GPU workers
or change the frozen producer fingerprints. It waits for every planned run's
metrics to be committed, then generates the figures in a new output directory.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

python scripts/plot_cart_publication.py \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029" \
  --wait --tmux cart-publication-plots
```

The job continues after SSH disconnects. The figures will appear in:

```text
~/Masters_Research/PaperMerge/.papermerge/local-tests/cart-acds-full-100r-20261029/publication_curves/
```

Watch its status with:

```bash
tail -f .papermerge/local-tests/cart-acds-full-100r-20261029/publication_plot.log
```

When all 24 runs finish, the log prints the destination and number of figures.
If a training run fails, the plot job exits without presenting incomplete seed
groups as final results. Resume training, then start the plot job again. A
nonempty figure directory is preserved; use `--destination PATH` for a new
export rather than overwriting it.

After completion, the same command without `--wait --tmux ...` generates plots
immediately in the foreground. There is no need to rerun model training.

The output includes all per-round seed measurements, aggregate means and SDs,
captions, and `learning_curves_manifest.json` with producer, input and output
hashes. The legacy experiment report is left intact; use `publication_curves/`
for the requested line-only thesis/paper figures.
