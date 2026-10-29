# Completed full comparison: terminal observations

The author supplied a terminal report of 24/24 completed configurations, no
failed or pending workers, 240,000 activations and 2:08:18 supervisor elapsed time.
The following means are calculated from the displayed, rounded final-round
accuracies over seeds 2025–2027. They are provisional summaries of that terminal
report, not recomputations from exported metric arrays or primary-endpoint values.

| Dataset / partition | SHARP + ACDS | SHARP + CART | CART minus ACDS |
| --- | ---: | ---: | ---: |
| MNIST IID | 90.25% | 89.28% | −0.97 pp |
| MNIST Dirichlet | 90.34% | 85.23% | −5.11 pp |
| CIFAR-10 IID | 16.70% | 31.24% | +14.54 pp |
| CIFAR-10 Dirichlet | 10.00% | 27.43% | +17.43 pp |

These results oppose a general superiority claim: the direction differs by
dataset. The ACDS comparison uses the original SHARP pool selection, not the
new checked-pooling variant. That stronger comparison and matched component
ablations are prepared in [CART_CONTRIBUTION_HPC.md](CART_CONTRIBUTION_HPC.md).

Before updating thesis tables, independently infer the final checkpoints on HPC
and export the original plan, source snapshots and metric arrays:

```bash
python scripts/verify_cart_comparison_results.py \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029"

python scripts/export_cart_comparison.py \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029" \
  --destination "$PWD/docs/results/cart-acds-full-100r-20261029"
```

The already started publication watcher should have written its line figures
under the original output's `publication_curves/`. Inspect `publication_plot.log`
and that directory. If generation failed, preserve its error and any partial
figures, then use `plot_cart_publication.py` with a fresh `--destination`. These
curves measure development-holdout accuracy; the completion table does not
establish independent benchmark-test results or originality.
