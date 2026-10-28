# Completed 72-run CART v2 results

All 72 configurations completed: MNIST/CIFAR-10, IID/Dirichlet, three seeds,
six conditions, sigma 0.01, 100 logical clients, 100 rounds, five SGD steps per
activation. The exported results were published by the user in `42da8f81`.

Local verification checks all 171 exported file hashes, recomputes the summary
from all 72 saved metric arrays, matches the frozen source/runtime contracts,
and checks 21 figure-artifact hashes and their metric inputs. The
[verification receipt](validation/CART_HPC_EXPORTED_METRICS_20261028.json)
records these checks. Checkpoints were excluded from the transfer, so no fresh
inference on the HPC-trained models has been performed locally.

## Open the plots

| Dataset | Mean learning curves | Final-round mean bars | Final-ten-round mean bars |
|---|---|---|---|
| CIFAR-10 | [PNG](results/cart-v2-72-100r-20261028/figures/cifar10_sigma_0.01_learning_curves.png) · [PDF](results/cart-v2-72-100r-20261028/figures/cifar10_sigma_0.01_learning_curves.pdf) | [PNG](results/cart-v2-72-100r-20261028/figures/cifar10_sigma_0.01_final_bars.png) | [PNG](results/cart-v2-72-100r-20261028/figures/cifar10_sigma_0.01_primary_bars.png) |
| MNIST | [PNG](results/cart-v2-72-100r-20261028/figures/mnist_sigma_0.01_learning_curves.png) · [PDF](results/cart-v2-72-100r-20261028/figures/mnist_sigma_0.01_learning_curves.pdf) | [PNG](results/cart-v2-72-100r-20261028/figures/mnist_sigma_0.01_final_bars.png) | [PNG](results/cart-v2-72-100r-20261028/figures/mnist_sigma_0.01_primary_bars.png) |

Each image shows IID and non-IID side by side. Curves and bars average three
seeds; shaded regions/error bars are sample SD, not confidence intervals. All
accuracy values use the existing TRAIN holdout; official TEST is unused.
[Per-round averages](results/cart-v2-72-100r-20261028/figures/round_averages.csv)
and [per-seed endpoints](results/cart-v2-72-100r-20261028/figures/seed_metrics.csv)
are available for further analysis.

## Supported conclusions

| Dataset / partition | Clean final | Both mitigations | Both mitigations + CART | CART paired final gain, pp |
|---|---:|---:|---:|---:|
| MNIST IID | 97.41% | 90.82% | 90.85% | +0.027 ± 0.025 |
| MNIST Dirichlet | 97.21% | 89.23% | 89.22% | −0.010 ± 0.017 |
| CIFAR-10 IID | 63.07% | 20.80% | 23.38% | +2.583 ± 3.924 |
| CIFAR-10 Dirichlet | 56.60% | 10.00% | 10.00% | 0.000 ± 0.000 |

The MNIST effects are small and mixed. CIFAR-10 IID improves descriptively,
but the paired seed gains are +7.108, +0.510 and +0.130 percentage points; most
of the mean improvement comes from one seed. Its final-ten-round paired mean
gain is +2.718 ± 3.815 points. The non-IID CIFAR-10 endpoint has no recovery.
These results do not establish the intended near-clean or general CART benefit.

The CIFAR-10 learning problem begins **before** the Hidden attack, which starts
in displayed round 21. Seed 2025 non-IID accuracy at round 20 is 10.146% for
both mitigations and 10.144% with CART; the matched clean model is at 40.440%.
At round 100 the clean model reaches 56.262%, while both protected conditions
are at 10%. Attack-only/noise conditions collapse after onset, whereas screened
conditions were already learning poorly. Attacks cannot explain the entire
pre-onset failure.

No process failure does not imply a successful defense. The small differences
between some final displayed means and the precise paired differences result
from rounding. Manager status can contain an older cached completion count;
all 72 individual completed records and the saved horizon were checked.

## Development decision

Preserve this complete v2 baseline. Compare new CART mechanisms against these
frozen controls only where the underlying protocol/objective remains matched.
A selector or objective change needs its own new control. The
[continuity candidate](CART_CONTINUITY_CANDIDATE.md) is a separate experiment,
with explicit simpler ablations; it is not silently substituted into this study.

The local metric check is a normal script command:

```bash
python scripts/check_cart_export.py \
  --output docs/results/cart-v2-72-100r-20261028
```
