# Review of the completed HPC comparison

The completed 66-configuration export was pulled from GitHub commit `000d5f17`.
All 347 exported file checksums match. Each configuration completed 100 rounds
with 100 clients and five mini-batch steps per activation: 660,000 activations
and 3,300,000 optimizer steps overall. The saved independent CPU-inference
receipt covers all 6,600 final client models. Three configurations have a maximum
client discrepancy of one image on the 1,000-image validation holdout, within
the predefined verification tolerance; the others have zero discrepancy.

## Accuracy from the saved arrays

Values below are percentages, mean ± sample SD across three seeds. The primary
measure averages all client models over rounds 91–100. These values are
recomputed from arrays, not rounded terminal output. Different client models
within one run are not independent seed replicates.

| Dataset | Partition | Method | Rounds 91–100 | Round 100 |
| --- | --- | --- | ---: | ---: |
| MNIST | IID | SHARP + CART | 89.252 ± 0.235 | 89.281 ± 0.196 |
| MNIST | IID | SHARP + ACDS | 88.108 ± 0.144 | 88.010 ± 0.202 |
| MNIST | non-IID | SHARP + CART | 85.239 ± 0.621 | 85.233 ± 0.658 |
| MNIST | non-IID | SHARP + ACDS | 87.531 ± 0.523 | 87.796 ± 0.517 |
| CIFAR-10 | IID | SHARP | 36.315 ± 0.420 | 36.474 ± 0.593 |
| CIFAR-10 | IID | SHARP + CART | 31.105 ± 0.834 | 31.242 ± 0.885 |
| CIFAR-10 | IID | SHARP + ACDS | 36.829 ± 0.229 | 37.025 ± 0.144 |
| CIFAR-10 | non-IID | SHARP | 19.048 ± 0.508 | 19.319 ± 0.782 |
| CIFAR-10 | non-IID | SHARP + CART | 27.382 ± 0.691 | 27.431 ± 0.606 |
| CIFAR-10 | non-IID | SHARP + ACDS | 34.244 ± 0.924 | 34.520 ± 1.058 |

ACDS here is attached to SHARP, not BASIL. Its extra access to shared fitting
examples must be stated when interpreting the comparison. The realized-pool
quality check is used in the SHARP/ACDS controls. The saved full CART method uses
its declared incoming-model and complete-update checks. This study compares
those full procedures; it does not make every difference between them a pure
single-component effect.

## What the additional comparisons show

The version of CART that checks incoming models but does not restrict the
finished local training update reaches CIFAR-10 final accuracies of 37.608% IID
and 33.086% non-IID. Those are higher than full CART's 31.242% and 27.431%.
Conversely, checking only the finished update gives 22.173% IID and 10.000%
non-IID. These observations favor investigating the restriction on the completed
local update, rather than asserting that every added check improves learning.
The simpler incoming-only version is a separate variant and is not relabelled
as the full CART method in the main comparison.

Removing the class-specific protection term gives 31.385% IID and 28.063%
non-IID, providing no observed accuracy advantage for that term at this setting.
Removing the noise allowance reduces full CART's accuracy, while removing the
classifier adjustment has a small observed effect. Removing training-time class
balancing substantially reduces non-IID accuracy. These are three-seed
observations, not general performance guarantees or significance claims.

In full CART's final ten CIFAR-10 non-IID rounds, clients choose their own
previous model in 96.93% of activations. The complete local update is reduced in
79.47% of checked activations and fully rejected in 39.60%; the average retained
scale is 0.327. This behavior is consistent with a restrictive private-data gate
and the nearly flat learning curve. It does not prove a single cause, since
changing the gate changes later models and subsequent decisions.

Class outcomes also require caution. At the final CIFAR-10 non-IID round, the
lowest class recall averaged across clients and then seeds is 7.01% for full
CART, 12.07% for SHARP and 13.73% for SHARP + ACDS. A higher overall accuracy
therefore does not establish preservation of every class.

## Figure review and readiness

New figures use IID/non-IID titles and plain method names. Internal labels for
implementation variants are replaced with descriptions of the change being
tested. Main curves show three-seed means, sample-SD shading, all 100 recorded
rounds and the attack-onset marker. They are line figures, with no bar graphs,
smoothing, internal version titles or test-split banners. Their axis reads
validation accuracy because that is the saved measurement.

The current review figures are readable, but they are **not the professor's
complete BASIL–SHARP figures**: this export lacks genuine BASIL conditions and
the full matched clean/attack-only matrix. They are held for review at
`docs/thesis-figures/cart-review-20261029/`; the seven existing thesis SVGs have
not been replaced. The requested final figures require six IID and eight
non-IID curves per dataset, as specified in
[BASIL_SHARP_THESIS_PLOT_SPEC.md](BASIL_SHARP_THESIS_PLOT_SPEC.md).

The machine-readable review and all seed-level values are stored alongside the
review figures. The original HPC export remains unchanged.
