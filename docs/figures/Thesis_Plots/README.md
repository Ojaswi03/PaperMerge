# Thesis plots for the committee revision

Updated 18 November 2026. These eight validation-accuracy figures include the
completed FedSharp and FedSharp+ noise-only controls. Each figure has one main
plot, no zoom panel, and a legend below. PNG files are 600 DPI; SVG files are
native vector. The filenames match the supplied thesis's `\studyplot` calls.

Each curve has its own color and marker, consistent across figures. BASIL uses
dotted lines for impaired conditions, and BASIL + ACDS uses a dash-dot line.
The two clean controls have exactly identical measured trajectories. A blue
solid line with circles and a thinner dark dashed line with X markers show
their overlap. Marker positions are staggered along the same measured curves;
no values have been shifted, smoothed or jittered.

## Upload to Overleaf

Replace the following sixteen files in the project's `Images/` folder: one
PNG and one SVG for each basename below. Replace both formats because the
thesis chooses SVG or PNG according to compiler settings and file availability.
An older SVG can otherwise take precedence over a newly uploaded PNG.

| Current thesis figure | Panel | Basename, with `.png` and `.svg` |
| --- | --- | --- |
| MNIST IID | Gaussian | `mnist_iid_gaussian_accuracy` |
| MNIST IID | Hidden | `mnist_iid_hidden_accuracy` |
| MNIST non-IID, Figure 4.2 in the supplied PDF | Gaussian | `mnist_non_iid_gaussian_accuracy` |
| MNIST non-IID, Figure 4.2 in the supplied PDF | Hidden | `mnist_non_iid_hidden_accuracy` |
| CIFAR-10 IID | Gaussian | `cifar10_iid_gaussian_accuracy` |
| CIFAR-10 IID | Hidden | `cifar10_iid_hidden_accuracy` |
| CIFAR-10 non-IID | Gaussian | `cifar10_non_iid_gaussian_accuracy` |
| CIFAR-10 non-IID | Hidden | `cifar10_non_iid_hidden_accuracy` |

The existing filenames and figure labels can stay. The PNG fallback is already
supported by the supplied LaTeX. If the stacked panels make the legends too
small, place each panel in its own figure at full text width. That is a layout
recommendation; the supplied thesis has not been edited or recompiled here.

## What the figures measure

Every line is the mean development-holdout accuracy of all 100 retained client
models, averaged across seeds 2025, 2026 and 2027. Shading shows sample SD
across the three seeds at each of the 500 unsmoothed rounds. It is not a
confidence interval or a statistical test.

Noise-only means channel sigma 0.01 and zero attackers. Clean and noise-only
references are repeated in both attack panels. The Gaussian or Hidden context
applies only to attacked curves, which use 33 attackers beginning at round 21.
These are validation curves, not official test-set curves.

Available noise-only curves are BASIL on MNIST, FedSharp on both datasets and
both partitions, and FedSharp+ on both non-IID datasets. There are no BASIL
CIFAR-10 noise-only, BASIL + ACDS noise-only, or FedSharp+ IID results to add.
Missing curves must not be invented. FedSharp+ has no separate clean run.

All eight noise-only method/dataset/partition groups, including the lower
MNIST FedSharp+ outcome, remain visible. Primary endpoints, averaged over
rounds 491 to 500, are:

| Dataset / partition | BASIL | FedSharp | FedSharp+ |
| --- | ---: | ---: | ---: |
| MNIST IID | 73.645% | **88.793%** | -- |
| MNIST non-IID | 71.269% | **85.935%** | 84.442% |
| CIFAR-10 IID | -- | 46.456% | -- |
| CIFAR-10 non-IID | -- | 26.723% | **29.062%** |

Use the [three-seed group summary](../postdefense_results/group_summary.csv)
for full precision and sample SDs. Do not describe FedSharp+ as uniformly
better: base FedSharp is higher on MNIST noise-only, while FedSharp+ is higher
on CIFAR-10 non-IID noise-only. No statistical significance is established.

## Suggested caption additions

Append this shared explanation to the corresponding figure captions, adapting
method availability for the dataset and partition:

> Lines show three-seed mean development-holdout accuracy; shaded bands show
> sample standard deviation. Clean and noise-only controls are repeated in
> both panels. Noise-only uses channel sigma 0.01 with no attackers. The
> vertical line at round 21 marks attack onset for attacked curves only. The
> matched BASIL and FedSharp clean controls coincide exactly and are identified
> by different colors, markers and line styles.

For MNIST IID, mention the BASIL and FedSharp noise-only controls. For MNIST
non-IID, also mention FedSharp+ noise-only. For CIFAR-10 IID, mention FedSharp
noise-only. For CIFAR-10 non-IID, mention FedSharp and FedSharp+ noise-only.

## Related thesis corrections to review

The supplied 18 November LaTeX still describes 150 executions and 50 groups.
The current selection contains 168 executions and 56 groups after adding 18
noise-only runs. The paired clean family records are repetitions of the same
control policy, not extra independent seeds.

The configuration matrix, result descriptions and limitations still say that
FedSharp noise-only and CIFAR-10 noise-only runs are absent. Those statements
need updating. The existing claim that BASIL degradation is "caused by link
noise" is stronger than a matched-trajectory observation alone establishes;
the new controls should be discussed without inventing a causal explanation.

For the committee's other comments:

- Bold the best accuracy within each matched condition. Keep ties bold and
  explain that bold does not mean statistical significance. Do not compare
  clean and attacked rows as though only the environment changed.
- Restate each research question before its answer in Conclusions.
- Frame Future Work as recommendations using "should" where appropriate.
- Move the bibliography before `\appendix`, retaining the existing bibliography
  commands and references.
- Replace red emphasis with `\textbf{...}` while preserving its wording unless
  a separate content correction is approved.

No equation, numerical source result, thesis LaTeX, bibliography or supplied
PDF was modified by generating these figures.
