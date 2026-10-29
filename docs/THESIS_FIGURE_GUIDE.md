# Thesis figures from completed experiments

All current thesis figures are in `docs/thesis-figures/cart-20261029/`. PDF is
recommended for LaTeX. PNG previews and SVG exports are also included. Figures
use scientific labels rather than campaign identities or validation disclaimers.
The experimental-setup text and captions must still identify validation accuracy
and sample SD across three seeds.

| Thesis purpose | Exact PDF filename in that folder |
| --- | --- |
| MNIST learning curves | `combined_revised_mnist_sigma_0.01_learning_curves.pdf` |
| CIFAR learning curves | `combined_revised_cifar10_sigma_0.01_learning_curves.pdf` |
| Main CIFAR primary endpoint | `combined_revised_cifar10_sigma_0.01_primary_bars.pdf` |
| Optional final-round endpoint / appendix | `combined_revised_cifar10_sigma_0.01_final_bars.pdf` |
| IID per-class recall | `cifar10_iid_class_recall_recomputed.pdf` |
| Non-IID per-class recall | `cifar10_dirichlet_class_recall_recomputed.pdf` |
| Non-IID worst-client outcome | `cifar10_dirichlet_worst_client_recomputed.pdf` |
| Evidence that MNIST partitions differ | `mnist_partition_class_balance.pdf` |

The first seven names match the supplied updated thesis exactly. Copy the PDFs
into the Overleaf project's `figures/` folder. The ring and CART process diagrams
in that thesis are inline TikZ and do not need external image files. The eighth
plot is optional; it shows the distribution of dominant-class fractions across
clients and the three seeds, not independent trial replications.

The MNIST plot retains the original classifier-repair CART. Revised admission/
commitment was run only on CIFAR. Do not imply the same CART version was tested
on both datasets. The CIFAR diagnostic plots explicitly distinguish clean,
SHARP, original head repair and current admission/commitment.

MNIST ownership is genuinely heterogeneous. Across the three seeds, mean
dominant-class fraction is about 11.26% for IID and 37.95% for non-IID; average
class coverage is 10 versus about 9.31 of ten classes per client. Partition hashes
differ. Similar accuracy is not evidence of an identical partition, but neither
does it demonstrate a CART benefit. Its original CART/control effect is nearly
zero on MNIST.

Generate the files again with:

```bash
python scripts/build_cart_thesis_figures.py
```

`manifest.json` hashes every output and the archived inputs. The CSV files retain
seed-level class recall, worst-client values, and partition statistics. This is
array/provenance analysis, not new training or independent model inference.

Two details in the supplied updated thesis need attention: its source table
should name `reporting/cart_v2_campaign.py`, not the nonexistent
`reporting/cart_v2_experiment.py`; and the generated package should not claim
that `audit/EVIDENCE_AUDIT.md` or other accompanying audit files exist unless they
were actually included. The current source/figure evidence is in the repository
exports and this figure folder. Retain the established CART name rather than
silently inventing a new acronym expansion.
