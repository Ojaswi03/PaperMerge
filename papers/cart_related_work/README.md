# Class-aware federated retention: related work

These primary-source papers supplement the existing `prototype_retention/`
library. They are relevant comparisons for CART's proposed retention mechanisms;
their presence or absence does not establish novelty of the complete framework.

| Paper | Relevant mechanism | Difference requiring explicit comparison |
|---|---|---|
| [Dark Experience Replay](dark_experience_replay_2020.pdf) | Historical logit matching, rehearsal and knowledge distillation | General continual learning; a necessary comparison for CART's proposed receiver-local historical prediction targets |
| [C²Prompt](c2prompt_2025.pdf) | Class-aware client knowledge interaction, local distribution compensation and prompt aggregation | Prompt-based federated continual learning; different model and communication protocol from the selected-snapshot ring |
| [FedProTIP](fedprotip_2025.pdf) | Replay-free local gradient projection, server-combined subspaces and task identity prediction | Server aggregation and projection change the optimization rule; they are not supplied by a class registry |
| [FedCBDR](classwise_balanced_replay_2025.pdf) | Class-balanced replay with globally informed sampling and task-aware temperature scaling | Additional replay/global coordination differs from exclusively local TRAIN data and raw receiver-local selection |

The FedProTIP filename retains the 2025 arXiv submission year; the downloaded
v3 paper states TMLR publication in June 2026. C²Prompt is the March 2026 v2 of
its 2025 submission. The exact PDF hashes and resolved download URLs are in
[`download_receipt.json`](download_receipt.json), and bibliographic records are
in [`download_manifest.json`](download_manifest.json) and
[`references.bib`](references.bib).

CART is PaperMerge's proposed Class-Aware Ring Training framework. Its
registry/controller integration should be described separately from BASIL's
ring/selection foundation, proximal regularization, and established replay,
prototype or projection techniques. A defensible contribution requires a precise
mechanism, explicit data/trust assumptions, and controlled empirical evidence.
This targeted collection is not an exhaustive novelty search or a robustness
guarantee. No training method or experiment was enabled by adding these files.
