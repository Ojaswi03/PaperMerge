# CART contribution and novelty wording

CART may be presented as the proposed receiver-local mechanism developed and
evaluated in this project. The available evidence does not establish that it is
the first class-aware decentralized defense, that its ingredients are new, or
that the entire method is generally Byzantine robust.

A defensible candidate contribution is the combined use of coherent-model
admission and transactional commitment of a complete local update, guided by
two receiver-private class-recognition probes, in the specified noisy,
Byzantine sequential ring. Whole-model commitment can alter the backbone as
well as the classifier. This positioning is a hypothesis about the combined
design, not an exhaustive literature-priority finding.

The closest mechanisms must be distinguished explicitly:

| Prior work | Established mechanism | Difference to assess for CART |
| --- | --- | --- |
| [BASIL](https://arxiv.org/abs/2109.07706) | Sequential, memory-assisted performance-based selection | CART adds receiver-owned alternatives and a separate check of the complete trained update |
| [Bristle](https://arxiv.org/abs/2110.11006) | Decentralized Byzantine/non-IID model integration using a frozen non-output network and output-layer integration | CART trains the complete model and checks whole-vector commitment; this distinction does not by itself prove novelty |
| [FedFomo](https://openreview.net/pdf?id=ehJqJQk9cw) | Client-local evaluation of useful peer models and personalized model combinations | CART applies its private class score within a specified Byzantine/noisy ring and checks the subsequent local update |
| [Zeno](https://proceedings.mlr.press/v97/xie19b.html) | Suspicion-based ranking of worker gradients using observed loss change | Update screening is prior art; CART's receiver-local ingress/commit sequence is the candidate combined contribution |
| [Logit adjustment](https://openreview.net/forum?id=37nvvqkCo5) | Training-time use of label priors to correct imbalance | The local-prior extension is an adaptation, not an invented loss |
| [BALANCE](https://arxiv.org/abs/2406.10416) | Uses the receiver's own model as a similarity reference for filtering received models | An owned-model alternative alone is not new; CART's private class utility and complete-update commitment are the narrower comparison |
| [DFL-C](https://arxiv.org/html/2608.22552v1) | Common-subset consensus and spatial/temporal trust scoring, aiming for consistent models | CART instead retains distinct receiver-owned models in a sequential ring; it has no consensus or equivocation guarantee |
| [Robust decentralized federated distillation](https://arxiv.org/html/2609.07230v1) | Screens received public-data logits and validates resulting distillation gradients with a supervised private-data gradient | This September 2026 work overlaps with the broad idea of receiver screening followed by private update validation. CART's specific distinction is complete-state transport, two-probe class MCC and finite-scale commitment of a full SGD proposal without a public distillation dataset |

MCC, a quadratic proximal anchor, ridge repair, backtracking, local validation
and sampled parameter perturbations also have established precedents. Making
the peer-candidate fallback consistent with SHARP corrects an implementation
inconsistency; it is not an independent research novelty claim.

Suggested thesis wording:

> This thesis proposes and evaluates CART, a receiver-local class-aware mechanism
> for admitting incoming model states and checking complete local updates in
> decentralized ring training under Byzantine corruption and parameter-space
> communication noise.

The full ACDS/CART comparison measures behavior under matched simulation
conditions. It does not prove originality. Isolating the benefits of ingress,
commitment and class awareness, comparing the closest relevant methods, and
checking the literature for overlapping designs remain necessary for a stronger
novelty claim. The targeted primary-source review supporting this note was
performed on 2026-10-29 and is not an exhaustive systematic review.

## What the contribution claim needs

The latest targeted review found substantial prior art for the general two-stage
idea. Consequently, do not claim the first receiver-local screening-and-validation
defense, or present the number of combined ingredients as evidence of originality.
The narrower parameter-transport contract above is a candidate contribution that
must be discussed explicitly against these approaches. The review establishes
mechanism overlap and differences; it does not certify publication priority.
DFL-Dual (CVPR 2024) also warrants a full-text comparison; its primary PDF could
not be retrieved during this check, so no detailed exclusion claim is made here.

The [two-stage finite-probe inequality](CART_EMPIRICAL_CONTRACT.md) makes the
implemented contract precise. It follows directly from the gates and should not
be advertised as a new convergence or Byzantine theorem. The
[66-case matched study](CART_CONTRIBUTION_HPC.md) tests whether ingress, commitment,
class sensitivity, noise allowance, training priors and head repair contribute.
Negative outcomes must narrow the claim rather than be removed from the thesis.
Local-only and always-single controls are available as optional variants; the
default study is not an exhaustive baseline or factorial evaluation.

The author's completed terminal table reports CART below ACDS on MNIST and above
ACDS on CIFAR-10, including 10% non-IID ACDS outcomes. Those observations do not
establish general superiority, and the realized-pool check provides a stronger
ACDS comparison that must be evaluated before treating the CIFAR gap as decisive.
The thesis should distinguish the original and checked-pooling baseline versions.
Independent checkpoint inference and recomputation from exported arrays remain
separate from the terminal's completion banner.
