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
