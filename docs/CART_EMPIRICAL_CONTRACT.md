# CART's conditional two-stage receiver contract

Fix one activation with established receiver recognition. Hold its two private
probes, supported class set, owned-model targets, qualifying mask and sampled
noise allowances fixed. Let Q be the resulting finite-probe utility, o the
previous owned model, a the admitted model, and c the committed full model.
Assume all scores are finite and repeated inference is deterministic.

The owned model is included among the ingress candidates. Selection within a
δ-band of the maximum gives

\[
Q(a)\geq\max_{z\in\mathcal K} Q(z)-\delta\geq Q(o)-\delta.
\]

The complete-update check tries scales 1, 1/2, 1/4, 1/8 and 0 and accepts a
candidate with utility at least Q(a)−δ. Scale zero returns a itself, so a feasible
candidate exists under the assumptions. Consequently

\[
\boxed{Q(c)\geq Q(o)-2\delta},\qquad \delta=0.01.
\]

The implementation's numerical acceptance tolerance adds at most its stated
floating-point comparison tolerance. This composition is a direct consequence
of the implemented acceptance rules; it is not a new statistical theorem.

The bound concerns the entire parameter vector, including the backbone.
Reference metadata for an uncommitted full proposal is not accepted after a
partial-scale commitment. That transactional rule prevents the classifier
reference from silently describing a different model.

This statement does not apply during cold start, without an owned model, when
fewer than two classes have probe support, or when the receiver gate is inactive.
It is not a bound on benchmark accuracy, every individual class, Byzantine
contamination, or future activations with a changed utility. Adaptively reused
probes are not independent tests. A maximum over 19 Gaussian perturbations is
not a high-confidence upper bound on future channel noise. The inequality cannot
be iterated across changing utilities into a population convergence guarantee.

The implementation mapping is `cart_prior_neutral_routing.selector` for the
owned candidate and ingress band, and `cart_prior_neutral_commit.commit` for
full-update scales. Warm verification exercises informative recognition, the
19-draw branch, scaled commitment, and the composed inequality. Numerical
checkpoint tests separately check that interruption/recovery preserves complete
models and measurements. Neither verification establishes that this contract
causes a useful empirical improvement; the matched ablations test that question.
