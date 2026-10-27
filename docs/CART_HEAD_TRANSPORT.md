# CART classifier transport: candidate and evidence

Status, 27 October 2026: implemented opt-in development candidate
`coherent_head_transport`. A positive one-seed five-round development comparison
is measured below; replicated benefit and algorithmic novelty are not established.
This replaces neither the historical reproduction nor its
recorded negative results. The previous projected repair lost 6.75 percentage
points on average across three matched Dirichlet seeds.

## Mechanism

Keep complete BASIL handoff and five full SGD/EBM epochs. Write the completed
proposal as a feature extractor and linear classifier, `z(x)=h(x)W+b`.
Keep its feature extractor fixed during this additional CART operation. A
receiver-owned whole reference supplies historical logits on local TRAIN
probes. No raw examples or extra teacher payload are transmitted.

This addresses two concrete problems with earlier candidates. Pulling back the
whole network can erase useful feature adaptation. Reusing old latent vectors
can also be inconsistent after the feature extractor changes. Here the old
model is evaluated on the same private images as the new model, while the
residual is fitted in the **current** feature coordinates. The encoder remains
fully trainable during ordinary local epochs; only this final classifier solve
holds it fixed. This differs from assuming a permanently frozen encoder or
that old and new latent coordinates coincide.

For each supported class, use up to 16 fitting probes and up to 16 additional,
disjoint selection probes. Both sets were also seen by ordinary SGD; neither
is an independent population validation set. Classes without selection support
are not eligible for a memory correction.

Let `L_p,c` and `L_r,c` be fitting-probe raw CE for proposal and reference.
The reference's stored channel-calibrated allowance is `u_c = b_c - L_r,c`.
Define the bounded gate

```
d_c = max(0, L_p,c - L_r,c)
g_c = d_c / (d_c + u_c + 1e-6)
```

Set `g_c=0` when the reference's measured class recall is at most .1, or the
class has no selection probe. The gate is a heuristic signal/allowance ratio,
not a calibrated probability. The reference allowance uses the existing
19-draw specified Gaussian parameter channel construction; its narrow
reference-corruption coverage does not imply a guarantee for this correction.

For fitting example i, center each model's ten logits by its row mean and form
`E_i = g_(y_i) * (center(z_ref(x_i)) - center(z_proposal(x_i)))`.
Let A contain current proposal features with an appended constant column.
Each supported class has equal total fitting weight: `D_ii=1/(K*n_(y_i))`.
The residual classifier solves

```
Delta* = argmin_Delta ||D^(1/2)(A Delta - E)||_F^2 + lambda ||Delta||_F^2
       = (A^T D A + lambda I)^(-1) A^T D E

lambda = 0.01 * trace(A^T D A) / number_of_columns(A).
```

This lambda belongs to the classifier ridge solve. The common EBM coefficient
remains the channel-coordinate variance; no EBM multiplier is introduced.

The code uses a linear solve, not an explicit inverse. For the current model
this is a 65-by-65 system and a 65-by-10 correction. Zero targets on other
classes discourage unnecessary changes to their fitting logits.

Choose alpha from `{0,.25,.5,.75,1}` using raw macro CE on the separate local
selection probes. Commit only an actual measured improvement of at least
`1e-4`; otherwise retain alpha zero. Add `alpha*Delta*` only to W and b.
Keep a new whole reference when its mean fitting CE improves; there is no
per-class hard constraint or stale-model interpolation at commit.

## What can be proved

1. **Unique ridge solution.** D is positive on fitting samples. The appended
   constant makes `trace(A^T D A)>0`; hence lambda is positive and
   `A^T D A + lambda I` is positive definite. Differentiating the quadratic
   gives the normal equations above. Its regularized fitting objective is no
   greater than that of the zero correction, subject to numerical checks.
2. **Feature preservation.** The correction changes only the final linear
   classifier. All backbone arrays are bitwise identical to the completed
   SGD/EBM proposal. It cannot undo feature learning by interpolating the
   backbone toward an earlier reference.
3. **Measured selection safety.** Alpha zero is an available candidate. The
   code rechecks the actual corrected model, and falls back to zero if local
   selection macro CE increases beyond tolerance. This is an invariant on
   those local TRAIN probes, not on every class or unseen examples.

None of these statements proves higher heldout accuracy, Byzantine tolerance,
global non-forgetting, or convergence. Those require appropriate evidence.

## Attribution and experimental gate

Logit distillation, ridge classifiers and classifier alignment have prior art:
Learning without Forgetting, ACIL, SLCA and newer analytic-classifier studies.
The proposed distinction under investigation is receiver-local, class- and
channel-dependent residual transport after a selected-model ring activation.
An identical prior method has not been ruled out. A label-only classifier
adaptation and an ungated/fixed-allowance transport are needed ablations before
attributing an improvement to this specific CART mechanism.

The immediate experiment uses five complete rounds (`--stage mini`), early
attack onset at round index 1, TRAIN-only heldout evaluation, identical controls,
and sigma .01. It is a short development experiment. Final scope remains both
partitions, three seeds, three noise values and the agreed full condition matrix.

## First completed full-data comparison

Dirichlet alpha .2, seed 2025, absolute sigma .01, five full rounds. Both arms
have Hidden corruption, BASIL and EBM; only the CART mechanism differs. The
primary endpoint is final-round mean accuracy of the ten trained node models
on the balanced 1,000-example TRAIN holdout. Official TEST was not loaded.

| Result | Matched control | CART classifier transport |
| --- | ---: | ---: |
| Final node-mean accuracy | 45.46% | 48.22% |
| Worst-node accuracy | 44.20% | 46.60% |
| Worst-class mean accuracy | 17.90% | 16.80% |
| Runtime | 726.86s | 775.73s |

Final paired mean difference: **+2.76 percentage points**. All ten node
differences are positive (1.2–5.1pp), but those nodes are dependent and do not
constitute ten independent seed replications. Mean trajectory difference is
+1.162pp; round two was slightly below control, so not every round improved.
The worst-class decrease is a limitation, not a non-forgetting guarantee.

Twenty-three of fifty activations selected a nonzero correction. The backbone
was unchanged by each commit repair; the first ten activation model hashes
match the control exactly. Saved-source/config/checkpoint verification passed.
Independent CPU reconstruction of all final node/class accuracies reproduced
both endpoints. Reconstructed predictions and receipts are under the run's
`reports/holdout_recheck/` directory, produced by `scripts/verify_cart_holdout.py`.

Artifacts: `.papermerge/local-tests/cart-head-transport-5r-iid-noniid-v1`.
The clean reference and IID comparisons are still running/pending. This result
does not establish proximity to clean, cross-seed robustness, performance at
other sigmas, a 100-round effect, or original novelty. Those are next tests.
