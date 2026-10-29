# CART training-time local-prior experiment

The balanced-loss commit pilot did not improve the matched 10-round Dirichlet
result: 21.443% final mean client accuracy versus 22.361% for original CART.
Its completed arrays, source snapshots, plots and independent inference receipt
are preserved under `docs/results/cart-balanced-dirichlet-10r-20261029/`.
Less rejection is not an established improvement by itself.

The separate `prior_cart` arm keeps the original MCC ingress, channel allowance,
classifier repair and MCC complete-update gate. Only local training changes.
For local fitting counts n_c, calculate pi_c = max(n_c,1)/sum_k max(n_k,1).
During training, use CE(y, logits + log(pi)) with adjustment strength one.
The EBM surrogate differentiates this adjusted CE, and the selected-snapshot
proximal anchor remains the actual admitted full model. Thus the gradient is
g + 2 lambda H g + mu(theta-anchor), with g and H from adjusted CE.
All five local mini-batch steps, keys and learning-rate decay are retained.

The local count vector is computed from the complete private fitting partition,
not a sampled batch or the global evaluation distribution. Counts never leave
the receiver as training data. Floor-one handling supplies a finite log prior
for absent classes, not missing examples. The nontrainable prior is recomputed
at each activation, so resume does not require persisting another optimizer
state. Snapshot scoring, MCC, classifier repair and model inference use raw
logits; there is no post-hoc inference prior subtraction.

Training-time logit adjustment is prior work, including Menon et al., Long-Tail
Learning via Logit Adjustment, ICLR 2021. This is an integration experiment,
not an invention of class-prior correction, FedLC reproduction, or robustness
guarantee. It also differs from changing the incoming-model trust score.

Validation includes an explicit toy Hessian-vector identity, training-only/raw
inference checks, SGD-budget checks, exact checkpoint recovery, and CPU versus
GPU full local objective comparisons for fresh and receiver-owned routes.
The exploratory 50-round trajectory uses CIFAR-10 Dirichlet alpha 0.5, 100 clients,
seed 2025, absolute sigma 0.01, and Hidden activation at zero-based round 20.
It is one development trajectory, not multi-seed confirmation or a final test.
It does not change the default HPC comparison presets.

The archived 100-round HPC CART controls use TensorFlow 2.21, whereas this PC
uses TensorFlow 2.18. Comparing their first 50 rounds provides context, not a
strictly identical-runtime causal comparison. The matched local 10-round screen
uses attack onset 3 and must not be relabeled as a 50-round onset-20 control.

To start or recover the one-case local experiment:

```bash
bash scripts/launch_cart_local_50r.sh
```

The launcher activates `environment/basil-noise-env`, verifies the GPU and new
objective, uses one worker capped at 6 GiB, and starts a detached tmux session.
It sends the existing self-notification on completion. The HPC launchers and
longer comparison matrices remain separate.

## Completed execution check and current longer run

The three-round, seed-2025 Dirichlet execution check finished without failures,
with 300 activations and 1,500 SGD steps. Fresh final-model CPU inference
reproduced its exported accuracies exactly. Its final mean client accuracy was
19.925%. The earlier ordinary-CART execution check under the same protocol,
initialization, fitting partition and numerical software versions was 11.732%.
The earlier check used a 4 GiB custom GPU profile and the new check a 6 GiB local
profile. This short early-onset-attack comparison is exploratory, not evidence
of a sustained gain or an independent final test. Its archive is
`docs/results/cart-prior-dirichlet-3r-20261029/`.

The local 50-round trajectory was launched separately under
`.papermerge/local-tests/cart-prior-dirichlet-50r-20261029/`. Its terminal status,
complete metrics, and subsequent independent inference must be checked before
reporting its outcome. The onset-20 run is not pooled with the onset-1 check.
The source and checkpoints are frozen separately for each run.

Published pilot SVGs have trailing whitespace normalized; the figure manifests
record the original and normalized hashes. Original local run artifacts remain
unchanged, and export checksums describe the published bytes.
