# Source equations and CIFAR adaptation

Source: `papers/002-Robust Federated Learning with Noisy Communication.pdf`,
printed pages 5–6. Checked against the PDF text, not repository comments.

## Source paper

Equation (14) is the paper's Taylor approximation of a **squared loss quantity**:

\[
\mathbb E\|F_j(w+w_j)\|^2
=\mathbb E\|F_j(w)+w_j\nabla F_j(w)+o(w)\|^2
\approx\mathbb E\|F_j(w)\|^2+\sigma_e^2\|\nabla F_j(w)\|^2.
\]

The paper interprets the second term as the additional training cost determined
by noise. Its notation uses `w_j` here for the perturbation; this is not a
node's trained parameter vector. Equations (13)/(22) define the proposed loss:
\(F_j^e=F_j+\sigma_e^2\|\nabla F_j\|^2\). The squared quantity in (14) and
the unsquared base objective in (13) must not be silently conflated.

Equation (15b) gives the local update:

\[
w_j^{t+1}=w^t-\eta\nabla F_j^e(w^t),\quad j=1,\ldots,N.
\]

Equation (23) is printed as:

\[
\nabla F_j^e(w)
=\nabla F_j(w)+\sigma_e^2\nabla\operatorname{tr}
 (\nabla F_j(w)\nabla F_j(w)^T)
=\nabla F_j(w)+\sigma_e^2\nabla F_j(w)
=(1+\sigma_e^2)\nabla F_j(w).
\]

That last simplification is **not a general identity**: differentiating the
gradient norm normally gives \(2H_F\nabla F\). We reproduce what the source
says without adopting that simplification as neural-network mathematics.
The paper's convexity/convergence discussion is not a guarantee for this CNN.

## Our CIFAR-10 / decentralized adaptation

Our batch base objective is ten-class sparse cross-entropy from logits, not the
source's binary/SVM experiment. We use strict ring handoff, not its centralized
weighted aggregation. `gradient_norm_objective` differentiates
\(CE+c\|\nabla CE\|^2\) using nested TensorFlow tapes. Its update contains
\(\nabla CE+2cH_{CE}\nabla CE\).

The channel is a separate operation, applied **after** outbound attack:

- `paper_absolute_gaussian`: independent coordinate noise of standard deviation
  `channelNoiseSigmaAbsolute`; \(c=\sigma_e^2\).
- `relative_l2_gaussian`: our model-scaled adaptation with
  \(s=\sigma_{rel}\|w\|/\sqrt d\); \(c=stop\_gradient(s^2)\), recalculated
  at the current weights **for every optimizer step**.

The model L2 norm calibrates this second channel; it is not the EBM loss.
`legacy_gradient_scale` keeps historical \((1+\lambda\sigma^2)\nabla CE\)
as a separately labeled compatibility condition, never as the full objective.

## BASIL CNN cross-check

`papers/001-Basil A Fast and Byzantine-Resilient Approach for Decentralized
Training.pdf`, printed page 20, Table II and adjacent text specify convolution
weights `3×16×3×3`, `16×64×4×4`, dense weights `64×384`, `384×192`, `192×10`,
and pooling sizes/strides 3 and 4. Valid convolutions give flatten size 64;
including biases gives 117,706 trainable scalar parameters. The paper specifies
zero biases, Glorot-uniform convolution kernels, and PyTorch-default dense
weights. Our dense kernels use the equivalent fan-in uniform bounds and zero
biases. We emit logits rather than an explicit softmax for stable cross-entropy;
softmax is applied only for inference probabilities.
