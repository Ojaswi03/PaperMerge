# Non-IID BASIL: current loss and A/B/C/D guide

This guide describes the **current 30-round Dirichlet GPU confirmation**, not
the older IID study or the earlier one-class-per-node experiments. It documents
the implemented method; it does not claim that every component is guaranteed
to improve accuracy or provide arbitrary-Byzantine robustness.

**Protocol:** `sequential_basil_dirichlet_v1`

**Revision:** `dirichlet_prior_readiness_diagnostic_v1`

**Configurations:** [`dirichlet_prior_confirmation_30r/`](../gui/configs/nonIID/dirichlet_prior_confirmation_30r/)

**Queue:** [`noniid_dirichlet_prior_abcd_30r_gpu.json`](../gui/queues/noniid_dirichlet_prior_abcd_30r_gpu.json)

## 1. The complete formula

For the active node \(i\), the loss at **each optimizer step** is

$$
\boxed{
\mathcal L_i(\theta;B_i)
=\underbrace{F_i(\theta;B_i)}_{\text{local-prior-adjusted classification}}
+\underbrace{c\,\|\nabla_\theta F_i(\theta;B_i)\|_2^2}_{\text{EBM: D only}}
+\underbrace{\frac{\mu}{2}\|\theta-\theta_{\mathrm{sel},i}\|_2^2}_{\text{snapshot anchor: A/B/C/D}}
}
$$

Here **\(\mu=0.01\)** in all four conditions. The EBM coefficient is
**\(c=0\)** in A/B/C and **\(c=\sigma_e^2=0.0001\)** in D.
There is **no additional lambda**.

The current classification term is not ordinary raw-logit CE. It is

$$
F_i(\theta;B_i)
=\frac{1}{|B_i|}\sum_{(x,y)\in B_i}
\operatorname{CE}\!\left(y,z_\theta(x)+\log\pi_i\right),
\qquad
\pi_{i,c}=\frac{\max(n_{i,c},1)}{\sum_{k=0}^{9}\max(n_{i,k},1)}.
$$

The logarithm is applied elementwise. Its coefficient is fixed at
\(\tau=1\). The implementation subtracts an offset
\(-\log\pi_i\), which is exactly the same as **adding** \(\log\pi_i\)
to the logits before numerically stable cross-entropy.

| Symbol | Meaning |
|---|---|
| \(i,c\) | Active node and CIFAR-10 class |
| \(B_i\) | Current local TRAIN minibatch; augmented for SGD |
| \(n_{i,c}\) | Number of actual TRAIN examples of class \(c\) owned by node \(i\) |
| \(\pi_i\) | Fixed node-local class prior, calculated from TRAIN counts |
| \(z_\theta(x)\) | Ten raw CNN logits, before any training-only prior adjustment |
| \(\theta\) | Current trainable weights and biases, changing after each SGD step |
| \(\theta_{\mathrm{sel},i}\) | Complete received snapshot selected before this activation; frozen for all five epochs |
| \(\mu\) | Snapshot-anchor strength, fixed at 0.01 |
| \(\sigma_e\) | Absolute channel standard deviation per parameter coordinate, 0.010 in C/D |
| \(c\) | EBM coefficient: zero in A/B/C; channel variance 0.0001 in D |

## 2. Exactly what is enabled in A/B/C/D

**BASIL selection, strict handoff, prior adjustment and anchoring are enabled
in every condition.** A is the clean control for this revised method, not an
ordinary-CE/no-anchor ablation.

| Component | A: clean | B: attack | C: attack + noise | D: attack + noise + EBM |
|---|---|---|---|---|
| Dirichlet \(\alpha=0.2\), same partition | Yes | Yes | Yes | Yes |
| BASIL raw-local-CE selection, memory \(S=5\) | On | On | On | On |
| Local-prior-adjusted classification \(F_i\) | On | On | On | On |
| Snapshot anchor, \(\mu=0.01\) | On | On | On | On |
| Actual attacker IDs | None | 0, 1, 5, 7 | 0, 1, 5, 7 | 0, 1, 5, 7 |
| Hidden outbound attack | Off | From R20 | From R20 | From R20 |
| Absolute Gaussian channel | Off | Off | From R0 | From R0 |
| Channel standard deviation \(\sigma_e\) | 0 | 0 | 0.010 | 0.010 |
| EBM coefficient \(c\) | 0 | 0 | **0 despite noise** | **0.0001** |
| EBM mode | `none` | `none` | `none` | `gradient_norm_objective` |
| Local objective identifier | `logit_prior_snapshot_anchor` | `logit_prior_snapshot_anchor` | `logit_prior_snapshot_anchor` | `logit_prior_ebm_snapshot_anchor` |

Therefore the four local losses are

$$
\begin{aligned}
\mathcal L_A&=F_i+0.005\|\theta-\theta_{\mathrm{sel},i}\|_2^2,\\
\mathcal L_B&=F_i+0.005\|\theta-\theta_{\mathrm{sel},i}\|_2^2,\\
\mathcal L_C&=F_i+0.005\|\theta-\theta_{\mathrm{sel},i}\|_2^2,\\
\mathcal L_D&=F_i+0.0001\|\nabla F_i\|_2^2
+0.005\|\theta-\theta_{\mathrm{sel},i}\|_2^2.
\end{aligned}
$$

The identical expression in A/B/C does **not** imply identical training:
their received snapshots and subsequent trajectories differ because of
attacks and channel noise. Every activation gets its own frozen selected
reference; the reference is not one shared constant across the experiment.

## 3. How the terms address non-IID data

The current split is **Dirichlet label skew with \(\alpha=0.2\)**. Nodes own
different class proportions and different numbers of examples. It is no longer
the deterministic one-class-per-node split, nor an equal-sized IID split.

| Node | Actual local TRAIN samples | Batches/epoch | Steps over five epochs |
|---|---:|---:|---:|
| 0 | 2,938 | 6 | 30 |
| 1 | 4,407 | 9 | 45 |
| 2 | 4,768 | 10 | 50 |
| 3 | 989 | 2 | 10 |
| 4 | 9,660 | 19 | 95 |
| 5 | 6,693 | 14 | 70 |
| 6 | 2,945 | 6 | 30 |
| 7 | 5,622 | 11 | 55 |
| 8 | 4,115 | 9 | 45 |
| 9 | 7,863 | 16 | 80 |
| **Total** | **50,000** | **102** | **510 per ring round** |

### Training-only local-prior adjustment

Ordinary local CE can favor the classes a node sees most often. Adding
\(\log\pi_i\) explicitly accounts for local class frequency during fitting.
For a rare class, the added log prior is more negative; correctly fitting its
examples requires a stronger raw-logit margin relative to common classes.
This is intended to reduce majority-class specialization in the learned raw
classifier; it is **not** a guarantee of balanced recall.

The floor of one keeps the logarithm finite for absent classes. It creates
**no synthetic examples**, provides no missing-class images and does not by
itself teach an unseen class. Every node uses only its own counts and data.

**Inference and Snapshot Selection use raw logits, without this adjustment.**

### Frozen snapshot anchor

The anchor penalizes movement away from the selected incoming model:

$$
L_{\mathrm{anchor}}=\frac{0.01}{2}\|\theta-\theta_{\mathrm{sel},i}\|_2^2,
\qquad
\nabla L_{\mathrm{anchor}}=0.01(\theta-\theta_{\mathrm{sel},i}).
$$

At the first step, \(\theta=\theta_{\mathrm{sel},i}\), so the anchor penalty
and its gradient are zero. As training changes the parameters, the anchor
resists drift. The reference stays immutable throughout all five full epochs.

This is a **parameter-preservation mechanism**, not direct supervision on
previous classes. It may reduce forgetting but cannot guarantee it. A poor or
attack-corrupted selected snapshot can also be preserved; the anchor is not an
independent Byzantine filter.

The distance is measured using the **current** parameters at every step, not
only once between the final trained model and the incoming snapshot.

## 4. BASIL handles candidate selection separately from the loss

For each available candidate snapshot \(\theta_j\), the receiver computes

$$
L_{i,j}^{SS}
=\frac{1}{|B_i^{SS}|}\sum_{(x,y)\in B_i^{SS}}
\operatorname{CE}\!\left(y,z_{\theta_j}(x)\right),
\qquad
\theta_{\mathrm{sel},i}=\arg\min_j L_{i,j}^{SS}.
$$

All candidates see the **same deterministic, unaugmented receiver-local TRAIN
batch**, with up to 512 examples. Selection uses ordinary raw-logit CE, not the
prior-adjusted training objective, EBM or the anchor penalty. Sender-reported
metrics and global TEST accuracy never decide the winner.

Losses within the implemented absolute tolerance of \(10^{-8}\) are tied;
the nearest counterclockwise sender wins. The selected model completely
replaces the training start state: **no model averaging**.

Memory holds at most one latest snapshot from each of five distinct
counterclockwise predecessors. Immediately before Node 0 in R11:
`5:R10, 6:R10, 7:R10, 8:R10, 9:R10`. After Node 0 sends its update, Node 1 sees
`6:R10, 7:R10, 8:R10, 9:R10, 0:R11`.

This is the Byzantine-filtering mechanism in A/B/C/D. Under heterogeneous data,
low local CE does not necessarily imply high global accuracy or honesty.

## 5. Hidden attack and channel noise are different operations

### Project Hidden attack: B/C/D

Nodes `[0, 1, 5, 7]` train normally, then corrupt their outbound parameters
starting at **zero-based round 20**. For each parameter tensor \(w_q\):

$$
\begin{aligned}
\delta_q&\sim\mathcal N\!\left(0,
  [0.05\max(\operatorname{std}(w_q),0.01)]^2 I\right),\\
w_{\mathrm{attack},q}
&=0.45w_q+0.55(-1.4w_q+\delta_q)
=-0.32w_q+0.55\delta_q.
\end{aligned}
$$

This is `delayed_hidden_parameter_attack_v1`, the retained **project
adaptation**, not a claim of exact reproduction of the BASIL paper's Hidden
attack. Its random component does not make it the separate Gaussian-replacement
attack. Before R20, attacker-designated nodes send honest trained parameters.

### Actual Gaussian channel: C/D

After any outbound attack, each directed transmission receives independent
channel noise:

$$
\widetilde\theta_{i\to j}=\theta_{\mathrm{out},i}+\xi_{i\to j},
\qquad
\xi_{i\to j}\sim\mathcal N(0,0.0001I_d).
$$

**Standard deviation is 0.010; variance is 0.0001.** Noise starts at R0 and
affects honest as well as Byzantine senders. It is absolute-coordinate noise,
not relative-L2 noise. Model/noise norms are telemetry only and never rescale
the configured channel. C/D use paired keyed underlying Gaussian draws, while
different outgoing links receive independent samples.

The selected anchor is exactly the snapshot the receiver received. If that
snapshot was noisy, the frozen reference includes that received noise; the
receiver is not given an unavailable clean version.

## 6. What EBM changes in D

Let \(g_i=\nabla F_i\) and \(H_i=\nabla^2F_i\). The complete update direction is

$$
\boxed{
\nabla\mathcal L_i
=g_i+2cH_i g_i+0.01(\theta-\theta_{\mathrm{sel},i})
}
$$

In D, this is \(g_i+0.0002H_i g_i+0.01(\theta-\theta_{\mathrm{sel},i})\).
Nested TensorFlow GradientTapes differentiate the gradient-norm penalty without
materializing the Hessian. A/B/C have no Hessian-gradient correction.

EBM penalizes the gradient norm of **prior-adjusted classification loss** in
this revision. It does not penalize the gradient of the entire anchored loss.
There are no EBM/anchor cross terms hidden inside \(\|\nabla F_i\|^2\).

This adapts noise-aware optimization to the current classification objective;
it is not equivalent to scaling the CE gradient by a fixed multiplier. It does
not replace actual channel noise and does not guarantee protection against
structured attacks or forgetting. Legacy lambda-based scaling is inactive.

## 7. One activation, end to end

```text
Five latest predecessor snapshots available in receiver memory
                      |
          Same local TRAIN SS batch, ordinary raw CE
                      |
                 Select ONE model
                      |
       Load ALL weights; freeze selected received reference
                      |
     Reset SGD; train FIVE full local epochs on local data
       A/B/C: prior-adjusted CE + snapshot anchor
       D:     prior-adjusted CE + EBM + snapshot anchor
                      |
            Honest trained model (evaluate only)
                      |
        B/C/D: Hidden if sender is attacker and round >= 20
                      |
        C/D: independent absolute Gaussian noise PER LINK
                      |
           Refresh next FIVE clockwise memories
                      |
          Only immediate successor becomes active next
```

All conditions retain the 117,706-parameter CNN, SGD without momentum,
\(\eta_r=0.05/(1+0.05r)\), five complete local epochs, batch 512 with final
partial batches retained, no gradient clipping and no weight decay. One full
traversal of ten nodes is one round. No class images, replay examples, class
registry, helper gradients or CART mechanism are shared or enabled here.

## 8. What the comparisons can establish

| Comparison | Changed factor | Interpretation within this revised protocol |
|---|---|---|
| B − A | Hidden attack environment | Attack effect while BASIL remains active in both |
| C − B | Actual communication noise | Noise effect with the same local loss |
| D − C | EBM term | EBM effect under the same configured noisy attack environment |
| D − A | Attack, noise and EBM together | Remaining gap to the clean revised-method reference |

These four conditions do **not** independently isolate the anchor or prior
adjustment, because both are present throughout. A positive D−C is a measured
result for this setting, not evidence of universal benefit or statistical
significance from one seed.

The full 10,000-image CIFAR-10 TEST set remains evaluation-only: no test images
enter SGD or Snapshot Selection, and live test metrics do not adapt the learning
rate, coefficients, attacks or stopping rule. Earlier diagnostic comparisons
are exploratory development evidence, not a preregistered confirmatory study.
The current queue is a bounded 30-round confirmation with `researchValid=false`
and `fullResearchReady=false`, not approval for a 500-round campaign.

## 9. Sources, adaptations and implementation map

| Element | Attribution / qualification | Implementation |
|---|---|---|
| Ring, predecessor memory, receiver-local minimum-loss selection | BASIL Eq. (3), minibatch approximation | [`research_protocol.py`](../basil_core/research_protocol.py) |
| Training-side log-prior adjustment | Menon et al., ICLR 2021, Eq. (10); receiver-local priors and floor-one absent counts are declared adaptations | [`calibration_offsets`](../basil_core/dirichlet_calibration.py) |
| Gradient-norm noise-aware penalty | Ang et al.; adapted to adjusted multiclass CE and decentralized training | [`calibrated_anchor_gradients`](../basil_core/dirichlet_calibration.py) |
| Quadratic selected-snapshot anchor | FedProx-style proximal structure; reference changed to the BASIL-selected received model | [`noniid_study.py`](../basil_core/noniid_study.py), [`dirichlet_calibration.py`](../basil_core/dirichlet_calibration.py) |
| Retained delayed Hidden transformation | Explicit project adaptation; not exact-paper-reproduction claim | [`attacks.py`](../basil_core/attacks.py) |
| Approved configuration and evidence checks | Current bounded study, not historical baseline replacement | [`dirichlet_prior_study.py`](../basil_core/dirichlet_prior_study.py) |

The Ang paper's squared-loss approximation must not be presented as an exact
expected CNN cross-entropy identity. Its simplified gradient-scaling expression
is not the general derivative of the implemented gradient-norm objective.
The snapshot anchor is not attributed to BASIL or Ang as their mechanism, and
the current local-prior method is not claimed to be an exact FedLC reproduction.

For source links and detailed distinctions, see
[Dirichlet calibration sources](DIRICHLET_CALIBRATION_SMALL_TESTS.md),
[BASIL/noise/proximal attribution](DIRICHLET_BASIL_PROTOCOL_AND_ATTRIBUTION.md),
and [measured readiness evidence](DIRICHLET_RERUN_READINESS.md).

**Short explanation:** Local-prior-adjusted CE addresses skewed local class
frequencies; the frozen snapshot anchor limits parameter drift; BASIL selects
among received models to filter this attack realization; and EBM in D targets
communication-noise sensitivity. These are separate mechanisms, not guarantees
that one term solves all four problems.
