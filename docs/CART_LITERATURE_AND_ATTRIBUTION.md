# CART: literature, attribution and proposed feature-memory extension

Review date: **11 October 2026**. Scope: targeted primary-source review, not an
exhaustive novelty search. Twelve additional papers were downloaded; existing
BASIL, noisy-communication and FedProx PDFs were read without modification.

**Subsequent implementation:** [experimental CART feature-memory module](CART_FEATURE_MEMORY_PROTOCOL.md).
The proposal/status statements below describe the original literature-review
stage. The new module is isolated and its bounded validation is documented
separately; no baseline or production queue was replaced.

## Answer in brief

**Yes: the underlying mechanisms have substantial prior research.** Sharing
class prototypes, rehearsing older classes through features, and constraining
parameter drift are not new ideas. We must credit their sources. This review
did not establish an exact published match for the complete proposed strict
one-class BASIL integration, but that does **not** establish novelty.

**No: feature/prototype memory is not all of CART.** It would be a proposed
retention module within the existing registry, selection, control and telemetry
framework. It is not implemented, approved as a training change, or shown to
solve this repository's constant-class collapse. No experiments were launched
during this review.

## 1. Prior work we should credit

Publication venue matters: CVPR workshop papers below are not main-conference
CVPR papers. A preprint is not presented as a peer-reviewed publication.

| Paper / status | Established mechanism relevant here | Important difference from our strict ring |
|---|---|---|
| [Learning without Forgetting — Li & Hoiem, ECCV 2016; extended journal version](https://arxiv.org/abs/1606.09282) | Retain teacher outputs on new-task inputs through distillation without original task data | A collapsed teacher and a single new class need not provide useful old-class supervision |
| [PASS — Zhu et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html) | Store class prototypes and augment features for old-class decision boundaries, with self-supervision | Centralized incremental learning; its self-supervision would add another training mechanism |
| [FedProto — Tan et al., AAAI 2022](https://arxiv.org/abs/2105.00243) | Exchange class-mean embeddings and align local prototypes with server-aggregated prototypes | Local prototype alignment is not automatically classifier rehearsal for nine absent classes; centralized prototype aggregation |
| [FeTrIL — Petit et al., WACV 2023](https://arxiv.org/abs/2211.13131) | Translate current features using old/new class centroids to create pseudo-old-class features | Feature extractor is fixed after the initial state; freezing our backbone would change the approved training path |
| [FedSpace — Shenaj et al., CVPR Workshops 2023](https://arxiv.org/abs/2304.03626) | Asynchronous federated continual learning with prototypes, representation loss and fractal pretraining | Server coordination, pretraining and modified aggregation differ from our initialization and complete snapshot replacement |
| [FedPAC — Xu et al., ICLR 2023](https://openreview.net/forum?id=SXZr8aDKia) | Feature alignment and collaborative classifiers address client heterogeneity | Personalized classifier combination is not strict handoff and is not the proposed replay mechanism |
| [FedProK — Gao et al., CVPR Workshops 2024](https://arxiv.org/abs/2405.02685) | Federated class-incremental prototype knowledge transfer and pseudo-feature rehearsal | Freezes the feature extractor when new data arrives; server fuses prototypes and uses FedAvg |
| [FedTGP — Zhang et al., AAAI 2024](https://arxiv.org/abs/2401.03230) | Trainable global prototypes with adaptive-margin contrastive learning | Server-trained prototypes and contrastive objectives are different from our local CE, EBM and anchor |
| [PearFL — Qi et al., Mathematics 2025](https://www.mdpi.com/2227-7390/13/2/237) | Peer-to-peer prototype exchange, multi-hop propagation and inter-epoch refresh | Uses decentralized parameter averaging and prototype alignment, not BASIL's selected-model replacement |
| [Hybrid rehearsal — Khademi Nori et al., ICLR 2025](https://openreview.net/forum?id=ydREOIttdC) | Latent exemplars address local forgetting; data-free methods address global forgetting | Includes an autoencoder and reconstruction/global coordination; not just a small classifier feature buffer |
| [STP-FCL — He et al., Sensors 2026](https://www.mdpi.com/1424-8220/26/19/6273) | Prototype replay, contrastive/source-anchor constraints, SAM and projected server aggregation | Dynamic-domain clients and centralized aggregation; feature augmentation noise is not our parameter channel noise |
| [Noise-induced prototype degradation — Wang et al., arXiv April 2026](https://arxiv.org/abs/2604.27833) | Studies privacy-related Gaussian prototype perturbation and prototype-release degradation | LDP clipping/perturbation of embeddings is not absolute Gaussian corruption of transmitted CNN weights; venue unverified |

The STP-FCL downloaded PDF reports **3 October 2026**; its publisher webpage
lists **2 October**. Both precede this review. Preserve that date discrepancy
rather than assigning an unsupported publication chronology.

**Closest reading order:** FedProK → FeTrIL/PASS → FedProto → PearFL → FedSpace.
Recent work strengthens the need for careful attribution; it does not validate
our particular integration. All PDFs, source URLs and download hashes are
listed in the [paper index](../papers/prototype_retention/README.md).

## 2. Exact mechanisms: similar names do not mean the same objective

Let `h_phi(x)` be the encoder and `a_psi(z)` the classifier producing logits.
Then `theta = (phi, psi)` denotes the complete CNN parameters.

### Class prototype

For a class `c` available in a training dataset:

$$
p_c = \frac{1}{|D_c|}\sum_{x\in D_c} h_\phi(x).
$$

A prototype is a mean embedding, not a raw image, model snapshot, gradient or
per-class accuracy. This construction is used by FedProto, FedProK and PearFL.

| Mechanism | What receives supervision? | Why it is not interchangeable |
|---|---|---|
| Prototype alignment | Encoder features for locally observed classes | Can align the current class without teaching the classifier absent classes |
| Feature replay | Classifier predictions on labeled stored/pseudo-features from other classes | Supplies additional labels, but detached features alone do not train the encoder |
| Parameter anchor | Current parameters relative to incoming selected parameters | Limits movement without directly supplying missing-class labels |
| Class registry | Metadata about class support, source, age and performance claims | A historical score is neither a replay example nor a gradient |
| EBM | Sensitivity of the local CE objective to parameter perturbation | Does not explicitly require preservation of older classes |

FedProto's local objective (Eqs. 7–8) adds a weighted distance between local and
global prototypes. In contrast, FedProK's Eq. 8 creates old-class pseudo-features:

$$
\widehat z_{old}=h_\phi(x_{new})+p_{old}-p_{new}.
$$

It sends these features to the classification layer alongside new-class
features. FeTrIL has a related translation construction. **We must cite these
methods if we adopt translation, rather than describe it as an original CART
invention.** Their frozen-representation assumptions cannot be omitted.

FedProK also has a source ambiguity: Eq. 6 is written as cosine similarity,
while Eq. 7 minimizes it and the prose calls the result nearest. An exact
reproduction needs clarification; do not silently copy that base-class rule.
The present proposed one-new-class setting has no multi-new-class choice.

## 3. Attribution map for the complete framework

| Component | Attribution / status | What we can responsibly say |
|---|---|---|
| Logical ring, five predecessor memories, min receiver-local CE selection | [BASIL, Elkordy, Prakash & Avestimehr](https://arxiv.org/abs/2109.07706); original [PDF](../papers/001-Basil%20A%20Fast%20and%20Byzantine-Resilient%20Approach%20for%20Decentralized%20Training.pdf) already present | Uses BASIL's sequential, memory-assisted, performance-based selection; not a claim that BASIL guarantees our extreme split |
| Absolute Gaussian parameter channel and gradient-norm regularization concept | [Ang et al.](https://arxiv.org/abs/1911.00251); original [PDF](../papers/002-Robust%20Federated%20Learning%20with%20Noisy%20Communication.pdf) already present | Multiclass CNN CE is our adaptation of the source concept |
| Proximal parameter-distance penalty | [FedProx / Li et al.](https://arxiv.org/abs/1812.06127); existing [PDF](../papers/018-Federated%20Optimization%20in%20Heterogeneous%20Networks.pdf) | Proximal regularization is established; choosing a BASIL-selected reference is an adaptation, not a new generic penalty |
| Class registry and receiver-supported confidence/gap controller | Current PaperMerge CART revision | Repository-specific design; historical claims are not current globally verified knowledge |
| Proposed class prototype/feature memory | FedProto, PASS, FeTrIL, FedSpace, FedProK and decentralized PearFL as appropriate | Established principles proposed for a new integration, not a proven solution or novelty claim |
| Privacy/noise handling of released features | Separate threat model; recent prototype privacy work is relevant | No claim that feature sharing is private or Byzantine-safe by construction |

BASIL's non-IID ACDS mechanism shares anonymous raw data. We are **not** using
ACDS, and feature sharing must not be misdescribed as its exact reproduction.
FedProx usually anchors to a server reference; our selected received snapshot
is different. The present registry controller is likewise not a BASIL equation.

## 4. Existing CART versus the proposed extension

Current implementation:
[`basil_core/strict_cart.py`](../basil_core/strict_cart.py), revision
`strict_cart_receiver_supported_confidence_gap_v1`.
Its [measured audit](FORGETTING_MECHANISM_AUDIT.md) found that only the current
class had local support. The supported gap stayed zero while previous-class
predictions were lost. Registry metadata alone did not provide their gradients.

```text
EXISTING / IMPLEMENTED
Latest snapshots from five predecessors
                 |
Same receiver-local SS batch -> minimum CE -> choose ONE snapshot
                 |
Complete parameter load; frozen selected reference; reset SGD
                 |
Carried registry claims -> locally supported verification / unavailable flags
                 |
Current CART confidence/gap controller -> epoch-wise anchor strength
                 |
Five full epochs: local CE + optional EBM + selected-snapshot anchor
                 |
Outbound attack -> independent per-link channel noise -> memory refresh

PROPOSED / NOT IMPLEMENTED
Training-side class owners -> bounded feature/prototype records
                 |
Provenance, feature-coordinate compatibility and trust checks
                 |
Labeled older-class feature replay -> ADDITIONAL retention supervision
                 |
Integrate as a separately revisioned CART memory module
```

Class-owner hash-bound evaluations are a separate measurement proposal, not
feature replay. They could reveal retention loss without sharing images, but
scores alone do not generate old-class training gradients. Model hashes bind
a claim to parameters; they do not authenticate its correctness.

## 5. Candidate loss — proposal only

For a separately approved feature-replay revision, a possible decomposition is:

$$
\mathcal L_i(\theta)=
\underbrace{CE_i(\theta)}_{\text{local learning}}
+\underbrace{\sigma_e^2\|\nabla_\theta CE_i(\theta)\|_2^2}_{\text{source-motivated EBM}}
+\underbrace{\frac{\mu_i}{2}\|\theta-\theta_{sel,i}\|_2^2}_{\text{snapshot anchor}}
+\underbrace{\beta\,L_{feature}(\psi;\mathcal M_i)}_{\text{proposed CART retention memory}}.
$$

One simple candidate feature term is class-balanced classifier CE:

$$
L_{feature}=
\frac1{|\mathcal C_{old}|}\sum_{c\in\mathcal C_{old}}
\frac1{|\mathcal M_{i,c}|}\sum_{z\in\mathcal M_{i,c}}
CE(a_\psi(\operatorname{stop\_gradient}(z)),c).
$$

| Symbol | Meaning / unresolved choice |
|---|---|
| `theta` | Current trainable weights **and biases**, evaluated at each SGD step |
| `theta_sel,i` | Exactly the received BASIL-selected snapshot, frozen throughout all five epochs |
| `sigma_e` | Fixed per-coordinate channel standard deviation; model norm is telemetry only |
| `mu_i` | Anchor strength; fixed or the explicitly labeled existing receiver-local controller |
| `M_i,c` | Accepted compatible training-side features or pseudo-features for old class `c`; no test images |
| `C_old` | Available accepted old classes; missing classes must be marked unavailable |
| `beta` | Separate retention weight, **not selected** by this review; not a multiplier on EBM |

This is a design candidate, **not a new implemented loss**. The existing
three-term objective remains unchanged. There is still no extra lambda in EBM.
Adding a retention coefficient does not justify changing the channel variance
or forcing a predetermined gradient multiplier.

Detached feature replay has no direct encoder gradient. It can constrain the
head while the backbone still drifts. Replaying one mean per class also loses
within-class diversity. Neither limitation disappears by adding the registry.
This review therefore does not prescribe a beta, freeze the backbone, add
teacher distillation, or stack all surveyed objectives into one method.

EBM here remains the gradient norm of **local CE**, outside the anchor and
feature term. Applying EBM to `CE + feature replay` instead would introduce
additional Hessian/cross-term effects and require a separately defined mode.

## 6. Keep the source equation qualification explicit

Ang et al.'s Eq. 13 introduces `F_j + sigma_e² ||gradient F_j||²`. Its Eq. 14
uses a Taylor approximation for an expected **squared loss/norm**, not a
general identity for expected multiclass CE under parameter noise. Eq. 15b
updates along the gradient of the modified objective.

Our CNN adaptation is evaluated exactly as defined:

$$
\nabla(CE+\sigma_e^2\|\nabla CE\|^2)
=g+2\sigma_e^2H_{CE}g.
$$

The source's Eq. 23 scalar-gradient expression is not a general nonlinear-CNN
identity. Nested tapes calculate our full second-order correction; no source
convexity/convergence guarantee is transferred to this CNN. Nor is this loss
generally identical to Monte Carlo expected noisy CE. Channel corruption is
a separate operation after the outbound attack.

The feature augmentation in PASS/STP-FCL and the privacy-related prototype
noise in Wang et al. also act on different objects from our transmitted CNN
parameters. Their noise values and robustness claims are not interchangeable.

## 7. Unresolved engineering and scientific conditions

| Issue | Required before implementation / meaningful evidence |
|---|---|
| Changing feature coordinates | Record encoder identity/version; decide how representations stay compatible as weights and channel noise alter the encoder. Same vector length is not compatibility |
| Frozen-backbone assumption in close prior methods | Do not silently freeze our encoder or introduce multiclass pretraining; either would be a protocol change |
| Only one owner per class | Four Byzantine class owners could poison the only direct feature evidence for their classes. Local one-class CE cannot validate other labels |
| Artifact binding | Bind memory to source/model/encoder hashes and age. This limits mix-ups, not malicious claims; a rejected snapshot must not silently supply trusted features |
| Additional communication | Prototype/feature transfer changes the information-sharing protocol even without raw images; define its topology, bandwidth and channel/threat semantics |
| Privacy | Features/prototypes can leak training information. No formal privacy guarantee is claimed, and DP clipping/noise must not be silently enabled |
| Stale evidence | Historical maximum accuracy is not current model retention; track unsupported/stale entries honestly |
| First traversal | Old-class memory may be absent. Never fill it with test features, invented classes or fabricated accuracy |
| Hyperparameter selection | Predeclare a small training-side validation strategy; no test-based beta, mu, sigma or stopping choice |
| Evaluation | Keep full-test curves evaluation-only. Measure old-class retention **and** new-class learning, not merely preservation of a collapsed teacher |

A defensible next step would be a separately approved **clean, bounded**
feature-memory diagnostic first, followed by matched noise conditions only if
it demonstrates interpretable retention. No such run was authorized by this
literature question or started here. This is not approval for an A/B/C/D queue.

## 8. Safe wording for a report

> We investigate a source-grounded integration of BASIL's sequential
> snapshot selection, noise-aware gradient-norm regularization adapted to
> multiclass cross-entropy, and established prototype/feature rehearsal
> principles. PaperMerge's registry and selected-snapshot anchoring coordinate
> these components under an extreme one-class-per-node protocol. The proposed
> feature-memory extension remains to be implemented and evaluated; novelty,
> privacy and Byzantine robustness of that integration are not established.

When writing a paper, cite **the actual adopted mechanism**, not every method
as though implemented. For example, translated-feature replay needs FeTrIL and
FedProK credit; decentralized prototype exchange needs PearFL discussion;
encoder alignment needs FedProto credit. Cite BASIL and Ang et al. separately
for their distinct contributions. Preserve existing hidden-attack provenance:
the repository transform is not claimed to be an exact author-code reproduction.

## 9. Evidence and reproduction

| Check | Evidence |
|---|---|
| Additional downloads | 12/12 successful, 27,294,588 bytes. OpenReview returned HTTP 403 for FedPAC and hybrid rehearsal; their author arXiv PDFs downloaded successfully, with both attempts recorded |
| PDF validation | Every file parsed by `pdfinfo` and `pdftotext`; header/trailer, SHA-256 and byte counts recorded |
| Provenance | [download manifest](../papers/prototype_retention/download_manifest.json), [receipt](../papers/prototype_retention/download_receipt.json), [BibTeX](../papers/prototype_retention/references.bib) |
| Original papers | Before/after: 25 PDFs, 35,302,657 bytes; all 25 SHA-256 hashes matched, zero changed, excluding only the new literature directory |
| Saved research / plot / paper baseline | Rechecked all 23,751 files listed in `gui/worker_state/forgetting_audit_before.json`: 23,751 matched, zero changed; this count includes the original papers |
| Focused unit/import tests | **58 passed, 0 failed, 0 skipped**, 6.85 seconds; two existing TensorFlow dependency deprecation warnings |
| Compile | Download helper and affected test modules compiled successfully |
| Training / queue / W&B | No training, queue execution, protocol changes or W&B runs |

```bash
# No network, data loading or training; prints the paper download plan.
environment/basil-noise-env/bin/python scripts/download_cart_literature.py

# Explicit network action for a clean destination only; refuses overwrites.
# Already executed for this review; do not rerun against existing PDFs.
environment/basil-noise-env/bin/python scripts/download_cart_literature.py --download

# Executed verification; no scientific experiment or network call in these tests.
environment/basil-noise-env/bin/python -m compileall -q scripts/download_cart_literature.py tests/test_cart_literature.py tests/test_lazy_imports.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_literature.py tests/test_lazy_imports.py
```

The reviewed sources do not justify a promised accuracy increase or a claim
that the complete proposed CART integration is new. They do justify crediting
well-established ingredients and testing a clearly differentiated adaptation.
