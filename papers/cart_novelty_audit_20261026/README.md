# CART loss-memory novelty comparison

Review date: 26 October 2026. This is a targeted mechanism comparison, not an
exhaustive novelty certification. The official proceedings PDF is stored here;
its source URL, file size and SHA-256 are in `download_receipt.json`.

Additional useful sources are indexed in `additional_download_receipts.json`:
Shafer–Vovk's JMLR tutorial supplies established exchangeability/calibration
background; CPCL (ICML 2025) studies prediction uncertainty under continual
learning; CPCF (arXiv 2025) measures forgetting using conformal confidence.
These works prevent attributing conformal calibration or uncertainty-based
forgetting assessment itself to CART. Original PDFs are saved here.

27 October additions, with original PDFs and download receipts:

- **Gradient-Guided Epsilon Constraint Method for Online Continual Learning**,
  Lai et al., NeurIPS 2025. This is a particularly close recent comparator:
  explicit remembered-loss slack and gradient-dependent control of stability
  versus plasticity already exist. See Sections 3–5 in the downloaded paper and
  its [official proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/b3c2854d9e94282a373d8fa58b567b27-Paper-Conference.pdf).
  CART cannot claim loss constraints, allowable forgetting or dynamic retention
  control in general as new. Proposed narrower distinctions are calibration
  under the modeled parameter channel, supported class constraints at a
  receiver-owned coherent reference, and measured nonlinear restoration at
  reception and optimizer boundaries. Their novelty remains unestablished.
- **Long-tail learning via logit adjustment**, Menon et al., ICLR 2021.
  Section 5.1, Equation 10 establishes training logits plus log class priors
  and ordinary raw-logit inference. Our local floor-one prior is an adaptation;
  a matched baseline gain from it is not an original CART contribution.
- **Visual Language Model based Cross-modal Semantic Communication Systems**,
  Jiang et al., arXiv 2407.00020v1. It combines channel impairments and
  memory-assisted continual learning for semantic communication. This rules
  out a broad claim that combining channel noise and memory-assisted retention
  is itself unprecedented. Its transmitted semantic features and continual
  encoder/decoder setting differ from channel-corrupted ring model snapshots.

CPCL's Section 4 uses sequential calibration scores and quantile regression
forests to quantify prediction uncertainty. CPCF measures forgetting using
conformal prediction confidence and evaluates conventional CE/EWC training.
Neither inspected mechanism is the present receiver's parameter-channel loss
budget, but these targeted comparisons do not prove absence of another match.

The OpenReview follow-up *Revisiting Gradient Episodic Memory for Continual
Learning* is an ICLR 2020 submission whose acceptance was not established. Its
public PDF endpoints return HTTP 403 in this environment. The available full
text was saved as `revisiting_gem_extracted_full_text.md`, explicitly labeled
as extracted text rather than an original PDF. It discusses representative
memory examples, relaxed gradient constraints, and backward transfer. A
constraint-relaxation claim alone therefore also needs prior-work comparison.

## Gradient Episodic Memory (GEM), NeurIPS 2017

David Lopez-Paz and Marc'Aurelio Ranzato,
[Gradient Episodic Memory for Continual Learning](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning).
Read the official PDF's Section 3, particularly Equations 5–8.

GEM stores observed examples in episodic memories and restricts increases in
their losses. Its Equation 6 constrains each earlier task's memory loss by that
of the preceding predictor. Equations 7–8 use first-order gradient constraints
and project the proposed update onto their feasible set. The paper explicitly
discusses overfitting when simply minimizing losses on stored examples.

The existing CART class-loss barrier stores receiver-local TRAIN probes and
class-specific historical minimum CE targets. It adds a soft hinge penalty when
a supported class's current probe CE exceeds its target plus a fixed slack.
The budgeted variant caps the added correction at 10% of the base CE gradient
norm. It does not solve GEM's quadratic projection problem or enforce GEM's
per-update loss constraints.

The following table describes the historical barrier, not the newer projected
commit candidate.

| Dimension | GEM | Historical CART barrier candidate |
| --- | --- | --- |
| Memory grouping | Earlier tasks | Receiver-local supported classes |
| Reference | Earlier predictor's episodic loss | Best admitted class-probe CE from selected/trained boundaries |
| Update | Gradient projection with constraints | Soft class-loss hinge gradient with a norm budget |
| Training setting | Sequential continual learning tasks | Selected-snapshot decentralized ring, optional attacks and parameter channel noise |
| Guarantee | First-order constraint formulation under stated assumptions | No demonstrated non-forgetting or Byzantine guarantee |

Remembered losses, retention constraints, and gradient modifications are prior
ideas. CART's precise receiver-local class-target policy and its interaction
with BASIL/EBM are candidate distinctions requiring evaluation against relevant
prior methods. A different name, a sum of familiar loss terms, or a new setting
alone does not establish a new algorithmic contribution.

## 27 October: projected commit candidate versus GEC

The newer `coherent_projected_error` candidate now uses gradient projection;
the historical barrier's “does not solve a QP” distinction no longer applies.
GEC Section 5, Equations 9–13, already projects the current task gradient
against an average-memory loss barrier. It explicitly permits per-task
constraints; Appendix D/E describe reference updates, dynamic thresholds and
warm-up. Those ideas cannot be attributed to CART.

| Aspect | GEC in the downloaded paper | Current proposed CART commit repair |
| --- | --- | --- |
| Object being corrected | Current mini-batch gradient | Completed five-epoch activation displacement from a whole receiver reference |
| Constraints | Memory CE/barrier, optionally grouped by task | Supported-class probe error budgets; class CE endpoint gradients are a surrogate direction construction |
| Allowance | Dynamic loss threshold, with a 5% margin in the implementation description | Joint score from 19 modeled Gaussian parameter corruptions of a frozen reference |
| Final acceptance | Gradient-guided update under its assumptions | Explicit actual nonlinear probe-error checks on a finite grid, compared with a scalar counterfactual |
| Setting | Online continual-learning tasks and episodic memory | Receiver-owned state in a selected-snapshot ring with separately modeled corruption and channel noise |

These are concrete implementation differences, not proof of original novelty
or superior learning. A control using the same data and training recipe with
GEM/GEC-style constraints is still needed. The candidate's four-arm GPU smoke
had no qualified protection and supplies no predictive advantage. Its first
full-data three-round Dirichlet pair was negative: projected CART 33.61% versus
control 37.59%. Projection was selected on six of fourteen protected
activations and maintained the measured probe-risk envelope; that invariant
did not translate into higher accuracy. Further seed and partition checks are
pending. The preceding qualified scalar candidate also underperformed its
matched 10-round control (43.34% versus 46.74%). Neither result establishes a
positive CART contribution.

## Scope of a thesis claim

It is accurate to describe CART as the approach proposed and evaluated in this
project. Its algorithmic novelty remains unestablished. Attribute BASIL,
noise-robust objectives, distillation/replay and memory constraints to their
sources. State the precise proposed change and its data/trust assumptions.

The completed three-seed 30-round CART replication underperformed its paired
controls. The later 86.17% clean IID development pilot had CART disabled and
used a new common residual-model/optimizer recipe; it does not demonstrate a
CART improvement. Positive, reproducible ablations are still needed for any
performance claim. A negative result can still be reported accurately.
