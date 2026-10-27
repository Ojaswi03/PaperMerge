# Classifier transport prior work

Original PDFs downloaded on 27 October 2026. Source URLs, hashes and sizes are
in `download_receipts.json`. The new candidate does not establish novelty by
using ridge regression, logit distillation or classifier-only adaptation.

- **ACIL: Analytic Class-Incremental Learning with Absolute Memorization and
  Privacy Protection**, Zhuang et al., NeurIPS 2022. Establishes analytic/ridge
  classifier updates with accumulated statistics; do not claim closed-form
  class retention as a new idea. The PDF is from official NeurIPS proceedings.
- **Class-Wise Adaptive Self Distillation for Federated Learning on Non-IID Data
  (Student Abstract)**, He et al., AAAI 2022 **Student Abstract and Poster
  Program**, 36(11), 12967–12968. The publisher's two-page PDF is downloaded.
  Its detailed workshop version, **Class-Wise Adaptive Self Distillation for
  Heterogeneous Federated Learning**, is also downloaded. This is FedCAD:
  global-teacher class confidence from auxiliary data controls CE/KD weighting
  during client SGD. Class-wise adaptive distillation is prior work. Current
  CART instead uses receiver-owned reference gaps and a channel allowance to
  fit a post-training classifier residual in current feature coordinates.
  That is a candidate distinction requiring ablations, not certified novelty.
- **SLCA: Slow Learner with Classifier Alignment for Continual Learning on a
  Pre-trained Model**, Zhang et al., ICCV 2023. Addresses representation learning
  and classifier alignment in continual learning. Its pre-trained setting
  differs from our from-scratch ring but does not make classifier alignment new.
- **Multi-Anchor Distillation with Text-Guided Analytic Classifier for Continual
  Learning**, Qier Meng, De Cheng, Jiahao Li and Cheng Deng. The downloaded PDF
  is hosted under an ECCV2026 conference path; publication/acceptance status is
  not independently verified. It combines historical encoder distillation and
  an analytic classifier with CLIP text priors. This rules out a broad novelty
  claim for combining historical-model distillation and analytic classification.

The current proposed CART difference is local class/channel gating of a
classifier residual in the current proposal's feature coordinates, after full
BASIL/SGD training, with measured selection on separate local probes. This is
a candidate distinction requiring close comparisons and effectiveness evidence,
not a certification of novelty. No pretraining, CLIP, text embedding, globally
shared sufficient statistics or public helper dataset is enabled here.
