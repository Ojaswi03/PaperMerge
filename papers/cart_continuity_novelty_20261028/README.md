# CART continuity: downloaded primary papers

Downloaded 28 October 2026 using public author/publisher PDFs. These papers were
useful in the Firecrawl-assisted mechanism review. Every PDF is accompanied by
its URL, byte count and SHA-256 in [download_receipts.json](download_receipts.json).
The arXiv author version is distinguished from the venue page where applicable.

| Paper | Local PDF and primary source | Important part to read | What CART must distinguish |
|---|---|---|---|
| FedFomo: Personalized Federated Learning with First Order Model Optimization, ICLR 2021 | [PDF](fedfomo_iclr_2021_arxiv.pdf), [author version](https://arxiv.org/abs/2012.08565) | Section 3, especially Equation 3: local loss improvement relative to the previous client model determines model weights; negative benefits retain the local model | Keeping a previous local model when peers look worse is already prior work. A scalar local-loss gate alone is not our novelty. |
| FedWeIT: Federated Continual Learning with Weighted Inter-client Transfer, ICML 2021 | [PDF](fedweit_icml_2021.pdf), [publisher](https://proceedings.mlr.press/v139/yoon21b.html) | Section 3, parameter decomposition and weighted task-adaptive transfer | Dense shared parameters, sparse task parameters and inter-client transfer already address interference. CART currently trains a complete model rather than this decomposition. |
| FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration at Edge | [PDF](fedknow_2022_arxiv.pdf), [author preprint](https://arxiv.org/abs/2212.01738) | Framework and gradient integration sections: knowledge extractor, gradient restorer, gradient integrator | Remembering model knowledge and avoiding harmful transfer through gradient integration are established ideas. This download is an author preprint, not a claimed independent replication. |
| Federated Continual Learning Goes Online: Uncertainty-Aware Memory Management for Vision Tasks and Beyond, ICLR 2025 | [PDF](uncertainty_aware_memory_iclr_2025.pdf), [venue page](https://iclr.cc/virtual/2025/poster/28896), [author version](https://arxiv.org/abs/2405.18925) | Method section: Bregman Information used to store/retrieve examples in a replay buffer | Predictive-uncertainty sample replay differs from our simulated parameter-channel allowance. Calling any uncertainty-based memory new would be incorrect. |
| The Matthews Correlation Coefficient (MCC) Is More Informative Than Cohen’s Kappa and Brier Score in Binary Classification Assessment, Chicco, Warrens and Jurman, IEEE Access 2021 | [PDF](mcc_kappa_brier_ieee_access_2021.pdf), [DOI](https://doi.org/10.1109/ACCESS.2021.3084050), [institutional PDF](https://pure.rug.nl/ws/portalfiles/portal/173877725/09440903.pdf) | Definition and comparison of MCC and kappa, including imbalanced binary cases | The per-class recognition statistic is established MCC, not a new CART metric. It discounts constant-class guesses but is not generally invariant to prevalence. |
| Why Cohen’s Kappa Should Be Avoided as Performance Measure in Classification, Delgado and Tibau, PLOS ONE 2019 | [PDF](why_cohens_kappa_should_be_avoided_plos_2019.pdf), [publisher](https://doi.org/10.1371/journal.pone.0222916) | Kappa’s behavior under changing class distributions | Avoid claiming that chance correction automatically solves imbalance. The candidate uses MCC and still needs IID/non-IID experiments. |
| Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance, ICML 2019 | [PDF](zeno_icml_2019.pdf), [publisher](https://proceedings.mlr.press/v97/xie19b.html) | Section 3: validation-descent minus update-norm score and ranked update admission | Validation-based update checking already exists. CART’s proposed distinction is a joint receiver-local ingress/commit architecture in noisy decentralized training, not the existence of a validation gate. |
| Zeno++: Robust Fully Asynchronous SGD, ICML 2020 | [PDF](zenopp_icml_2020.pdf), [publisher](https://proceedings.mlr.press/v119/xie20c.html) | Server-side validation-gradient comparison for asynchronous updates | Asynchrony and validation-based Byzantine filtering are also prior work; a new distributed setting alone is insufficient evidence of novelty. |

The OpenReview PDF endpoint returned HTTP 403; the public arXiv version was
successfully saved instead. No paywall or access control was bypassed.

The unchanged IEEE Access PDF is attributed above and carries
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) in the PDF.
The PLOS paper carries [CC BY](https://creativecommons.org/licenses/by/4.0/).
Other PDFs retain their original publisher/author notices; no license to modify
or commercially redistribute them is implied.

## Other closest work already in this repository

[Bristle](../022-Bristle%20-%20Decentralized%20Federated%20Learning%20in%20Byzantine%2C%20Non-i.i.d.%20Environments.pdf),
Sections 3.1–3.3, is especially important: it already combines decentralized
Byzantine/non-IID learning and per-class performance-based integration. Its
pretrained frozen backbone and output-layer exchange differ from the proposed
complete-model routing, but that difference alone does not prove novelty or
better performance.

[GEM and GEC](../cart_novelty_audit_20261026/README.md) already constrain remembered
loss/gradient interference. [FedCAD and ACIL](../cart_head_transport_20261027/README.md)
cover class-aware distillation and analytic classifier fitting. These must be
credited when describing the existing CART head repair.

Read the [candidate architecture, equations and falsifiable tests](../../docs/CART_CONTINUITY_CANDIDATE.md)
for the narrow research hypothesis. This targeted review does not establish an
unpublished mechanism, patentability, or a successful new defense.
