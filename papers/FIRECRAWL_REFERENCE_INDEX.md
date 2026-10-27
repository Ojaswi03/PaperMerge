# Downloaded research references

Audited 27 October 2026.

Added **13 original PDFs** in this audit. **66 PDFs** are now indexed across `papers/`, including earlier downloads, alternate paper versions, one supplement, and preserved original files. Existing correct copies were reused.

Scope: papers read or cited during the recorded Firecrawl research, plus explicitly labeled background and filename corrections. Incidental search hits, unrelated results, and papers merely appearing in another paper’s bibliography are not represented as references we used. This index is an archive record, not a certification of CART novelty.

## Added in this audit

| Paper | Local copy | Original source |
| --- | --- | --- |
| Federated Learning with Label Distribution Skew via Logits Calibration | [PDF](firecrawl_reference_audit_20261027/fedlc_2022.pdf) | [Source](https://proceedings.mlr.press/v162/zhang22p/zhang22p.pdf) |
| Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [PDF](firecrawl_reference_audit_20261027/yin_trimmed_mean_2018.pdf) | [Source](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf) |
| The Hidden Vulnerability of Distributed Learning in Byzantium | [PDF](firecrawl_reference_audit_20261027/hidden_vulnerability_2018.pdf) | [Source](https://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf) |
| The Hidden Vulnerability of Distributed Learning in Byzantium - Supplementary Material | [PDF](firecrawl_reference_audit_20261027/hidden_vulnerability_2018_supplement.pdf) | [Source](https://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a-supp.pdf) |
| FlashbackCL: Mitigating Temporal Forgetting in Federated Learning | [PDF](firecrawl_reference_audit_20261027/flashbackcl_2026.pdf) | [Source](https://arxiv.org/pdf/2606.03939v1) |
| Class-Balanced Loss Based on Effective Number of Samples | [PDF](firecrawl_reference_audit_20261027/class_balanced_loss_2019.pdf) | [Source](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) |
| Bayesian Nonparametric Federated Learning of Neural Networks | [PDF](firecrawl_reference_audit_20261027/federated_bayesian_neural_networks_2019.pdf) | [Source](https://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf) |
| Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [PDF](firecrawl_reference_audit_20261027/hsu_nonidentical_distribution_2019.pdf) | [Source](https://arxiv.org/pdf/1909.06335) |
| Byzantine-Robust Decentralized Learning via ClippedGossip | [PDF](firecrawl_reference_audit_20261027/clippedgossip_2022.pdf) | [Source](https://arxiv.org/pdf/2202.01545) |
| Analyzing Federated Learning through an Adversarial Lens | [PDF](firecrawl_reference_audit_20261027/adversarial_lens_2019.pdf) | [Source](https://proceedings.mlr.press/v97/bhagoji19a/bhagoji19a.pdf) |
| Collaborative Machine Learning at the Wireless Edge with Blind Transmitters | [PDF](firecrawl_reference_audit_20261027/blind_transmitters_2020.pdf) | [Source](https://arxiv.org/pdf/1907.03909) |
| Geometric median and robust estimation in Banach spaces | [PDF](firecrawl_reference_audit_20261027/geometric_median_banach_2015.pdf) | [Source](https://arxiv.org/pdf/1308.1334) |
| Federated Learning with Matched Averaging | [PDF](firecrawl_reference_audit_20261027/fedma_2020.pdf) | [Source](https://arxiv.org/pdf/2002.06440) |

FedLC was already cached in `.firecrawl/`; its original PDF is now also saved inside `papers/`. The Hsu paper and FedMA are different papers: their archive titles, URLs and files have been checked separately. The Bayesian neural-network, Hsu and geometric-median entries include additional attribution/background coverage, rather than being labeled as newly discovered CART mechanisms.

## Unavailable original PDF

- **Revisiting Gradient Episodic Memory for Continual Learning**: [OpenReview source](https://openreview.net/forum?id=H1g79ySYvB). Original PDF, attachment and public metadata endpoints returned HTTP 403. [Saved extracted text](cart_novelty_audit_20261026/revisiting_gem_extracted_full_text.md) is available; it is **not an original PDF**. ICLR 2020 submission; acceptance has not been established.

## Existing collections

- [Prototype and retention references](prototype_retention/README.md).
- [Class-aware replay and projection references](cart_related_work/README.md).
- [GEM, GEC, conformal prediction and logit adjustment](cart_novelty_audit_20261026/README.md).
- [FedCAD, ACIL and classifier alignment references](cart_head_transport_20261027/README.md).

FedCAD’s full workshop paper and two-page AAAI student abstract are separate versions. A conference-looking filename or hosting path does not establish acceptance; the Multi-Anchor paper’s acceptance remains unverified. Preprints retain their preprint status.

## Original collection: title corrections

| Original file | Actual paper | Action |
| --- | --- | --- |
| [010-Analyzing Federated Learning Through an Adversarial Lens.pdf](010-Analyzing%20Federated%20Learning%20Through%20an%20Adversarial%20Lens.pdf) | FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping | Filename incorrectly says Analyzing Federated Learning through an Adversarial Lens. The correct Adversarial Lens PDF is in the new audit folder. |
| [011-Geometric Median and Robust Estimation.pdf](011-Geometric%20Median%20and%20Robust%20Estimation.pdf) | Distributed optimization over time-varying directed graphs | Filename incorrectly says Geometric Median and Robust Estimation. Kept unchanged. The new Minsker geometric-median paper is separately identified; it does not resolve an unspecified legacy bibliography key. |
| [014-Machine Learning with Adversaries - Byzantine-Tolerant Gradient Descent.pdf](014-Machine%20Learning%20with%20Adversaries%20-%20Byzantine-Tolerant%20Gradient%20Descent.pdf) | Byzantine-Tolerant Machine Learning | Actual title differs from the descriptive filename. |
| [015-Collaborative Machine Learning at the Wireless Edge with Blind Transmitters.pdf](015-Collaborative%20Machine%20Learning%20at%20the%20Wireless%20Edge%20with%20Blind%20Transmitters.pdf) | Fibrations Between Finite Topological Spaces | Unrelated topology paper stored under the Blind Transmitters filename. Excluded from research references; original preserved. The correct wireless paper is in the new audit folder. |
| [024-Knowledge Distillation for Federated Learning - A Practical Guide.pdf](024-Knowledge%20Distillation%20for%20Federated%20Learning%20-%20A%20Practical%20Guide.pdf) | Knowledge Distillation in Federated Learning: A Practical Guide | Actual title uses in rather than for. |

The original `005` wireless channel-estimation paper was checked and matches its title. It is `015` that contains the unrelated topology paper.

## Complete local PDF index

| Paper | Copy | Pages | Source |
| --- | --- | --- | --- |
| Basil A Fast and Byzantine-Resilient Approach for Decentralized Training | [PDF](001-Basil%20A%20Fast%20and%20Byzantine-Resilient%20Approach%20for%20Decentralized%20Training.pdf) | 23 | [Source](https://arxiv.org/abs/2109.07706v3) |
| Robust Federated Learning with Noisy Communication | [PDF](002-Robust%20Federated%20Learning%20with%20Noisy%20Communication.pdf) | 12 | [Source](https://arxiv.org/abs/1911.00251v1) |
| Decentralized Federated Learning: Balancing Communication and Computing Costs | [PDF](003-Decentralized%20Federated%20Learning%20-%20Balancing%20Communication%20and%20Computing%20Costs.pdf) | 24 | [Source](https://arxiv.org/abs/2107.12048v4) |
| Decentralized Federated Learning with Unreliable Communications | [PDF](004-Decentralized%20Federated%20Learning%20with%20Unreliable%20Communications.pdf) | 13 | [Source](https://arxiv.org/abs/2108.02397v1) |
| Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation | [PDF](005-Robust%20Federated%20Learning%20for%20Wireless%20Networks%20-%20A%20Demonstration%20with%20Channel%20Estimation.pdf) | 6 | [Source](https://arxiv.org/abs/2404.03088v2) |
| pFedGame: Decentralized Federated Learning Using Game Theory in Dynamic Topology | [PDF](006-pFedGame%20-%20Decentralized%20Federated%20Learning%20Using%20Game%20Theory%20in%20Dynamic%20Topology.pdf) | 5 | [Source](https://arxiv.org/abs/2410.04058v1) |
| Byzantine-Robust Decentralized Federated Learning | [PDF](007-Byzantine-Robust%20Decentralized%20Federated%20Learning.pdf) | 18 | [Source](https://arxiv.org/abs/2406.10416v4) |
| Adaptive Federated Optimization | [PDF](008-Adaptive%20Federated%20Optimization.pdf) | 38 | [Source](https://arxiv.org/abs/2003.00295v5) |
| Fall of Empires: Breaking Byzantine-Tolerant SGD | [PDF](009-Fall%20of%20Empires%20-%20Breaking%20Byzantine-Tolerant%20SGD.pdf) | 10 | [Source](https://arxiv.org/abs/1903.03936v1) |
| FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping | [PDF](010-Analyzing%20Federated%20Learning%20Through%20an%20Adversarial%20Lens.pdf) | 18 | [Source](https://arxiv.org/abs/2012.13995v3) |
| Distributed optimization over time-varying directed graphs | [PDF](011-Geometric%20Median%20and%20Robust%20Estimation.pdf) | 15 | [Source](https://arxiv.org/abs/1303.2289v2) |
| Local SGD Converges Fast and Communicates Little | [PDF](012-Local%20SGD%20Converges%20Fast%20and%20Communicates%20Little.pdf) | 19 | [Source](https://arxiv.org/abs/1805.09767v3) |
| Byzantine-Robust Learning on Heterogeneous Data | [PDF](013-Byzantine-Robust%20Learning%20on%20Heterogeneous%20Data.pdf) | 22 | [Source](https://arxiv.org/abs/2302.06079v2) |
| Byzantine-Tolerant Machine Learning | [PDF](014-Machine%20Learning%20with%20Adversaries%20-%20Byzantine-Tolerant%20Gradient%20Descent.pdf) | 16 | [Source](https://arxiv.org/abs/1703.02757v1) |
| Fibrations Between Finite Topological Spaces **(unrelated preserved original)** | [PDF](015-Collaborative%20Machine%20Learning%20at%20the%20Wireless%20Edge%20with%20Blind%20Transmitters.pdf) | 33 | [Source](https://arxiv.org/abs/1907.03972v1) |
| Communication-Efficient Learning of Deep Networks from Decentralized Data | [PDF](016-Communication-Efficient%20Learning%20of%20Deep%20Networks%20from%20Decentralized%20Data.pdf) | 10 | Source not independently recovered in this audit; see PDF/collection receipt |
| Federated Learning with Non-IID Data | [PDF](017-Federated%20Learning%20with%20Non-IID%20Data.pdf) | 12 | [Source](https://arxiv.org/abs/1806.00582v2) |
| Federated Optimization in Heterogeneous Networks | [PDF](018-Federated%20Optimization%20in%20Heterogeneous%20Networks.pdf) | 22 | [Source](https://arxiv.org/abs/1812.06127v5) |
| SCAFFOLD: Stochastic Controlled Averaging for Federated Learning | [PDF](019-SCAFFOLD%20-%20Stochastic%20Controlled%20Averaging%20for%20Federated%20Learning.pdf) | 41 | [Source](https://arxiv.org/abs/1910.06378v4) |
| Fedisp: An Incremental Subgradient-Proximal-Based Ring-Type Architecture for Decentralized Federated Learning | [PDF](020-Fedisp%20-%20An%20Incremental%20Subgradient-Proximal-Based%20Ring-Type%20Architecture%20for%20Decentralized%20Federated%20Learning.pdf) | 16 | [Source](https://doi.org/10.1007/s40747-023-01272-4) |
| Byzantine-Robust Federated Learning over Ring-All-Reduce Distributed Computing | [PDF](021-Byzantine-Robust%20Federated%20Learning%20over%20Ring-All-Reduce%20Distributed%20Computing.pdf) | 5 | [Source](https://arxiv.org/abs/2501.17392v2) |
| Bristle: Decentralized Federated Learning in Byzantine, Non-i.i.d. Environments | [PDF](022-Bristle%20-%20Decentralized%20Federated%20Learning%20in%20Byzantine%2C%20Non-i.i.d.%20Environments.pdf) | 15 | [Source](https://arxiv.org/abs/2110.11006v1) |
| Decentralized Federated Learning of Deep Neural Networks on Non-IID Data | [PDF](023-Decentralized%20Federated%20Learning%20of%20Deep%20Neural%20Networks%20on%20Non-IID%20Data.pdf) | 7 | [Source](https://arxiv.org/abs/2107.08517v2) |
| Knowledge Distillation in Federated Learning: A Practical Guide | [PDF](024-Knowledge%20Distillation%20for%20Federated%20Learning%20-%20A%20Practical%20Guide.pdf) | 9 | [Source](https://arxiv.org/abs/2211.04742v2) |
| Decentralized Federated Learning with Non-IID Data: Challenges, Trends, and Future Opportunities | [PDF](025-Decentralized%20Federated%20Learning%20with%20Non-IID%20Data%20-%20Challenges%2C%20Trends%2C%20and%20Future%20Opportunities.pdf) | 41 | [Source](https://doi.org/10.1145/3785657) |
| ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection | [PDF](cart_head_transport_20261027/acil_neurips_2022.pdf) | 13 | [Source](https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf) |
| Class-Wise Adaptive Self Distillation for Heterogeneous Federated Learning | [PDF](cart_head_transport_20261027/class_wise_adaptive_self_distillation_2022_workshop_version.pdf) | 7 | [Source](https://federated-learning.org/fl-aaai-2022/Papers/FL-AAAI-22_paper_43.pdf) |
| Class-Wise Adaptive Self Distillation for Federated Learning on Non-IID Data (Student Abstract) | [PDF](cart_head_transport_20261027/fedcad_aaai_2022_student_abstract.pdf) | 2 | [Source](https://ojs.aaai.org/index.php/AAAI/article/download/21620/21369) |
| Multi-Anchor Distillation with Text-Guided Analytic Classifier for Continual Learning | [PDF](cart_head_transport_20261027/multi_anchor_analytic_classifier_eccv_2026.pdf) | 17 | [Source](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/337.pdf) |
| SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model | [PDF](cart_head_transport_20261027/slca_iccv_2023.pdf) | 12 | [Source](https://arxiv.org/pdf/2303.05118) |
| A Conformal Predictive Measure for Assessing Catastrophic Forgetting | [PDF](cart_novelty_audit_20261026/conformal_predictive_measure_forgetting_2025.pdf) | 6 | [Source](https://arxiv.org/pdf/2505.10677v1) |
| Model Uncertainty Quantification by Conformal Prediction in Continual Learning | [PDF](cart_novelty_audit_20261026/cpcl_model_uncertainty_continual_learning_icml_2025.pdf) | 17 | [Source](https://raw.githubusercontent.com/mlresearch/v267/main/assets/gao25i/gao25i.pdf) |
| Gradient-Guided Epsilon Constraint Method for Online Continual Learning | [PDF](cart_novelty_audit_20261026/gec_epsilon_constraint_neurips_2025.pdf) | 30 | [Source](https://proceedings.neurips.cc/paper_files/paper/2025/file/b3c2854d9e94282a373d8fa58b567b27-Paper-Conference.pdf) |
| Gradient Episodic Memory for Continual Learning | [PDF](cart_novelty_audit_20261026/gradient_episodic_memory_neurips_2017.pdf) | 10 | [Source](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) |
| Long-tail learning via logit adjustment | [PDF](cart_novelty_audit_20261026/menon_logit_adjustment_iclr_2021.pdf) | 27 | [Source](https://arxiv.org/pdf/2007.07314v2) |
| A Tutorial on Conformal Prediction | [PDF](cart_novelty_audit_20261026/shafer_vovk_conformal_prediction_jmlr_2008.pdf) | 51 | [Source](https://www.jmlr.org/papers/volume9/shafer08a/shafer08a.pdf) |
| Visual Language Model based Cross-modal Semantic Communication Systems | [PDF](cart_novelty_audit_20261026/vlm_cross_modal_semantic_communication_2024.pdf) | 12 | [Source](https://arxiv.org/pdf/2407.00020v1) |
| C^2Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning | [PDF](cart_related_work/c2prompt_2025.pdf) | 31 | [Source](https://arxiv.org/pdf/2509.19674v2) |
| Class-wise Balancing Data Replay for Federated Class-Incremental Learning | [PDF](cart_related_work/classwise_balanced_replay_2025.pdf) | 26 | [Source](https://papers.nips.cc/paper_files/paper/2025/file/d611d06e3207330555fbc10810e70163-Paper-Conference.pdf) |
| Dark Experience for General Continual Learning: a Strong, Simple Baseline | [PDF](cart_related_work/dark_experience_replay_2020.pdf) | 11 | [Source](https://proceedings.neurips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) |
| FedProTIP: Task-Agnostic Federated Continual Learning via Replay-Free Gradient Projection | [PDF](cart_related_work/fedprotip_2025.pdf) | 34 | [Source](https://arxiv.org/pdf/2509.21606) |
| Analyzing Federated Learning through an Adversarial Lens | [PDF](firecrawl_reference_audit_20261027/adversarial_lens_2019.pdf) | 10 | [Source](https://proceedings.mlr.press/v97/bhagoji19a/bhagoji19a.pdf) |
| Collaborative Machine Learning at the Wireless Edge with Blind Transmitters | [PDF](firecrawl_reference_audit_20261027/blind_transmitters_2020.pdf) | 5 | [Source](https://arxiv.org/pdf/1907.03909) |
| Class-Balanced Loss Based on Effective Number of Samples | [PDF](firecrawl_reference_audit_20261027/class_balanced_loss_2019.pdf) | 10 | [Source](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) |
| Byzantine-Robust Decentralized Learning via ClippedGossip | [PDF](firecrawl_reference_audit_20261027/clippedgossip_2022.pdf) | 37 | [Source](https://arxiv.org/pdf/2202.01545) |
| Bayesian Nonparametric Federated Learning of Neural Networks | [PDF](firecrawl_reference_audit_20261027/federated_bayesian_neural_networks_2019.pdf) | 10 | [Source](https://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf) |
| Federated Learning with Label Distribution Skew via Logits Calibration | [PDF](firecrawl_reference_audit_20261027/fedlc_2022.pdf) | 19 | [Source](https://proceedings.mlr.press/v162/zhang22p/zhang22p.pdf) |
| Federated Learning with Matched Averaging | [PDF](firecrawl_reference_audit_20261027/fedma_2020.pdf) | 16 | [Source](https://arxiv.org/pdf/2002.06440) |
| FlashbackCL: Mitigating Temporal Forgetting in Federated Learning | [PDF](firecrawl_reference_audit_20261027/flashbackcl_2026.pdf) | 9 | [Source](https://arxiv.org/pdf/2606.03939v1) |
| Geometric median and robust estimation in Banach spaces | [PDF](firecrawl_reference_audit_20261027/geometric_median_banach_2015.pdf) | 29 | [Source](https://arxiv.org/pdf/1308.1334) |
| The Hidden Vulnerability of Distributed Learning in Byzantium | [PDF](firecrawl_reference_audit_20261027/hidden_vulnerability_2018.pdf) | 10 | [Source](https://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf) |
| The Hidden Vulnerability of Distributed Learning in Byzantium - Supplementary Material | [PDF](firecrawl_reference_audit_20261027/hidden_vulnerability_2018_supplement.pdf) | 3 | [Source](https://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a-supp.pdf) |
| Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [PDF](firecrawl_reference_audit_20261027/hsu_nonidentical_distribution_2019.pdf) | 5 | [Source](https://arxiv.org/pdf/1909.06335) |
| Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [PDF](firecrawl_reference_audit_20261027/yin_trimmed_mean_2018.pdf) | 10 | [Source](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf) |
| Personalized Federated Learning with Feature Alignment and Classifier Collaboration | [PDF](prototype_retention/fedpac_2023.pdf) | 25 | [Source](https://arxiv.org/pdf/2306.11867) |
| FedProK: Trustworthy Federated Class-Incremental Learning via Prototypical Feature Knowledge Transfer | [PDF](prototype_retention/fedprok_2024.pdf) | 10 | [Source](https://openaccess.thecvf.com/content/CVPR2024W/FedVision-2024/papers/Gao_FedProK_Trustworthy_Federated_Class-Incremental_Learning_via_Prototypical_Feature_Knowledge_Transfer_CVPRW_2024_paper.pdf) |
| FedProto: Federated Prototype Learning across Heterogeneous Clients | [PDF](prototype_retention/fedproto_2022.pdf) | 16 | [Source](https://arxiv.org/pdf/2105.00243v4) |
| Asynchronous Federated Continual Learning | [PDF](prototype_retention/fedspace_2023.pdf) | 9 | [Source](https://arxiv.org/pdf/2304.03626) |
| FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning | [PDF](prototype_retention/fedtgp_2024.pdf) | 9 | [Source](https://ojs.aaai.org/index.php/AAAI/article/download/29617/31046) |
| FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning | [PDF](prototype_retention/fetril_2023.pdf) | 10 | [Source](https://openaccess.thecvf.com/content/WACV2023/papers/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.pdf) |
| Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free Techniques to Address Local and Global Forgetting | [PDF](prototype_retention/hybrid_rehearsal_2025.pdf) | 14 | [Source](https://arxiv.org/pdf/2501.15356) |
| Learning without Forgetting | [PDF](prototype_retention/learning_without_forgetting_2016.pdf) | 13 | [Source](https://arxiv.org/pdf/1606.09282) |
| Taming Noise-Induced Prototype Degradation for Privacy-Preserving Personalized Federated Fine-Tuning | [PDF](prototype_retention/noise_prototype_degradation_2026.pdf) | 17 | [Source](https://arxiv.org/pdf/2604.27833v1) |
| Prototype Augmentation and Self-Supervision for Incremental Learning | [PDF](prototype_retention/pass_2021.pdf) | 10 | [Source](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf) |
| Decentralized Federated Learning with Prototype Exchange | [PDF](prototype_retention/pearfl_2025.pdf) | 20 | [Source](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-00237/article_deploy/mathematics-13-00237.pdf) |
| Mitigating Spatio-Temporal Forgetting in Federated Continual Learning with Dynamic-Domain Clients | [PDF](prototype_retention/stp_fcl_2026.pdf) | 23 | [Source](https://mdpi-res.com/d_attachment/sensors/sensors-26-06273/article_deploy/sensors-26-06273.pdf) |

## Verification and provenance

All 66 indexed files passed the PDF header check and `pdfinfo` parsing. First-page text was extracted to verify identity, and every available receipt hash matched the current file. SHA-256 hashes, sizes, page counts and first-page previews are in the [validated PDF inventory](firecrawl_reference_audit_20261027/validated_pdf_inventory.json). Original sources and any failed attempts for new downloads are in the [download receipts](firecrawl_reference_audit_20261027/download_receipts.json); the [manifest](firecrawl_reference_audit_20261027/download_manifest.json) records why each paper was included and the relevant Firecrawl artifact names.

Original user PDFs were preserved. The title corrections above identify the actual contents so a misleading filename is not used as a citation.
