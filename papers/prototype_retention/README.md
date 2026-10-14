# Prototype / feature-retention literature

Twelve primary-source PDFs downloaded for the CART attribution review on
11 October 2026. **This folder is literature, not experiment output.**
No existing PDF was replaced. No training method was enabled.

Read the [method comparison and framework proposal](../../docs/CART_LITERATURE_AND_ATTRIBUTION.md)
before adapting an algorithm. Prototype alignment, classifier feature replay,
parameter anchoring and EBM are distinct mechanisms.

## Suggested reading order and local PDFs

| Priority / subject | PDF | Primary source / publication status |
|---|---|---|
| 1 — Closest federated retention mechanism | [FedProK](fedprok_2024.pdf) | [Gao et al., CVPR Workshops 2024](https://arxiv.org/abs/2405.02685) |
| 2 — Feature translation | [FeTrIL](fetril_2023.pdf) | [Petit et al., WACV 2023](https://arxiv.org/abs/2211.13131) |
| 3 — Old-class prototype augmentation | [PASS](pass_2021.pdf) | [Zhu et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html) |
| 4 — Federated class prototypes | [FedProto](fedproto_2022.pdf) | [Tan et al., AAAI 2022](https://arxiv.org/abs/2105.00243); downloaded v4 |
| 5 — Decentralized exchange | [PearFL](pearfl_2025.pdf) | [Qi et al., Mathematics 2025](https://www.mdpi.com/2227-7390/13/2/237) |
| 6 — Asynchronous continual federation | [FedSpace](fedspace_2023.pdf) | [Shenaj et al., CVPR Workshops 2023](https://arxiv.org/abs/2304.03626) |
| Classifier / feature collaboration | [FedPAC](fedpac_2023.pdf) | [Xu et al., ICLR 2023](https://openreview.net/forum?id=SXZr8aDKia) |
| Prototype separability | [FedTGP](fedtgp_2024.pdf) | [Zhang et al., AAAI 2024](https://arxiv.org/abs/2401.03230) |
| Latent exemplars / global forgetting | [Hybrid rehearsal](hybrid_rehearsal_2025.pdf) | [Khademi Nori et al., ICLR 2025](https://openreview.net/forum?id=ydREOIttdC) |
| Recent prototype-preserving framework | [STP-FCL](stp_fcl_2026.pdf) | [He et al., Sensors 2026](https://www.mdpi.com/1424-8220/26/19/6273); PDF date 3 October, webpage 2 October |
| Prototype privacy / noise caveat | [Noise-induced prototype degradation](noise_prototype_degradation_2026.pdf) | [Wang et al., arXiv:2604.27833v1](https://arxiv.org/abs/2604.27833); venue not assumed |
| Teacher distillation alternative | [Learning without Forgetting](learning_without_forgetting_2016.pdf) | [Li & Hoiem](https://arxiv.org/abs/1606.09282); ECCV 2016 concept, downloaded extended journal manuscript |

## Download provenance

* [Manifest](download_manifest.json): titles, primary source and approved PDF URLs.
* [Receipt](download_receipt.json): actual requested/resolved URL, size, SHA-256 and fallback attempts for each downloaded file.
* [BibTeX](references.bib): citation starters. Select the exact proceedings/journal version when preparing a publication; do not cite an unverified venue for a preprint.

The download helper defaults to a zero-network, zero-training plan:

```bash
environment/basil-noise-env/bin/python scripts/download_cart_literature.py
```

`--download` explicitly downloads the listed PDFs and refuses to replace
existing files or the receipt. All twelve were also parsed with `pdfinfo` and
`pdftotext` after download. The original 25 PDFs remain in the parent folder.

Review limitations: targeted search, not exhaustive novelty certification.
Published results from other architectures, pretraining, averaging or
continual-learning settings are not expected outputs for PaperMerge.
