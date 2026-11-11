# EuroSAT RGB evaluation

A fixed sample of 2,000 satellite images, 200 from each land-cover class. The sample uses NumPy's default random generator with seed 2025 over sorted archive paths.
RGB images were resized from 64×64 to 32×32 with box resampling and scaled to [0, 1]. The four frozen CIFAR-10 models use round 500, training seed 2025, non-IID partitioning, and Hidden attack + noise. Each method contributes 100 client models.

| Method | CIFAR-10 mean confidence | EuroSAT mean confidence | EuroSAT predictions at least 90% confident | Mean confidence AUROC | Most frequent EuroSAT output |
|---|---:|---:|---:|---:|---|
| BASIL | 30.91% | 24.07% | 0.0005% | 0.786 | horse (22.98%) |
| BASIL + ACDS | 34.04% | 29.27% | 1.0390% | 0.568 | deer (30.03%) |
| FedSharp | 37.92% | 33.03% | 0.0030% | 0.628 | frog (23.75%) |
| FedSharp+ | 35.03% | 27.43% | 0.0010% | 0.685 | ship (32.35%) |

Confidence is the largest predicted CIFAR-10 class probability. AUROC measures how well that score ranks original CIFAR-10 images above EuroSAT images, averaged over clients. The 90% cutoff is descriptive.
Land-cover accuracy requires output labels that match EuroSAT. The current output labels represent CIFAR-10 objects.

[EuroSAT RGB source](https://zenodo.org/records/7711810)
