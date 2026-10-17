# IID: 100-round overall accuracy

Arithmetic mean of full-test accuracy across all ten trained node models.

These are read-only derived views, not new training runs or resume checkpoints.
Raw artifacts remain at their original paths because queue validation and resume depend on them.
IID 30/100-round views are prefixes of the saved 500-round IID runs, not independent experiments.
Only manifest-committed rounds are shown. Missing rounds/conditions are never filled.

| Condition | Measured rounds | Status | Source |
|---|---:|---|---|
| A | 100/100 | complete | `newResults/IID/A_clean_no_attack_ce_500r_gpu` |
| B | 100/100 | complete | `newResults/IID/B_attack_clean_ce_500r_gpu` |
| C | 100/100 | complete | `newResults/IID/C_attack_noise_ce_500r_gpu` |
| D | 100/100 | complete | `newResults/IID/D_attack_noise_ebm_500r_gpu` |
