# Five-epoch diagnostic evidence

Protocol revision: `five_epoch_variance_v2`. All rows use batch size 512 and five complete local epochs.

Smoke runs use 512 examples/class and 100 test images/class; full-data preflights use all 50,000 training and 10,000 test images. Neither family is production research evidence.

| Condition | Rounds | Epochs | Samples/class | Attack start | Channel | SS | EBM | Final avg | Worst | Best | Trend | Class retention | Status/notes |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|---|---|
| preflight_clean_dirichlet | 5 | 5 | 5000 | 20 | off | off | none | 15.18% | 10.77% | 19.33% | up | own 17.6%; other 14.9% | completed |
| preflight_clean_iid | 5 | 5 | 5000 | 20 | off | off | none | 33.74% | 31.87% | 36.31% | up | own 33.6%; other 33.8% | completed |
| preflight_clean_one_class_per_node | 5 | 5 | 5000 | 20 | off | off | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_clean_dirichlet | 3 | 5 | 512 | 1 | off | off | none | 10.52% | 10.00% | 15.20% | up | own 10.0%; other 10.6% | completed |
| smoke_clean_iid | 3 | 5 | 512 | 1 | off | off | none | 11.64% | 10.00% | 13.10% | up | own 8.4%; other 12.0% | completed |
| smoke_clean_one_class_per_node | 3 | 5 | 512 | 1 | off | off | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_abs_0.005_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_abs_0.005_none | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | off | none | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_abs_0.005_ss | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_abs_0.005_ss_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | on | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_abs_0.01_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_abs_0.01_none | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | off | none | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_abs_0.01_ss | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_abs_0.01_ss_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | on | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_abs_0.02_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_abs_0.02_none | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | off | none | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_abs_0.02_ss | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_abs_0.02_ss_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | on | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_rel_0.2_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 80.0%; other 2.2% | completed |
| smoke_hidden_noise_rel_0.2_none | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | off | none | 10.00% | 10.00% | 10.00% | flat | own 80.0%; other 2.2% | completed |
| smoke_hidden_noise_rel_0.2_ss | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_rel_0.2_ss_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | on | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_rel_0.4_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 80.0%; other 2.2% | completed |
| smoke_hidden_noise_rel_0.4_none | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | none | 10.00% | 10.00% | 10.00% | flat | own 80.0%; other 2.2% | completed |
| smoke_hidden_noise_rel_0.4_ss | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_rel_0.4_ss_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | on | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_rel_0.6_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_rel_0.6_none | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | off | none | 10.00% | 10.00% | 10.00% | flat | own 90.0%; other 1.1% | completed |
| smoke_hidden_noise_rel_0.6_ss | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_noise_rel_0.6_ss_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | on | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_hidden_none | 3 | 5 | 512 | 1 | off | off | none | 10.00% | 10.00% | 10.00% | flat | own 80.0%; other 2.2% | completed |
| smoke_hidden_ss | 3 | 5 | 512 | 1 | off | on | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_abs_0.005_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | off | gradient_norm_objective | 10.28% | 10.00% | 11.40% | flat | own 83.1%; other 2.2% | completed |
| smoke_noise_abs_0.005_legacy | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | off | legacy_gradient_scale | 10.26% | 10.00% | 11.50% | flat | own 82.7%; other 2.2% | completed |
| smoke_noise_abs_0.005_none | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.005 | off | none | 10.26% | 10.00% | 11.30% | flat | own 82.9%; other 2.2% | completed |
| smoke_noise_abs_0.01_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_abs_0.01_legacy | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | off | legacy_gradient_scale | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_abs_0.01_none | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.01 | off | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_abs_0.02_ebm | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_abs_0.02_legacy | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | off | legacy_gradient_scale | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_abs_0.02_none | 3 | 5 | 512 | 1 | paper_absolute_gaussian 0.02 | off | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.2_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.2_legacy | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | off | legacy_gradient_scale | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.2_none | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.2 | off | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.4_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | gradient_norm_objective | — | — | — | unavailable | unavailable | non-finite loss/gradient; see execution.log |
| smoke_noise_rel_0.4_ebm (isolated reference recheck) | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | gradient_norm_objective | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.4_ebm (worker recheck) | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | gradient_norm_objective | — | — | — | unavailable | unavailable | non-finite loss/gradient; see execution.log |
| smoke_noise_rel_0.4_legacy | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | legacy_gradient_scale | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.4_none | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.4 | off | none | 10.00% | 10.00% | 10.00% | flat | own 100.0%; other 0.0% | completed |
| smoke_noise_rel_0.6_ebm | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | off | gradient_norm_objective | — | — | — | unavailable | unavailable | non-finite loss/gradient; see execution.log |
| smoke_noise_rel_0.6_legacy | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | off | legacy_gradient_scale | — | — | — | unavailable | unavailable | non-finite loss/gradient; see execution.log |
| smoke_noise_rel_0.6_none | 3 | 5 | 512 | 1 | relative_l2_gaussian 0.6 | off | none | — | — | — | unavailable | unavailable | non-finite loss/gradient; see execution.log |

## Interpretation

Compare the clean controls before interpreting threat defenses. An own-class-dominated node-by-class matrix measures catastrophic forgetting, not a software acceptance failure. Numerical divergence is recorded explicitly; no clipping, weight decay, epoch reduction or test-driven tuning was added to rescue a curve.

The initial matrix recorded four non-finite relative-noise failures. An isolated sigma_rel=0.4/full-EBM reference invocation completed near chance, but a subsequent normal-worker recheck failed again. All three observations are retained, not replaced. Invocation/build sensitivity is observed; its exact numerical cause is not isolated. Further reproducibility/stability review is required before production. New manifests record source hashes and runtime versions so future comparisons can distinguish implementation changes.

## Runtime planning

Linear CPU extrapolations from five-round clean full-data runs (same epoch-level evaluation frequency; not GPU promises):
- preflight_clean_one_class_per_node: approximately 0.84 hours for 100 rounds.
- preflight_clean_iid: approximately 1.00 hours for 100 rounds.
- preflight_clean_dirichlet: approximately 0.95 hours for 100 rounds.

Second-order EBM and five candidate evaluations add cost. Use these as baseline estimates, not guarantees for the entire matrix. Production remains explicitly gated.

## Prepared production configurations

All are generated, not launched. Legacy scaling is a separately labeled compatibility ablation.
- `gui/configs/sequential_basil/production/clean_iid.json`
- `gui/configs/sequential_basil/production/clean_dirichlet.json`
- `gui/configs/sequential_basil/production/clean_one_class_per_node.json`
- `gui/configs/sequential_basil/production/hidden_none.json`
- `gui/configs/sequential_basil/production/hidden_ss.json`
- `gui/configs/sequential_basil/production/noise_rel_0.2_none.json`
- `gui/configs/sequential_basil/production/noise_rel_0.2_ebm.json`
- `gui/configs/sequential_basil/production/noise_rel_0.2_legacy.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.2_none.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.2_ss.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.2_ebm.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.2_ss_ebm.json`
- `gui/configs/sequential_basil/production/noise_rel_0.4_none.json`
- `gui/configs/sequential_basil/production/noise_rel_0.4_ebm.json`
- `gui/configs/sequential_basil/production/noise_rel_0.4_legacy.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.4_none.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.4_ss.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.4_ebm.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.4_ss_ebm.json`
- `gui/configs/sequential_basil/production/noise_rel_0.6_none.json`
- `gui/configs/sequential_basil/production/noise_rel_0.6_ebm.json`
- `gui/configs/sequential_basil/production/noise_rel_0.6_legacy.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.6_none.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.6_ss.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.6_ebm.json`
- `gui/configs/sequential_basil/production/hidden_noise_rel_0.6_ss_ebm.json`
- `gui/configs/sequential_basil/production/noise_abs_0.005_none.json`
- `gui/configs/sequential_basil/production/noise_abs_0.005_ebm.json`
- `gui/configs/sequential_basil/production/noise_abs_0.005_legacy.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.005_none.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.005_ss.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.005_ebm.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.005_ss_ebm.json`
- `gui/configs/sequential_basil/production/noise_abs_0.01_none.json`
- `gui/configs/sequential_basil/production/noise_abs_0.01_ebm.json`
- `gui/configs/sequential_basil/production/noise_abs_0.01_legacy.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.01_none.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.01_ss.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.01_ebm.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.01_ss_ebm.json`
- `gui/configs/sequential_basil/production/noise_abs_0.02_none.json`
- `gui/configs/sequential_basil/production/noise_abs_0.02_ebm.json`
- `gui/configs/sequential_basil/production/noise_abs_0.02_legacy.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.02_none.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.02_ss.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.02_ebm.json`
- `gui/configs/sequential_basil/production/hidden_noise_abs_0.02_ss_ebm.json`
