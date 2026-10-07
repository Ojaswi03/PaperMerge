# Research production readiness

## NOT READY FOR PRODUCTION

Naming-only update: active modules, scripts and configurations now use scientific
terminology; deprecated identifiers remain readable. The naming verification
passed 154 fast tests and 644 subtests, with bit-identical numerical fixtures and
5,363 unchanged historical files. The unfiltered suite still fails in the legacy
collection-time convergence script. See [the migration evidence](RESEARCH_NAMING_MIGRATION.md).
This update changes no loss mathematics or scientific readiness assessment.

No 100-round or 47-condition production matrix was launched. GPU numerical
behavior is **unverified**; every new training diagnostic explicitly used CPU.

The implementation contracts pass, but relative-0.6 numerical failures and
repeatable direct/spawn gradient discrepancies remain. One-class collapse is
an honest research finding, not a reason to secretly alter the protocol.

## Implemented during this audit

- Opt-in stage/batch observability, hashes, failure checkpoints, operation-level
  dumps and mode-correct replay utilities.
- Float64 model/noise norm and coordinate-variance accumulation, preserving
  float32 training and stopped noise coefficients.
- Rejection of non-finite evaluation logits before accuracy reporting.
- Explicit research CPU runtime policy and worker-pool diagnostic argument
  routing; no historical worker lifecycle or queue format replacement.
- Before-training and all-five-epoch confusion matrices, per-class figures,
  evaluation-only candidate-quality observations and channel-only growth study.
- New regression tests and the four detailed audits linked below.

Five full epochs, ten one-class nodes, batch 512, strict handoff, S=5, reset
SGD with initial LR 0.05/round decay, four seeded attackers, production attack
start 20, channel start 0, and no clipping/weight decay/momentum are unchanged.
Unknown historical fields, old engines and persisted research results remain intact.

## Executed checks

The evidence below is from the earlier numerical-stability audit, not newly
rerun production research. Current naming checks are recorded in the migration
report linked above; historical command references now use current script names.

The commands below were executed using `environment/basil-noise-env/bin/python`
(shown as `python` after activating that environment).

```bash
python -m compileall -q gui basil_core reporting scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 python -m pytest -q --ignore=tests/test_convergence.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 python -m pytest -q tests/test_research_audit.py tests/test_research_protocol.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 python -m pytest -q --ignore=tests/test_convergence.py
python scripts/run_research_validation.py --suite smoke --output experiments/research_validation_results/final_cpu
python scripts/run_research_validation.py --suite preflight
python scripts/run_research_validation.py --suite repro --output experiments/research_validation_results/final_cpu
python scripts/run_research_validation.py --suite repro --output experiments/research_validation_results/keyed_runtime
MPLCONFIGDIR=/tmp/papermerge_mpl python scripts/report_research_validation.py --plots
git diff --check
```

| Check | Actual outcome |
|---|---|
| Pre-edit fast baseline | 133 passed, 644 subtests, 2 protobuf warnings |
| Focused audit/protocol intermediate suite | 44 passed, 2 warnings |
| Final fast suite | **143 passed**, **644 subtests passed**, 2 warnings, 44.41 s |
| Compile and whitespace checks | exit 0 |
| Final targeted smoke | **9 completed / 12**, **3 numerical failures**, exit 1 |
| Full-data clean preflights | **3 completed / 3**, exit 0 |
| Each final/keyed execution-mode comparison | **4 completed / 4**, exit 0; spawn hashes **do not match** |
| Epoch-level plots | **60 PNGs**, 900 full-test confusion matrices represented |
| Historical hashes/counts | **4,458 before / 4,458 after**, all SHA-256 hashes match |

There are no failed/skipped tests in the selected final suite. The deliberately
excluded `tests/test_convergence.py` performs research training during collection
and is **not claimed to pass**. Automatic third-party pytest plugin loading is
disabled because the system ROS plugin requires an unavailable `lark` module.
No GUI/GPU smoke was repeated for this scientific-only task.

Calibration commands completed for both semantics using CPU, oneDNN off and
one-thread TensorFlow. Exact reports are in `validation_noise_calibration/`.
`scripts/analyze_channel_noise_growth.py` completed a channel-only 20-stream,
30-transmission analysis for each relative sigma. Saved-batch replays generated
finite ranges and non-finite tensor values without training a new experiment.

## Scientific outcomes and remaining gates

1. Relative-0.4 full EBM completes in the new reference invocations, but old
   failure evidence is preserved and runtime sensitivity is not dismissed.
2. Relative-0.6 none/legacy/full EBM still overflow. The channel amplifies norm;
   large finite SGD steps and second-order corrections amplify it further.
3. IID/Dirichlet five-round preflights reach 33.697%/14.873% mean accuracy;
   one-class stays at 10% with constant-class predictions and temporal forgetting.
4. SS memory/selection contracts pass, but these near-chance smokes do not
   establish ten-class defense efficacy. Global metrics remain evaluation-only.
5. One fully matched serial four-mode run was followed by mismatching spawn
   repeats, even with a fixed hash/BLAS environment. Exact TensorFlow-level
   ordering remains unresolved. Completion/final-score equality is not enough.

Before production, resolve or explicitly accept the reproducibility limitation
and define an authorized handling policy for numerically failing conditions.
Do not lower sigma, clip gradients, change epochs, add weight decay, blend CART
into the baseline or tune against the test set without a separate explicit
research amendment. GPU validation and production launch each require their
own authorization; this document is not that authorization.

## Protected artifacts

| Tree | Before | After | Verification |
|---|---:|---:|---|
| `experiments/results4/` | 2,153 | 2,153 | every hash matches |
| `plots4/` | 737 | 737 | every hash matches |
| Existing research results/logs | 842 | 842 | every hash matches |
| Existing research plots | 726 | 726 | every hash matches |

The original manifest is `/tmp/papermerge_validation_before.sha256`; the
independently enumerated after-manifest is `/tmp/papermerge_validation_after.sha256`.
`sha256sum -c` and `cmp` both exited 0. Filenames containing spaces were
enumerated with NUL delimiters. New outputs use separate validation namespaces.
The complete path/hash list is also retained durably in
`docs/validation_artifacts.json`, verified against the pre-edit manifest.

## Handoff map

- [Numerical stability](NUMERICAL_STABILITY_AUDIT.md)
- [Reproducibility](REPRODUCIBILITY_AUDIT.md)
- [One-class learning](ONE_CLASS_LEARNING_ANALYSIS.md)
- [Snapshot Selection](SNAPSHOT_SELECTION_AUDIT.md)
- `validation_evidence.json`: results, first divergences and selection summaries
- `basil_core/research_audit.py`: optional instrumentation
- `scripts/audit_research_protocol.py`, `run_research_validation.py`: bounded execution
- `scripts/replay_numerical_failure.py`: saved-batch arithmetic inspection
- `scripts/report_research_validation.py`: read-only summaries/plots
- `tests/test_research_audit.py`: precision, Hessian, evaluation and replay regressions
