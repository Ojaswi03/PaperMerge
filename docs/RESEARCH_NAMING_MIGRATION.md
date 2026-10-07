# Scientific naming migration

This is nomenclature-only: losses, optimizer updates, epochs, partitions, attacks,
noise, strict handoff and Snapshot Selection mathematics are unchanged. The
separate source-grounded loss study was not mixed into this rename. Production
remains gated.

## Inventory and renames

Before edits, the inventory contained 2,608 matching file paths, 343 matching
directories and 3,468 matching text lines. Of the paths, 2,473 are protected
historical files. Ignored research outputs were included; Git internals,
dependency virtualenvs and generated caches were excluded. Binary files were
enumerated by path rather than interpreted as source text.

- [Complete pre-edit inventory](RESEARCH_NAMING_INVENTORY.json): every matching
  path, directory and text line, recorded before renaming.
- [Full old-to-new mapping](RESEARCH_NAMING_RENAMES.json): 33 files and three
  directories directly renamed, covering 135 active file paths in total.
- [Remaining occurrences](RESEARCH_NAMING_REMAINING.json): every scanned remaining
  text line with its reason, plus all retained historical filenames.

| Area | Current names |
|---|---|
| Engine / observability | `basil_core/research_protocol.py`, `research_audit.py` |
| Configuration | `gui/research_protocol.py`, `gui/configs/sequential_basil/` |
| Telemetry | `gui/services/research_telemetry.py` |
| Plotting | `reporting/sequential_basil_plots.py` |
| Tests | `test_research_protocol.py`, `test_research_audit.py`, new `test_research_naming.py` |
| Execution | `run_research_protocol.py`, `run_research_diagnostics.py`, `run_research_validation.py`, `audit_research_protocol.py` |
| Reports | `report_research_diagnostics.py`, `report_research_validation.py` |
| Utilities | `calibrate_channel_noise.py`, `generate_research_protocol_configs.py`, `smoke_research_gui.py`, `replay_numerical_failure.py`, `analyze_channel_noise_growth.py`, `inference_demo.py`, `plot_research_protocol.py`, `verify_research_results.py` |
| Documentation | `RESEARCH_PROTOCOL_VERIFICATION.md`, `RESEARCH_DIAGNOSTIC_REPORT.md`, `SOURCE_EQUATIONS.md`, `NUMERICAL_STABILITY_AUDIT.md`, `REPRODUCIBILITY_AUDIT.md`, `ONE_CLASS_LEARNING_ANALYSIS.md`, `SNAPSHOT_SELECTION_AUDIT.md`, `RESEARCH_PRODUCTION_READINESS.md`, `SEQUENTIAL_BASIL_PROTOCOL.md` |
| Evidence | `validation_artifacts.json`, `validation_evidence.json`, `validation_noise_growth.json` |
| Calibration directories | `docs/noise_calibration/`, `docs/validation_noise_calibration/` |

The development files/directories being renamed were untracked, so filesystem
moves were used: `git mv` cannot move untracked files. Existing tracked edits
were preserved. Nothing was staged, committed or zipped. Twenty-three obsolete
bytecode files were removed; compileall regenerates them under current names.
Their paths are recorded in `RESEARCH_NAMING_CACHE_CLEANUP.json`.
The proposed loss-study, ablation and no-lambda reports did not yet exist; no
placeholder scientific reports were created.

## Identifiers, labels and compatibility

Canonical protocol: `sequential_basil_one_class_v1`. New experiment labels use
`Sequential BASIL`; new production purpose is `research_protocol_production`;
new run IDs start with `basil-`. The generator prepares 47 production, 47 smoke
and three preflight configurations without executing them.

Identifiers now include `run_research_protocol`, `PlotService.generate_research`
and `researchRecords`. Temporary local-variable and test-function labels were
replaced throughout active code. The GUI preset is **Sequential BASIL five-epoch
protocol**. Imports, subprocess targets, configuration paths, CLI names, source
hash paths, README commands and Markdown links use the new names.

`basil_core/protocol_compatibility.py` explicitly documents deprecated protocol,
purpose and display-name aliases. Old saved configs and queue entries load
without changing the files; explicit saves emit canonical known naming fields.
Scientific settings, schema versions, unknown metadata and historical run IDs
are preserved. Aliases do not invent missing scientific defaults. Old identifiers
also retain the production execution gate.

New results: `experiments/research_protocol_results/` and
`experiments/research_validation_results/`. New plots: `plots/research_protocol/`
and `plots/research_validation/`. Retained roots remain accessible through Results,
replay, report generation and orphan-worker checks. The shared output guard
rejects writes into every retained historical result/plot root.

## Remaining occurrences

There are **zero unintended occurrences** in active research code or filenames.
The remaining inventory records 2,981 text lines outside the naming archives
and 2,473 historical file paths. Every line has an explicit reason:

1. Protected historical filenames, metadata, logs and recorded hashes remain
   byte-identical to preserve experimental provenance.
2. The compatibility module contains explicitly deprecated aliases and read-only
   historical discovery paths. `.gitignore` retains the two old result roots so
   historical artifacts are not accidentally staged.
3. Five audit/verification documents retain literal paths to existing historical
   evidence or original manifests. Machine evidence JSON retains original source
   hashes and paths rather than falsifying provenance.
4. Naming inventory/mapping/hash archives intentionally record the old state.
   Those archives are excluded from recursive self-inventory to avoid duplicating
   the same records indefinitely.

No old module shim, duplicate configuration directory, or temporary-named
active source file remains.

## Mathematical equivalence

[Comparison evidence](RESEARCH_NAMING_EQUIVALENCE.json) covers ten before/after
arrays: initialization, CE and gradient-norm loss values, full second-order EBM
gradients, SGD updates, logits, keyed absolute-noise samples, rolling memory,
selected sender, five-epoch updates and epoch losses. **All are byte-identical;
maximum absolute difference is 0.**

The deterministic CPU fixture uses seed 2025, oneDNN off, one intra/inter-op
thread, four CIFAR-shaped images with labels 0–3, LR 0.05, absolute sigma 0.01,
EBM coefficient 0.0001 and five complete local epochs. The model is the existing
BASIL paper CNN. This is rename-equivalence evidence, not an accuracy or GPU claim.

## Executed checks

Commands used `environment/basil-noise-env/bin/python`, abbreviated below as
`python`. CPU checks used `CUDA_VISIBLE_DEVICES=-1`; Matplotlib used a temporary
cache directory. Commands below were executed, not proposed:

```bash
python -m compileall -q gui basil_core reporting scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 python -m pytest -q --ignore=tests/test_convergence.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 python -m pytest -q tests/test_research_naming.py tests/test_research_protocol.py tests/test_research_audit.py tests/test_gui_architecture.py tests/test_config_library.py
TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 xvfb-run -a python scripts/smoke_research_gui.py
python scripts/generate_research_protocol_configs.py
python scripts/report_research_diagnostics.py --output /tmp/papermerge_naming_diagnostics.md
python scripts/calibrate_channel_noise.py --semantics paper_absolute_gaussian --sigma 0.005 0.01 0.02 --trials 3 --output /tmp/papermerge_naming_calibration.json
python scripts/calibrate_channel_noise.py --semantics relative_l2_gaussian --sigma 0.2 0.4 0.6 --trials 3 --output /tmp/papermerge_naming_relative_calibration.json
python scripts/run_research_protocol.py --config /tmp/papermerge_naming_config.json --result-root /tmp/papermerge_naming_worker
python scripts/verify_research_results.py /tmp/papermerge_naming_worker/smoke/basil-f9db27815f0082e4b6bc
python scripts/plot_research_protocol.py /tmp/papermerge_naming_worker/smoke/basil-f9db27815f0082e4b6bc --plot-root /tmp/papermerge_naming_plots
sha256sum -c /tmp/papermerge_loss_study_before.sha256
cmp /tmp/papermerge_loss_study_before.sha256 /tmp/papermerge_naming_after.sha256
git diff --check
```

| Check | Actual outcome |
|---|---|
| Pre-edit baseline | 143 passed, 644 subtests, two protobuf warnings |
| Final fast suite | **154 passed, 644 subtests**, two warnings, 48.41 s |
| Final focused naming/protocol/audit/GUI/config suite | **78 passed, 644 subtests**, two warnings, 6.40 s |
| Final migration-only regression suite | **11 passed**, two warnings, 2.30 s |
| Compile / whitespace | exit 0 |
| Config generation / migration | 97 configs prepared; alias load/save and queue tests pass |
| GUI startup / retained telemetry replay | passed under Xvfb with local display-socket access |
| Renamed CLI help | 12 entry points; all exit 0 |
| Diagnostic report | 52 retained runs, 47 completed; read-only input |
| Validation report | 63 runs, 15 comparisons, 9 selection audits, 3 preflights |
| Absolute / relative calibration | both exit 0; three trials at each sigma |
| Compatibility worker / result verifier | completed; ten activations verified |
| Saved failure replay | exit 0; preserved failure input, separate temporary output |
| Plot loading / inference demo | exit 0 on new and retained outputs; ten node predictions printed |
| Before/after numerical comparison | ten arrays byte-identical |

The compatibility worker uses an old identifier, one ring round, four diagnostic
training samples/class, one test sample/class and five complete diagnostic epochs.
It is explicitly `researchValid=false`; no new scientific conclusion or production
run is implied. Report, replay, plots and inference also read retained artifacts
without modifying them. Calibration performs no training.

Two unfiltered full-suite attempts were made. The first failed collection because
MNIST download/DNS was unavailable. The second populated a temporary `KERAS_HOME`
from the existing cached MNIST file and completed the unchanged convergence script:
**3 passed / 3 failed checks**, followed by `sys.exit(1)` during pytest collection
(pytest exit 3; 39.76 s). Both BASIL and FedAvg label their first post-training
history entry as pre-training in this legacy script; BASIL's upward-trend check
also failed in this invocation. No scientific behavior or assertions were altered
to rescue it. The unfiltered suite therefore **does not pass**. Only the completed
focused suite is claimed to pass. External pytest plugins were disabled because
of the existing ROS/lark issue. GPU numerical behavior remains unverified.

Existing audit commands were translated to current entry-point names, not claimed
as new experimental measurements; original paths and source hashes remain in the
machine evidence.

## Protected artifacts

[SHA-256 manifest evidence](RESEARCH_NAMING_ARTIFACTS.json) records every path/hash
and matching before/after manifest digests.

| Tree | Before | After | Hashes |
|---|---:|---:|---|
| `experiments/results4/` | 2,153 | 2,153 | all match |
| `plots4/` | 737 | 737 | all match |
| Retained protocol results | 842 | 842 | all match |
| Retained protocol plots | 726 | 726 | all match |
| Retained validation results | 845 | 845 | all match |
| Retained validation plots | 60 | 60 | all match |
| **Total** | **5,363** | **5,363** | **byte-identical** |

Scientific status stays **NOT READY FOR PRODUCTION**: this rename does not resolve
numerical instability, multiprocessing discrepancies or GPU verification.
