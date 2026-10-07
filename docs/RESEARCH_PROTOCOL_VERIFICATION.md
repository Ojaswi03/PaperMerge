# Implementation verification

Checks executed for the five-epoch amendment on the repository environment:

```bash
python -m compileall -q gui basil_core reporting scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 python -m pytest -q --ignore=tests/test_convergence.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 python -m pytest -q tests/test_research_protocol.py tests/test_gui_architecture.py tests/test_config_library.py
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 xvfb-run -a python scripts/smoke_research_gui.py
python scripts/generate_research_protocol_configs.py
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 python scripts/run_research_diagnostics.py --family smoke --jobs 2
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 python scripts/run_research_diagnostics.py --family preflight --jobs 1
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/papermerge_mpl python scripts/report_research_diagnostics.py --plots
```

- Compile check: exit 0.
- Fast suite: **133 passed**, **644 subtests passed**, two upstream protobuf
  deprecation warnings. No failed or skipped tests in the selected suite.
- Focused protocol/state/config suite: **57 passed**, **644 subtests passed**,
  the same two warnings.
- GUI smoke: passed under Xvfb outside the socket-restricted sandbox, including
  minimum-size command availability, five workspaces, noise/EBM labels,
  progressive disclosure and read-only research replay.
- Initial amended smoke matrix: **43 completed and contract-verified, 4 failed**
  on non-finite training values. This command appropriately exits nonzero.
- Full-data preflight: **3 completed and contract-verified, 0 failed**; each
  uses five rounds and five full epochs. The one-class run performs 50 optimizer
  steps per activation and evaluates the full balanced test set.
- Reference/current-worker rechecks preserve both the successful isolated
  invocation and subsequent failed worker invocation at relative sigma 0.4.
  These are separate evidence, not replacements for the original failed run.
- **636 PNG plots** exist for the amended smoke/preflight/recheck revision, outside
  the protected trees. Failed runs have no invented final-score plots.
- CIFAR-index and unlabeled-external-image inference demos both exited 0.
- `git diff --check`: exit 0.

Calibration commands also completed:

```bash
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 python scripts/calibrate_channel_noise.py --semantics relative_l2_gaussian --evaluate --output docs/noise_calibration/relative.json
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 python scripts/calibrate_channel_noise.py --semantics paper_absolute_gaussian --sigma 0.005 0.01 0.02 0.2 0.4 0.6 --evaluate --output docs/noise_calibration/absolute.json
```

These are initialization calibration, not trained-robustness claims. A separate
`--checkpoint` calibration also completed against a clean smoke checkpoint.

## Protected artifacts

The pre-edit manifest `/tmp/papermerge_professor_before.sha256` enumerated every
file. The post-edit and handoff manifests were enumerated independently, then
compared, not merely checked for missing original files.

| Tree | Before | After | SHA-256 |
|---|---:|---:|---|
| `experiments/results4/` | 2,153 | 2,153 | Every file matches |
| `plots4/` | 737 | 737 | Every file matches |
| Total | 2,890 | 2,890 | Manifests identical |

```bash
sha256sum -c /tmp/papermerge_research_before.sha256
rg --files -uuu experiments/results4 plots4 | LC_ALL=C sort | xargs -d '\n' sha256sum > /tmp/papermerge_research_handoff.sha256
cmp /tmp/papermerge_research_before.sha256 /tmp/papermerge_research_handoff.sha256
```

All checks exited 0. Protected research files were not rewritten or regenerated.

## Environment boundaries and remaining scientific gate

The default pytest command hits an unrelated system ROS plugin missing `lark`;
disabling automatic external plugin discovery resolves that environment issue.
`test_convergence.py` was deliberately excluded: it performs training during
collection and may fetch MNIST. This is not an assertion that it passed.
All research diagnostics used cached CIFAR and CPU; GPU numerical behavior was
not verified. No 100-round production condition was launched.

The clean one-class baseline forgets earlier classes. Some relative-noise runs
are numerically unstable, and invocation/build sensitivity is not fully isolated.
Production approval must consider these findings. No algorithm or training
invariant was weakened to improve accuracy. Existing research engines and
historical schema identifiers remain distinct from the new research protocol.
