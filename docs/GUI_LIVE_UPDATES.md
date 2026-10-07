# Live IID accuracy and network corrections

The IID stdout adapter previously forwarded progress per activation but only
round-level accuracy, and forwarded no network lifecycle or node telemetry.
Meanwhile every state notification destroyed/recreated the Dashboard and
deleted/reinserted every Queue row. Multiple notifications per event batch
amplified the flicker and lost queue selection.

Changes:

- Dashboard cards and chart persist; only changed text/data redraws.
- Queue updates rows by ID, preserving selection, focus and ordering.
- One observer notification is sent per drained execution-event batch.
- Completed node accuracy is a separate live series; round mean/worst remain
  actual full-traversal metrics, not invented partial averages. The first point
  is shown without waiting for a second round.
- The serial IID process emits compact selected-snapshot, candidate, attack,
  gradient, accuracy and outgoing-link telemetry using the replay adapter.
- Network receives start/node/round/finish events; node labels update on each
  activation, and mean/worst update at round completion.
- Elapsed/ETA labels tick from the existing main-thread poll without rebuilding
  the shell. Inspector contents are reused when the selected payload is unchanged.
- Starting another run clears only its live accuracy series, preventing curves
  from different conditions from being joined.
- User-facing Campaign 4 wording was removed. `NetworkView` is the current
  class. `CampaignNetworkView` is a deprecated import alias; the historical
  `campaignVersion` key remains solely for backward-compatible replay.
- Older worker messages containing accuracy alone are still supported; their
  unavailable candidate/link details are explicitly labelled, not fabricated.

No model, loss, noise draw, partition, attacker setting, selection criterion,
epoch count, learning rate, output schema or scientific metric was changed.
The new stdout telemetry reads existing activation records; it does not feed
anything back into training. Tests use simulated events, gradient contracts and
temporary fixtures, never a research execution or queue start.

Verification commands:

```bash
environment/basil-noise-env/bin/python -m compileall -q gui scripts tests
DISPLAY= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_gui_live_updates.py tests/test_gui_architecture.py tests/test_iid_campaign.py tests/test_config_library.py tests/test_execution_policy.py
DISPLAY=:99 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_gui_live_updates.py::test_real_widgets_stay_stable_and_network_and_chart_render tests/test_iid_campaign.py::test_real_gui_loads_four_idle_configs_and_presets_without_launch
DISPLAY= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_iid_study.py tests/test_iid_plot_pipeline.py -k 'not test_no_lambda_loss_gradients_updates_and_strict_load'
```

The first group passed 50 tests and 644 subtests, with two display tests skipped
headlessly. Both display tests passed separately under Xvfb with research launch
blocked. They verify stable widget identities, unchanged chart items on repeated
refresh, preserved queue selection, a visible first accuracy point, rendered
network candidates/links/accuracy, and no visible campaign wording.
The training test in the IID group is deliberately excluded; the unrelated
legacy convergence collection problem is not claimed fixed.
The IID regression/plot group passed 22 tests with one training test deselected.
Across these commands: 74 tests passed plus 644 subtests; no research was run.

SHA-256 before/after checks matched all 5,485 historical/reference files and
all 11 files in the new stopped-run output trees. The stopped A run (one complete
round) and the user's queue were not restarted, overwritten or reset.

After updating, reopen the GUI to load the fixes. This is not permission to
restart research. The subsequent [queue recovery implementation](IID_QUEUE_RECOVERY.md)
adds complete state checkpoints and verified reconstruction for older interrupted
runs; saved model weights alone remain insufficient.
