# Repository audit

## GUI architecture update

The former 5,429-line `gui/experiment_app.py` implementation has been replaced
by a forwarding shim. The active desktop application is organized into state,
services, components, and workspace views. Research engines remain in
`basil_core/`, plotting implementations remain in `reporting/`, and worker
process contracts remain in `gui/worker_pool.py` and the worker scripts.

## Outcome

The repository now separates source code from generated experiment artifacts.
Campaign 4 remains intact; superseded result and plot trees were removed.

## Changes made

- Removed `paper_review/`, `experiments/results/`, `results2/`, `results3/`,
  and `plots/`, `plots2/`, `plots3/`.
- Preserved `experiments/results4/` and `plots4/`.
- Moved reusable plot generation code into the `reporting/` package.
- Renamed application and utility modules to concise `snake_case` names.
- Replaced version-based source names with responsibility-based modules:
  `adaptive_study`, `baseline_study`, `experiment_engine`, `worker_pool`, and
  `execution_policy`.
- Renamed `Papers/` to `papers/` and retained the numbered bibliography.
- Removed runtime caches from the distributable tree.
- Decoupled the checked-in config library from deleted Campaign 3 results by
  keeping the approved CART schedule in source control.

## Remaining technical debt

- `gui/experiment_app.py` is now a compatibility shim; new responsibilities are
  split across application state, services and workspace views.
- Experiment JSON camelCase keys remain a compatibility contract, handled
  through the typed-model adapter rather than renamed in historical data.
- Legacy Campaign 3 paths remain supported by runtime code but start empty.
- TensorFlow-heavy integration checks require the project environment and GPU
  resources; they should run separately from fast unit tests.

## Research protocol amendment

The explicit schema-5 research engine is separate from historical engines.
Five full epochs are enforced in production, smoke and full-data preflight.
Absolute-coordinate and relative-L2 channels are distinct; full gradient-norm
EBM and legacy scaling are separate conditions. New output namespaces reject
writes beneath protected roots. See [the protocol](SEQUENTIAL_BASIL_PROTOCOL.md) and
[actual diagnostic evidence](RESEARCH_DIAGNOSTIC_REPORT.md). The clean one-class
baseline exhibits forgetting; no defaults were tuned to manufacture 50% accuracy.
