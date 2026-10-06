# Repository audit

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

- `gui/experiment_app.py` is still a large controller with UI, queue,
  persistence, execution, and plotting responsibilities.
- Experiment JSON uses legacy camelCase keys. These are part of the stored
  schema and should be migrated with a versioned adapter, not renamed blindly.
- Legacy Campaign 3 paths remain supported by runtime code but start empty.
- TensorFlow-heavy integration checks require the project environment and GPU
  resources; they should run separately from fast unit tests.
