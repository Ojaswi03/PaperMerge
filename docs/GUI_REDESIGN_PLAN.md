# GUI redesign implementation

## Current architecture

The redesign is implemented as a persistent Tkinter workspace. `gui/app.py`
owns application lifecycle and composes a small set of layers:

- `gui/state/` contains the canonical experiment model, camelCase schema
  adapter, centralized validation, application/execution states, and queue
  transitions. These modules do not import Tkinter.
- `gui/services/` owns atomic configuration and queue persistence, isolated
  worker/event adaptation, lazy retained-result discovery, metric loading, and
  guarded plot output.
- `gui/components/` contains the command bar, navigation rail, empty state,
  and scroll container.
- `gui/views/` contains Dashboard, Experiment Builder, Queue, Results,
  application shell, and integration of the existing Network view.

The old `gui/experiment_app.py` path remains only as a compatibility import.
There is one GUI implementation.

## Compatibility boundaries

Persisted configuration keys remain camelCase and schema versions 1–4 are
accepted. Unknown compatible fields are retained in the model's `extra`
mapping and restored on save. Queue persistence accepts both the historical
list-of-config-dictionaries form and the enriched status form. Versioned
baseline/adaptive runs continue to execute in isolated workers and graceful
stop still sends SIGTERM through the established worker pool.

The Results service reads only `run.json` during discovery. `metrics.npz` and
`telemetry.npz` are loaded on selection or replay. The retained
`experiments/results4/` and `plots4/` trees are read-only application inputs;
the plot service rejects output beneath the protected plot root.

## Product direction

Turn the GUI into an experiment workspace rather than one long configuration
form. The primary workflow should be: choose a study, configure it, validate
it, run it, and inspect results without opening separate dialogs.

## Information architecture

Use a persistent left navigation rail with five workspaces:

1. **Dashboard** — active run, queue health, latest accuracy, ETA, GPU status,
   and recent experiments.
2. **Experiment builder** — dataset, approach, topology, training, noise, and
   attack settings in progressive sections.
3. **Queue** — searchable table with bulk actions, validation state, runtime
   estimate, drag ordering, and worker assignment.
4. **Results** — run comparison, filters, live/final charts, metadata, and
   export actions.
5. **Network** — ring visualization and node telemetry for supported runs.

Keep global Run, Stop, Save, and queue-state controls in one top command bar.
Hide advanced parameters until their parent feature is enabled.

## Architecture

The default entry point now implements this state/service/view structure.
The research protocol adds schema-5 adapters, explicit channel/EBM controls,
read-only result/replay integration and a production-launch gate. Its algorithm
remains in `basil_core/research_protocol.py`, not in widget callbacks.

- Split `experiment_app.py` into `app`, `state`, `services`, and `views`.
- Use one typed experiment model as the source of truth. Tk variables should
  bind to that model rather than being read throughout execution code.
- Move process management, queue persistence, config I/O, and plotting behind
  narrow service interfaces.
- Route worker events through one event bus consumed by progress, log, chart,
  and network views.
- Keep output roots in one path module and make every generated artifact
  discoverable through a run manifest.

## Interaction improvements

- Validate fields inline and disable Run until blocking errors are resolved.
- Show a human-readable run summary before launch.
- Replace large modal pickers with searchable in-page drawers.
- Add empty, loading, running, failed, stopped, and completed states.
- Preserve queue edits automatically and clearly identify unsaved config edits.
- Add tooltips only for research terms; use plain labels for standard controls.

## Visual system

- Use an 8 px spacing scale, 12 px card radius, restrained borders, and one
  violet accent color with semantic success/warning/error colors.
- Use 14–16 px body text and stronger title/section hierarchy.
- Prefer cards and two-column responsive layouts over dense label grids.
- Keep charts visually quiet, with shared legends and accessible contrast.
- Support light and dark themes after component extraction is complete.

## Delivery phases

1. Extract state and services with characterization tests; no behavior change.
2. Build the navigation shell, dashboard, and shared components.
3. Rebuild the experiment builder with conditional sections and validation.
4. Integrate queue and results into first-class workspaces.
5. Add accessibility, keyboard navigation, theme support, and visual tests.

Each phase should keep the application runnable. Algorithm and experiment
schema changes are explicitly outside the visual redesign.
