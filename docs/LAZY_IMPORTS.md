# Lazy dependency loading

Numerical, plotting and optional libraries are loaded when their feature is
actually used, rather than just because a configuration, GUI or report module
was imported. The repository-wide audit covers the Python sources in
`basil_core/`, `noise_comm/`, `gui/`, `reporting/`, `scripts/`, `tests/` and
`run_gui.py`; it does not rewrite installed dependencies, datasets, papers or
saved research artifacts.

## Implementation

| Mechanism | Purpose | First-use behavior |
|---|---|---|
| Function-local imports | Training, evaluation, array processing, image loading and plots | Python imports/caches the real library when the function runs |
| `lazy_module()` | Global annotations and compatibility bindings | Attribute access resolves the real module once; errors propagate normally |
| `lazy_pyplot()` | Plot backends and styles | Backend/style setup occurs at the first plotting operation |
| `lazy_tf_function()` | Existing compiled augmentation/forward/gradient functions | Constructs and caches `tf.function` on first use with unchanged tracing options |
| Deferred `trainer.lossFn` | Sparse CE from logits | Returns the same cached Keras loss type, created on first request |
| `main()` guards | Standalone calibration and convergence scripts | Importing a script cannot start its dataset download, GPU setup or training |

```python
def evaluate_images(images):
    import numpy as np  # cached by Python after the first call
    return np.asarray(images)

from basil_core.lazy_imports import lazy_module, lazy_pyplot, lazy_tf_function

np = lazy_module("numpy")
plt = lazy_pyplot(backend="Agg")

@lazy_tf_function(reduce_retracing=True)
def transform(images):
    import tensorflow as tf
    return tf.identity(images)
```

Lightweight standard-library imports, compiler-required `__future__` imports,
and imports necessary for class definitions remain explicit. Tkinter loads
when GUI components are requested. A legacy TensorFlow schedule subclass must
resolve TensorFlow to define its base class. Guarded dependency-availability
checks in tests remain real imports so missing-library skips keep working.
It would be incorrect to hide an unavailable dependency behind a proxy and
pretend that it was installed.

The final audit finds **zero unconditional, direct module-level imports of
heavy third-party libraries**. Fresh-process tests additionally verify that
the GUI entry module, configuration library, strict CART controller/model
definitions, reporting modules and guarded standalone scripts do not load
NumPy, TensorFlow, Matplotlib, PyTorch, tonic or W&B merely on import. Optional
N-MNIST dependencies are checked only when N-MNIST is requested; the existing
missing-dependency fallback is retained. PyTorch's legacy noise helper still
executes under `no_grad`, now entered inside the function rather than during
import-time decorator construction. This is stronger than
checking whether an import statement was moved.

## Scientific equivalence and provenance

The import refactor was applied **after** the three-round CART comparison
completed. Original executed files are preserved in its
`execution_sources/` archive and verified against the original run manifest.
Report regeneration accepts a matching archived source; it never substitutes
the new source hash into the original execution manifest.

The deterministic before/after synthetic check verifies exact equality of:

- CE/full-EBM plus anchor loss terms and gradients;
- SGD parameter updates and initial/trained parameter hashes;
- BASIL selected sender and receiver-local loss;
- five tiny test updates, CART controller and carried registry state;
- predicted logits.

It is a mathematical unit check, **not another CIFAR research experiment**.
The proof is saved with the diagnostic as `lazy_import_equivalence.json`.
No training rule, EBM coefficient, attacker behavior, channel formula, epoch
count, batch size or queue configuration was changed by the import refactor.

Synthetic GPU verification was repeated on the RTX 4070 Ti after refactoring:

| Objective | CPU/GPU relative gradient difference | Maximum update difference |
|---|---:|---:|
| CE + anchor | 2.268286465e-7 | 7.450580597e-9 |
| Full EBM + anchor | 1.724218066e-7 | 7.450580597e-9 |

The tolerances remain 1% relative gradient L2 and 1e-4 maximum update difference.
Both 512/392 batch checks were finite and legacy lambda had zero effect.
This is not a claim of bit-identical CPU/GPU execution or multiprocessing
equivalence. Future CART execution checks all current dependency fingerprints,
not just the controller file.

The existing IID GPU verifier was also refreshed with the new source
fingerprints, so future GUI execution does not bypass or fail its hardware
guard solely because imports moved. `scripts/verify_iid_gpu.py` passed its
synthetic CE/EBM, no-lambda, augmentation, SGD, evaluation and same-device
repeatability checks. CPU/GPU relative gradient differences were 0.002993859
(CE) and 0.002993895 (EBM), below its 0.01 tolerance; predicted labels matched.
It started zero research runs and recorded 372390912 bytes peak GPU allocation,
well below the 6144 MiB cap. No research queue or W&B run was launched.

## Verification commands

```bash
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py \
  --report gui/worker_state/lazy_import_inventory_after.json

environment/basil-noise-env/bin/python -m compileall -q \
  basil_core gui noise_comm reporting scripts tests run_gui.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
MPLCONFIGDIR=/tmp/papermerge-cart-test-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python scripts/verify_lazy_import_equivalence.py

environment/basil-noise-env/bin/python scripts/run_strict_cart_diagnostic.py \
  --device GPU --verify-device

environment/basil-noise-env/bin/python scripts/verify_iid_gpu.py

xvfb-run -a env CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
MPLCONFIGDIR=/tmp/papermerge-lazy-gui PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
environment/basil-noise-env/bin/python -m pytest -q \
  tests/test_gui_live_updates.py::test_real_widgets_stay_stable_and_network_and_chart_render \
  tests/test_iid_campaign.py::test_real_gui_loads_four_idle_configs_and_presets_without_launch
```

The virtual-display GUI checks passed **2/2** without launching a worker.
The lazy-import-specific suite passed **42/42**, including standalone plot
imports with no initial project path. Complete-suite totals are recorded below.
Final complete-suite result: **436 passed, 644 subtests passed, 2 skipped,
2 dependency deprecation warnings**, in 71.07 seconds. Compile passed. All
**23708** existing result, plot and paper files matched their before-work
SHA-256 inventory; zero changed. Only the new CART diagnostic namespaces are
excluded from that historical comparison.
The manual MNIST convergence
experiment is now guarded, so pytest collection cannot initiate its download
or training. Its original learning assertions were not run or claimed passed.
An initial collection attempt encountered its pre-existing download-at-import
behavior and failed due to unavailable network access; no convergence training
occurred. The guard addresses that import-side-effect defect, not the numerical
outcome of that legacy experiment.

## Limitations

Lazy imports improve startup and avoid loading unused libraries. They do not
make SGD intrinsically faster, guarantee freedom from GPU OOM, free TensorFlow
VRAM caches automatically, or solve class forgetting. Existing memory guards
and finite-value checks remain necessary. A feature still requires its actual
dependency when used; failures are reported rather than silently changing the
scientific method.
