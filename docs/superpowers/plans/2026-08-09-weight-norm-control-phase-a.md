# Weight-Norm Control Phase A (Fixed Weight Decay) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an orthogonal, default-off `weightDecayCoefficient` config field, wire it into Campaign 4's SGD optimizer, and queue a 3-config diagnostic set to test whether it breaks the sigma=0.6 weight-norm-runaway collapse without touching EBM, SS, or the channel-noise definition.

**Architecture:** A single new numeric config field flows from `gui/adaptive_study.py`'s `make_config` through `run_campaign_four` into `SharedDeviceWorker`'s `tf.keras.optimizers.SGD` constructor, using TF's native decoupled weight decay. Default `0.0` maps to `weight_decay=None` on the optimizer, so every existing/regenerated config with the field unset behaves identically to before this change. Naming (`_condition_id`/`_experiment_name`) marks any config with `weightDecayCoefficient > 0` as a distinctly labeled condition, never colliding with `no_mitigation`.

**Tech Stack:** Python 3.12, TensorFlow 2.18.0 (`tf.keras.optimizers.SGD(weight_decay=...)`), existing Campaign 4 config-generation/engine/test modules.

## Global Constraints

- `weightDecayCoefficient` defaults to `0.0` on every config-generation path (`build_main_confirmation`, `build_static_controls`, `build_diagnostic`, `build_performance_benchmark`) — spec section "Phase A design: Config".
- This is additive: `mitigation`/`ebmMode` semantics, SS, and static/adaptive EBM stay completely untouched (spec Non-goals).
- `channelNoiseSemantics`/`channelNoiseSigma` (the noise definition) are never modified (spec Non-goals, Decision 1).
- Phase A validation configs use `approach=merged`, `nonIID`, `optimizerStateMode=persistent`, `mitigation=none`, `phase=diagnostic`, seed 2025, 100 rounds, `weightDecayCoefficient=1e-4` (spec "Validation set").
- Editing `gui/adaptive_study.py` and `basil_core/adaptive_experiment_engine.py` touches two of the project's `PROVENANCE_FILES`; regenerating the config library and re-running `sync_adaptive_configs.py --check` afterward is required (established precedent from the 2026-08-08 adaptive-EBM retune).
- Do not launch GPU training directly — build and queue configs into `gui/queue_state.json` for the user to run via the GUI (established practice this session).

**Correction to the spec's phrasing:** the spec's contract-test description says this change proves "`config_hash`/`runId` for every existing generated config is unchanged before/after." That is not literally achievable — `config_hash` hashes the full canonical config payload, so adding any new key changes every config's hash and `runId` (same as the 2026-08-08 adaptive-EBM constant edit did). The actual, correct regression guarantee is: **the new field defaults to `0.0` everywhere, and the resulting optimizer behavior is unchanged** (verified by Task 2's unit test showing `weight_decay=None`/`0.0` is a no-op) — not literal runId string stability. Task 1's contract test below verifies the correct claim.

---

### Task 1: Add `weightDecayCoefficient` config field, naming, and contract test

**Files:**
- Modify: `gui/adaptive_study.py:178-196` (`_condition_id`), `gui/adaptive_study.py:199-226` (`_experiment_name`), `gui/adaptive_study.py:229-409` (`make_config`)
- Test: `tests/test_adaptive_study_contracts.py`

**Interfaces:**
- Produces: `make_config(..., weight_decay_coefficient: float = 0.0)` — new keyword parameter, stored in the returned config dict as `config["weightDecayCoefficient"]` (float).
- Produces: `_condition_id(environment, mitigation, sigma, ebm_mode, weight_decay=0.0)` — new trailing keyword parameter; appends `_wd` to the returned string when `weight_decay > 0`.

- [ ] **Step 1: Write the failing contract test**

Add to `tests/test_adaptive_study_contracts.py` (near `test_environment_and_mitigation_semantics_are_explicit`, using the same imports already present in that file — `make_config`, `build_main_confirmation`, `build_static_controls`, `build_diagnostic`, `build_performance_benchmark`):

```python
    def test_weight_decay_defaults_to_zero_and_is_named_when_enabled(self):
        default_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
        )
        self.assertEqual(default_config["weightDecayCoefficient"], 0.0)
        self.assertNotIn("_wd", default_config["conditionId"])
        self.assertNotIn("weight_decay", default_config["experimentName"])

        decayed_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
            weight_decay_coefficient=1e-4,
        )
        self.assertEqual(decayed_config["weightDecayCoefficient"], 1e-4)
        self.assertTrue(decayed_config["conditionId"].endswith("_wd"))
        self.assertIn("weight_decay=0.0001", decayed_config["experimentName"])
        self.assertNotEqual(decayed_config["runId"], default_config["runId"])

        combined = (
            build_main_confirmation()
            + build_static_controls()
            + build_diagnostic("merged")
            + build_diagnostic("cart")
            + build_performance_benchmark()
        )
        self.assertTrue(
            all(config["weightDecayCoefficient"] == 0.0 for config in combined)
        )
        self.assertEqual(len(build_main_confirmation()), 396)
        self.assertEqual(len(build_static_controls()), 60)
        self.assertEqual(len(build_diagnostic("merged")), 33)
        self.assertEqual(len(build_diagnostic("cart")), 33)
        self.assertEqual(len(build_performance_benchmark()), 4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_study_contracts.Campaign4ContractTests.test_weight_decay_defaults_to_zero_and_is_named_when_enabled -v`

Expected: FAIL with `KeyError: 'weightDecayCoefficient'` (the field doesn't exist yet).

- [ ] **Step 3: Add the parameter to `_condition_id`**

In `gui/adaptive_study.py`, replace the `_condition_id` function (lines 178-196):

```python
def _condition_id(
    environment: str,
    mitigation: str,
    sigma: float,
    ebm_mode: str,
    weight_decay: float = 0.0,
) -> str:
    method = ""
    if mitigation in ("ebm", "ss_ebm"):
        method = f"_{ebm_mode}_ebm"
    suffix = "_wd" if float(weight_decay) > 0.0 else ""
    if environment == "clean":
        return "clean_pairwise_no_mitigation" + suffix
    if environment == "clean_ceiling":
        return "clean_full_consensus_ceiling" + suffix
    if environment == "hidden":
        return f"hidden_{mitigation}" + suffix
    prefix = f"sigma_{_sigma_label(sigma)}"
    if environment == "noise":
        return f"noise_{prefix}_{mitigation}{method}" + suffix
    return f"hidden_noise_{prefix}_{mitigation}{method}" + suffix
```

- [ ] **Step 4: Add the naming suffix to `_experiment_name`**

In `gui/adaptive_study.py`, in `_experiment_name` (lines 199-226), add a `weight_decay` local variable and splice it into the returned f-string. Replace the whole function:

```python
def _experiment_name(config: dict) -> str:
    sigma = float(config["channelNoiseSigma"])
    environment = config["environment"]
    environment_label = {
        "clean": "Protocol-Matched Clean Ring",
        "clean_ceiling": "Ideal Full-Consensus Clean Ceiling",
        "hidden": "Hidden Attack",
        "noise": f"Channel Noise sigma={sigma:.1f}",
        "hidden_noise": f"Hidden Attack + Channel Noise sigma={sigma:.1f}",
    }[environment]
    method = ""
    if config["ebmMode"] != "none":
        method = f" ({config['ebmMode'].title()} EBM)"
    cart = (
        f" | gamma={float(config['distillStrength']):g}"
        if config["approach"] == "cart"
        else ""
    )
    optimizer = (
        " | optimizer=persistent"
        if config.get("optimizerStateMode") == "persistent"
        else ""
    )
    weight_decay = (
        f" | weight_decay={float(config.get('weightDecayCoefficient', 0.0)):g}"
        if float(config.get("weightDecayCoefficient", 0.0)) > 0.0
        else ""
    )
    return (
        f"V4 | {config['split']} | {config['approach'].upper()} | "
        f"{environment_label} | {_MITIGATION_LABELS[config['mitigation']]}"
        f"{method}{cart}{optimizer}{weight_decay} | seed={config['seed']}"
    )
```

- [ ] **Step 5: Add the parameter to `make_config` and wire it through**

In `gui/adaptive_study.py`, in the `make_config` signature (starts line 229), add the new keyword parameter after `precision: str = "float32",`:

```python
    precision: str = "float32",
    weight_decay_coefficient: float = 0.0,
) -> dict:
```

In the config dict literal, add the field right after the `"momentum": 0.9,` line (near line 388):

```python
        "learningRate": 0.05,
        "momentum": 0.9,
        "weightDecayCoefficient": float(weight_decay_coefficient),
        "optimizerStateMode": optimizer_state_mode,
```

Replace the `config["conditionId"] = ...` call (lines 404-406) to pass the new value:

```python
    config["conditionId"] = _condition_id(
        environment, mitigation, sigma, ebm_mode, weight_decay_coefficient
    )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_study_contracts.Campaign4ContractTests.test_weight_decay_defaults_to_zero_and_is_named_when_enabled -v`

Expected: PASS

- [ ] **Step 7: Run the full contract test suite to check for regressions**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_study_contracts -v`

Expected: all 13 tests PASS (12 pre-existing + the new one). If `test_generated_manifest_matches_library` now fails, that's expected — Task 3 regenerates the library; do not fix it in this task.

- [ ] **Step 8: Commit**

```bash
git add gui/adaptive_study.py tests/test_adaptive_study_contracts.py
git commit -m "feat: add weightDecayCoefficient config field for Phase A weight-norm control

Orthogonal, default-off field (0.0 everywhere unless explicitly set) per
docs/superpowers/specs/2026-08-09-weight-norm-control-design.md. Naming
(_condition_id/_experiment_name) marks it as a distinct, separately-labeled
condition per Decision Rule 8 -- never collides with no_mitigation."
```

---

### Task 2: Wire `weight_decay` into the SGD optimizer with a unit test

**Files:**
- Modify: `basil_core/adaptive_experiment_engine.py:369-393` (`SharedDeviceWorker.__init__`), `basil_core/adaptive_experiment_engine.py:934-940` (`run_campaign_four`, `SharedDeviceWorker(...)` call site)
- Test: `tests/test_adaptive_experiment_engine.py`

**Interfaces:**
- Consumes: `config["weightDecayCoefficient"]` (float, produced by Task 1's `make_config`).
- Produces: `SharedDeviceWorker(..., weight_decay: float = 0.0)` — new keyword parameter.

- [ ] **Step 1: Write the failing unit test**

Add to `tests/test_adaptive_experiment_engine.py`, inside `Campaign4EngineTests` (after `test_run_emits_node_events_and_complete_telemetry`, reusing the existing `self._config`, `self._data`, `self.TinyModel` helpers already in that file):

```python
    def test_weight_decay_reduces_final_model_norm(self):
        train, test, metadata = self._data(seed=99)
        config_off = self._config(
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=99,
            nRounds=6,
            weightDecayCoefficient=0.0,
        )
        config_on = self._config(
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=99,
            nRounds=6,
            weightDecayCoefficient=0.5,
        )
        result_off = run_campaign_four(
            config=config_off,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        result_on = run_campaign_four(
            config=config_on,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        norm_off = float(result_off["telemetry"]["model_norm"][-1].mean())
        norm_on = float(result_on["telemetry"]["model_norm"][-1].mean())
        self.assertLess(norm_on, norm_off)
```

Note: `self._config(**overrides)` already applies `config.update(overrides)` after building the base config (see existing helper, `tests/test_adaptive_experiment_engine.py:74-91`), so passing `nRounds=6` and `weightDecayCoefficient=...` as overrides works without any change to the test helper itself.

- [ ] **Step 2: Run the test to verify it fails**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_experiment_engine.Campaign4EngineTests.test_weight_decay_reduces_final_model_norm -v`

Expected: FAIL — `norm_on` equals `norm_off` (the engine ignores `weightDecayCoefficient` entirely today, so both runs are numerically identical).

- [ ] **Step 3: Add the `weight_decay` parameter to `SharedDeviceWorker.__init__`**

In `basil_core/adaptive_experiment_engine.py`, replace the `__init__` signature and optimizer construction (lines 369-386):

```python
    def __init__(
        self,
        model,
        *,
        lr0: float,
        momentum: float,
        micro_batch_size: int,
        jit_compile: bool,
        adaptive_config: dict,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.micro_batch_size = int(micro_batch_size)
        self._lr = _MutableLR(float(lr0))
        self._optimizer = tf.keras.optimizers.SGD(
            learning_rate=self._lr,
            momentum=float(momentum),
            weight_decay=(float(weight_decay) if float(weight_decay) > 0.0 else None),
        )
```

- [ ] **Step 4: Thread the config value through `run_campaign_four`**

In `basil_core/adaptive_experiment_engine.py`, in the `SharedDeviceWorker(...)` call inside `run_campaign_four` (lines 938-944), add the new keyword argument:

```python
    worker = SharedDeviceWorker(
        model,
        lr0=float(config["learningRate"]),
        momentum=float(config.get("momentum", 0.0)),
        micro_batch_size=int(config.get("internalMicroBatchSize", 128)),
        jit_compile=bool(config.get("jitCompile", False)),
        adaptive_config=config,
        weight_decay=float(config.get("weightDecayCoefficient", 0.0)),
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_experiment_engine.Campaign4EngineTests.test_weight_decay_reduces_final_model_norm -v`

Expected: PASS

- [ ] **Step 6: Run the full engine test suite to check for regressions**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_experiment_engine -v`

Expected: all tests PASS, including `test_run_emits_node_events_and_complete_telemetry` and `test_second_order_gradient_matches_declared_objective` (both must be numerically unaffected since their configs have `weightDecayCoefficient` unset/0.0 → `weight_decay=None` → no-op).

- [ ] **Step 7: Commit**

```bash
git add basil_core/adaptive_experiment_engine.py tests/test_adaptive_experiment_engine.py
git commit -m "feat: wire weightDecayCoefficient into the Campaign 4 SGD optimizer

Uses TF's native decoupled weight decay (weight_decay=None when the config
value is 0.0, so every existing config's optimizer behavior is unchanged).
Unit test confirms nonzero decay measurably reduces final model_norm on a
tiny synthetic model, isolating the mechanism before the real diagnostic run."
```

---

### Task 3: Regenerate the config library

**Files:**
- Modify: `gui/configs/adaptive_study/**/*.json` (526 files), `gui/configs/adaptive_study/manifest.json`

**Interfaces:**
- Consumes: Task 1's `make_config(..., weight_decay_coefficient=0.0)` default behavior (every library config keeps `weightDecayCoefficient: 0.0` — no behavioral change, only a new field and new `runId` hashes, matching the 2026-08-08 precedent).

- [ ] **Step 1: Confirm the library is out of date**

Run: `environment/basil-noise-env/bin/python scripts/sync_adaptive_configs.py --check`

Expected: exits with `Campaign 4 config library: out of date (526 configs)` and a nonzero exit code (Task 1 changed every config's hash by adding a new field).

- [ ] **Step 2: Regenerate the library**

Run: `environment/basil-noise-env/bin/python scripts/sync_adaptive_configs.py`

Expected: `Wrote 526 Campaign 4 configs under .../gui/configs/adaptive_study`

- [ ] **Step 3: Confirm the library is now current**

Run: `environment/basil-noise-env/bin/python scripts/sync_adaptive_configs.py --check`

Expected: `Campaign 4 config library: current (526 configs)`, exit code 0.

- [ ] **Step 4: Run the full Campaign 4 test suite**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_adaptive_study_contracts tests.test_adaptive_experiment_engine tests.test_execution_policy tests.test_adaptive_worker tests.test_performance_profiles -v`

Expected: all tests PASS (this now includes `test_generated_manifest_matches_library`, which was expected to fail after Task 1 and should pass now that the library is regenerated).

- [ ] **Step 5: Commit**

```bash
git add gui/configs/adaptive_study/
git commit -m "chore: regenerate Campaign 4 config library with weightDecayCoefficient

Every config gains the new default-off field; runId hashes change for all
526 configs (expected -- same effect as the 2026-08-08 adaptive-EBM retune).
No config's semantic content beyond the new field changes."
```

---

### Task 4: Build and queue the Phase A validation set

**Files:**
- None created (one-off queue construction, matching this session's established pattern for validation batches — see the 12-config and 1-config batches built earlier via inline Python).

**Interfaces:**
- Consumes: `make_config(..., weight_decay_coefficient=1e-4)` (Task 1), `apply_performance_profile` and `load_worker_profile` (existing, from `gui.adaptive_study` and `gui.runtime_estimator`).

- [ ] **Step 1: Confirm the queue is empty before writing**

Run:
```bash
python3 -c "import json; q=json.load(open('gui/queue_state.json')); print('current queue:', len(q))"
```

Expected: `current queue: 0`. If nonzero, stop and ask the user before overwriting — do not silently discard queued work.

- [ ] **Step 2: Build the 3 configs and write the queue**

Run:
```python
import json
import sys
sys.path.insert(0, ".")
from gui.adaptive_study import make_config, apply_performance_profile, is_completed
from gui.runtime_estimator import load_worker_profile

def phase_a_config(**kw):
    return make_config(
        split="nonIID",
        approach="merged",
        seed=2025,
        phase="diagnostic",
        rounds=100,
        optimizer_state_mode="persistent",
        mitigation="none",
        weight_decay_coefficient=1e-4,
        **kw,
    )

configs = [
    phase_a_config(environment="clean"),
    phase_a_config(environment="noise", sigma=0.4),
    phase_a_config(environment="noise", sigma=0.6),
]

for config in configs:
    print(config["runId"], "|", config["experimentName"], "| already completed:", is_completed(config))

profile = load_worker_profile("experiments/results4/campaign4/performance_profile.json")
profiled = [apply_performance_profile(config, profile) for config in configs]

with open("gui/queue_state.json", "w") as f:
    json.dump(profiled, f, indent=2)
print("\nwrote gui/queue_state.json with", len(profiled), "items")
```

Expected: 3 lines printed, all `already completed: False` (this is new territory — no prior run used `weightDecayCoefficient=1e-4`), then `wrote gui/queue_state.json with 3 items`.

- [ ] **Step 3: Verify the queued configs are correct**

Run:
```bash
python3 -c "
import json
q = json.load(open('gui/queue_state.json'))
assert len(q) == 3
for c in q:
    assert c['weightDecayCoefficient'] == 1e-4
    assert c['optimizerStateMode'] == 'persistent'
    assert c['mitigation'] == 'none'
    assert c['phase'] == 'diagnostic'
    assert c['nRounds'] == 100
    print(c['environment'], c.get('channelNoiseSigma'), c['runId'], c['experimentName'])
print('all 3 configs verified')
"
```

Expected: 3 lines (one `clean`, one `noise 0.4`, one `noise 0.6`), then `all 3 configs verified`.

- [ ] **Step 4: Hand off to the user**

Tell the user the queue is ready and ask them to run it via `python run_gui.py` → **Run Queue** (per this session's established practice: GPU training is never launched directly by the agent). Do not create a git commit for this task — `gui/queue_state.json` is gitignored (`.gitignore:9`, confirmed present from the 2026-08-04 orphan-worker fix session).

---

## After Phase A completes

Once the user runs the 3 queued configs, pull `model_norm` telemetry and final/peak accuracy for all three and compare against the existing persistent-arm baselines (clean 0.468, noise-0.6-none 0.100, per `docs/superpowers/specs/2026-08-09-weight-norm-control-design.md` "Open questions"). That data — not a guess — determines Phase C's adaptive-controller bounds, which gets its own follow-up spec per the design doc.
