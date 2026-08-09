# Phase C: Adaptive Weight-Decay Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a genuinely reactive weight-decay controller (bounded, EMA-smoothed, rate-limited -- same pattern as adaptive EBM) that reads each node's own weight-norm growth and sets its weight-decay coefficient once per round, using bounds grounded in the Phase A calibration sweep. Then queue a 3-config validation set to check it reaches the working range on its own at sigma=0.6 and stays inert at lower sigma.

**Architecture:** `SharedDeviceWorker` is refactored so its `weight_decay` is always a `tf.Variable` (reassignable per node-visit via `train()`'s new argument, not fixed at construction) -- this serves both Phase A's static case (same value every visit, behavior unchanged) and Phase C's adaptive case (a new value each round) through one mechanism. The control law itself runs as plain Python in the round loop (not TF ops), reading each node's previous-round `model_norm` (the only value the engine actually computes -- at round end, not round start) against its own round-0 reference, and is persisted on new `LogicalNode` fields.

**Tech Stack:** Python 3.12, TensorFlow 2.18.0, existing Campaign 4 config-generation/engine/test modules.

## Global Constraints

- `adaptiveWeightDecayMode` defaults to `"none"` on every config-generation path -- Phase A's static `weightDecayCoefficient` field and behavior are completely unaffected when this is `"none"`.
- `make_config` rejects `adaptiveWeightDecayMode="adaptive"` when `useChannelNoise` is false, raising `ValueError` -- mirrors the existing `use_ebm and not has_noise` check exactly (reject at construction, not a silent no-op).
- `coefficient_min=1e-6` is a real floor, not `0.0` -- the multiplicative rate limiter (`previous / max_change_factor`, `previous * max_change_factor`) breaks permanently if the coefficient is ever allowed to reach exactly `0.0`. See `docs/superpowers/specs/2026-08-09-adaptive-weight-decay-phase-c-design.md`'s "Control law" section for the full correction.
- Control-law constants (all grounded in the Phase A 8-point sweep, not guessed): `target=2.0`, `gain=0.025`, `coefficient_min=1e-6`, `coefficient_max=0.05`, `beta=0.9`, `max_change_factor=2.0`.
- The controller updates once per round, not per step -- no new per-step compute.
- Validation configs use `approach=merged`, `nonIID`, `optimizerStateMode=persistent`, `mitigation=none`, `phase=diagnostic`, seed 2025, 100 rounds (spec's "Validation plan").
- Do not launch GPU training directly -- build and queue configs into `gui/queue_state.json` for the user to run via the GUI (established practice this session).
- Editing `gui/campaign4.py` and `basil_core/campaign4_engine.py` touches two `PROVENANCE_FILES` entries; regenerating the config library and re-running `sync_campaign4_configs.py --check` afterward is required (same precedent as the 2026-08-08 adaptive-EBM retune and Phase A).

---

### Task 1: Add `adaptiveWeightDecay*` config fields, validation, and contract test

**Files:**
- Modify: `gui/campaign4.py:240-318` (`make_config` signature, validation, dict)
- Test: `tests/test_campaign4_contracts.py`

**Interfaces:**
- Produces: `make_config(..., adaptive_weight_decay_mode: str = "none", adaptive_weight_decay_target_ratio: float = 2.0, adaptive_weight_decay_gain: float = 0.025, adaptive_weight_decay_coefficient_min: float = 1e-6, adaptive_weight_decay_coefficient_max: float = 0.05, adaptive_weight_decay_beta: float = 0.9, adaptive_weight_decay_max_change_factor: float = 2.0)` -- stored on the config dict with the exact camelCase keys listed in Step 3 below.

- [ ] **Step 1: Write the failing contract test**

Add to `tests/test_campaign4_contracts.py` (near `test_weight_decay_defaults_to_zero_and_is_named_when_enabled`, reusing the same `make_config` import already present):

```python
    def test_adaptive_weight_decay_defaults_off_and_requires_noise(self):
        default_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
        )
        self.assertEqual(default_config["adaptiveWeightDecayMode"], "none")
        self.assertEqual(default_config["adaptiveWeightDecayTargetRatio"], 2.0)
        self.assertEqual(default_config["adaptiveWeightDecayGain"], 0.025)
        self.assertEqual(default_config["adaptiveWeightDecayCoefficientMin"], 1e-6)
        self.assertEqual(default_config["adaptiveWeightDecayCoefficientMax"], 0.05)
        self.assertEqual(default_config["adaptiveWeightDecayBeta"], 0.9)
        self.assertEqual(default_config["adaptiveWeightDecayMaxChangeFactor"], 2.0)

        adaptive_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
            adaptive_weight_decay_mode="adaptive",
        )
        self.assertEqual(adaptive_config["adaptiveWeightDecayMode"], "adaptive")
        self.assertNotEqual(adaptive_config["runId"], default_config["runId"])

        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=2025,
                adaptive_weight_decay_mode="adaptive",
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_contracts.Campaign4ContractTests.test_adaptive_weight_decay_defaults_off_and_requires_noise -v`

Expected: FAIL with `KeyError: 'adaptiveWeightDecayMode'`.

- [ ] **Step 3: Add the parameters to `make_config` and wire them through**

In `gui/campaign4.py`, in the `make_config` signature (currently ends `weight_decay_coefficient: float = 0.0,` at line 256), add:

```python
    weight_decay_coefficient: float = 0.0,
    adaptive_weight_decay_mode: str = "none",
    adaptive_weight_decay_target_ratio: float = 2.0,
    adaptive_weight_decay_gain: float = 0.025,
    adaptive_weight_decay_coefficient_min: float = 1e-6,
    adaptive_weight_decay_coefficient_max: float = 0.05,
    adaptive_weight_decay_beta: float = 0.9,
    adaptive_weight_decay_max_change_factor: float = 2.0,
) -> dict:
```

In the validation block (currently lines 302-303 read):

```python
    if use_ebm and not has_noise:
        raise ValueError("EBM is valid only when channel noise is active.")
```

add immediately after:

```python
    if adaptive_weight_decay_mode not in ("none", "adaptive"):
        raise ValueError(
            f"Unknown adaptive weight decay mode: {adaptive_weight_decay_mode}"
        )
    if adaptive_weight_decay_mode == "adaptive" and not has_noise:
        raise ValueError(
            "Adaptive weight decay is valid only when channel noise is active."
        )
```

In the config dict literal, add the new fields right after the existing `"weightDecayCoefficient": float(weight_decay_coefficient),` line:

```python
        "weightDecayCoefficient": float(weight_decay_coefficient),
        "adaptiveWeightDecayMode": str(adaptive_weight_decay_mode),
        "adaptiveWeightDecayTargetRatio": float(adaptive_weight_decay_target_ratio),
        "adaptiveWeightDecayGain": float(adaptive_weight_decay_gain),
        "adaptiveWeightDecayCoefficientMin": float(adaptive_weight_decay_coefficient_min),
        "adaptiveWeightDecayCoefficientMax": float(adaptive_weight_decay_coefficient_max),
        "adaptiveWeightDecayBeta": float(adaptive_weight_decay_beta),
        "adaptiveWeightDecayMaxChangeFactor": float(adaptive_weight_decay_max_change_factor),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_contracts.Campaign4ContractTests.test_adaptive_weight_decay_defaults_off_and_requires_noise -v`

Expected: PASS

- [ ] **Step 5: Run the full contract test suite to check for regressions**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_contracts -v`

Expected: all tests PASS except `test_generated_manifest_matches_library`, which is expected to fail here (library not yet regenerated -- Task 5 handles that; do not fix it in this task).

- [ ] **Step 6: Commit**

```bash
git add gui/campaign4.py tests/test_campaign4_contracts.py
git commit -m "feat: add adaptiveWeightDecay config fields for Phase C controller

Orthogonal, default-off (adaptiveWeightDecayMode=none everywhere unless
explicitly set) per docs/superpowers/specs/2026-08-09-adaptive-weight-decay-phase-c-design.md.
Rejects adaptive mode without channel noise, mirroring EBM's own
use_ebm-requires-noise validation exactly."
```

---

### Task 2: Add the plain-Python control-law function and its unit test

**Files:**
- Modify: `basil_core/campaign4_engine.py` (new module-level function, placed near `_bounded_adaptive_coefficient`, currently ending around line 316)
- Test: `tests/test_campaign4_engine.py`

**Interfaces:**
- Produces: `_weight_decay_control_step(*, previous_coefficient: float, previous_smoothed_ratio: float, model_norm: float, model_norm_round0: float, target_ratio: float, gain: float, coefficient_min: float, coefficient_max: float, beta: float, max_change_factor: float) -> tuple[float, float]` -- returns `(applied_coefficient, smoothed_ratio)`.

- [ ] **Step 1: Write the failing unit test**

Add to `tests/test_campaign4_engine.py`, inside `Campaign4EngineTests` (near `test_adaptive_coefficient_is_bounded_and_rate_limited`; add `_weight_decay_control_step` to the existing `from basil_core.campaign4_engine import (...)` block at the top of the file):

```python
    def test_weight_decay_control_step_is_bounded_and_rate_limited(self):
        applied, smoothed = _weight_decay_control_step(
            previous_coefficient=1e-6,
            previous_smoothed_ratio=1.0,
            model_norm=346.6,
            model_norm_round0=56.4,
            target_ratio=2.0,
            gain=0.025,
            coefficient_min=1e-6,
            coefficient_max=0.05,
            beta=0.9,
            max_change_factor=2.0,
        )
        # growth_ratio = 346.6/56.4 approx 6.15; smoothed = 0.9*1.0 + 0.1*6.15 = 1.515
        # desired = clip(0.025*(1.515-2.0), 1e-6, 0.05) -> clipped to the coefficient_min floor
        # since (1.515-2.0) is negative -- rate-limited against a previous of 1e-6
        self.assertGreaterEqual(applied, 1e-6)
        self.assertLessEqual(applied, 1e-6 * 2.0)
        self.assertAlmostEqual(smoothed, 1.515, places=3)

        # A node already at a high coefficient facing continued high growth
        # should ramp toward the ceiling, not snap there in one round.
        applied_high, _ = _weight_decay_control_step(
            previous_coefficient=0.01,
            previous_smoothed_ratio=4.0,
            model_norm=346.6,
            model_norm_round0=56.4,
            target_ratio=2.0,
            gain=0.025,
            coefficient_min=1e-6,
            coefficient_max=0.05,
            beta=0.9,
            max_change_factor=2.0,
        )
        self.assertGreater(applied_high, 0.01)
        self.assertLessEqual(applied_high, 0.01 * 2.0)

        # Zero previous must not permanently lock the coefficient at zero
        # (the multiplicative-rate-limiter degeneracy this floor exists to avoid).
        applied_from_floor, _ = _weight_decay_control_step(
            previous_coefficient=1e-6,
            previous_smoothed_ratio=5.0,
            model_norm=346.6,
            model_norm_round0=56.4,
            target_ratio=2.0,
            gain=0.025,
            coefficient_min=1e-6,
            coefficient_max=0.05,
            beta=0.9,
            max_change_factor=2.0,
        )
        self.assertGreater(applied_from_floor, 1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine.Campaign4EngineTests.test_weight_decay_control_step_is_bounded_and_rate_limited -v`

Expected: FAIL with `ImportError` or `NameError` (`_weight_decay_control_step` doesn't exist yet).

- [ ] **Step 3: Implement the control-law function**

In `basil_core/campaign4_engine.py`, add this function immediately after `_bounded_adaptive_coefficient` (which currently ends around line 316, just before the `class LogicalNode` dataclass):

```python
def _weight_decay_control_step(
    *,
    previous_coefficient: float,
    previous_smoothed_ratio: float,
    model_norm: float,
    model_norm_round0: float,
    target_ratio: float,
    gain: float,
    coefficient_min: float,
    coefficient_max: float,
    beta: float,
    max_change_factor: float,
) -> tuple[float, float]:
    """Return (applied_coefficient, smoothed_ratio) for Phase C's weight-decay controller.

    Plain Python, not TF ops: this runs once per round in the round loop,
    not inside a per-step traced function, so it operates on the plain
    floats _state_diagnostics already returns. ``coefficient_min`` must be
    a small positive floor, never 0.0 -- the multiplicative rate limiter
    below would otherwise lock the coefficient at a permanent 0.0 once it
    reached there.
    """
    growth_ratio = model_norm / max(model_norm_round0, EPSILON)
    smoothed_ratio = beta * previous_smoothed_ratio + (1.0 - beta) * growth_ratio
    desired = max(
        coefficient_min,
        min(gain * (smoothed_ratio - target_ratio), coefficient_max),
    )
    previous = max(coefficient_min, min(previous_coefficient, coefficient_max))
    lower = previous / max(max_change_factor, 1.0)
    upper = previous * max(max_change_factor, 1.0)
    applied = max(lower, min(desired, upper))
    applied = max(coefficient_min, min(applied, coefficient_max))
    return applied, smoothed_ratio
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine.Campaign4EngineTests.test_weight_decay_control_step_is_bounded_and_rate_limited -v`

Expected: PASS

- [ ] **Step 5: Run the full engine test suite to check for regressions**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine -v`

Expected: all tests PASS (this function is standalone, not yet wired into `run_campaign_four` -- Task 4 does that).

- [ ] **Step 6: Commit**

```bash
git add basil_core/campaign4_engine.py tests/test_campaign4_engine.py
git commit -m "feat: add plain-Python weight-decay control-law function

Bounded, EMA-smoothed, rate-limited -- same pattern as
_bounded_adaptive_coefficient, but plain floats since it runs once per
round in the round loop, not per-step in-graph. coefficient_min=1e-6 is a
real floor: a permanent 0.0 would lock the multiplicative rate limiter
there forever, verified by the zero-previous unit test case."
```

---

### Task 3: Make `SharedDeviceWorker`'s weight decay a mutable `tf.Variable`

**Files:**
- Modify: `basil_core/campaign4_engine.py:369-389` (`SharedDeviceWorker.__init__`), `basil_core/campaign4_engine.py:567-596` (`SharedDeviceWorker.train`)
- Test: `tests/test_campaign4_engine.py`

**Interfaces:**
- Consumes: nothing new from prior tasks (independent of Task 2's function).
- Produces: `SharedDeviceWorker.train(..., weight_decay_coefficient: float)` -- new required keyword argument; every existing caller of `.train(...)` must be updated in Task 4 (not this task -- this task only changes the worker itself and proves the mechanism, `run_campaign_four` still passes the Phase A static value unchanged).

**Note:** this task changes `train()`'s signature, which `run_campaign_four` already calls (Phase A code, unmodified since that work). To keep this task's diff isolated and testable on its own, also update the two existing call sites in `run_campaign_four` (the `full_consensus` branch and the pairwise branch) to pass `weight_decay_coefficient=float(config.get("weightDecayCoefficient", 0.0))` -- i.e., today's static Phase A value, unchanged in effect. Task 4 replaces that expression with the real controller call.

- [ ] **Step 1: Write the failing regression test for the reassignment mechanism**

Add to `tests/test_campaign4_engine.py`, inside `Campaign4EngineTests`:

```python
    def test_weight_decay_variable_updates_live_without_retrace(self):
        tf.keras.utils.set_random_seed(21)
        model = self.TinyModel()
        worker = SharedDeviceWorker(
            model,
            lr0=0.1,
            momentum=0.0,
            micro_batch_size=4,
            jit_compile=False,
            adaptive_config={
                "adaptiveEbmTargetRatioBase": 0.05,
                "adaptiveEbmStressGain": 0.25,
                "adaptiveEbmRatioMin": 0.02,
                "adaptiveEbmRatioMax": 0.35,
                "adaptiveEbmCoefficientMin": 1e-6,
                "adaptiveEbmCoefficientMax": 0.01,
                "adaptiveEbmBeta": 0.9,
                "adaptiveEbmMaxChangeFactor": 2.0,
            },
        )
        params = worker.export()
        optimizer_state = worker.zero_optimizer_state()
        x = tf.ones((4, 4), dtype=tf.float32)
        y = tf.zeros(4, dtype=tf.int32)

        params_zero_decay, optimizer_state, _ = worker.train(
            params,
            optimizer_state=optimizer_state,
            data_iterator=iter([(x, y)] * 10),
            first_batch=(x, y),
            total_steps=5,
            lr=0.1,
            ebm_mode="none",
            initial_coefficient=0.0,
            stress=0.0,
            prox_mu=0.0,
            weight_decay_coefficient=0.0,
        )
        norm_zero_decay = float(tf.linalg.global_norm(params_zero_decay).numpy())

        params_with_decay, _, _ = worker.train(
            params,
            optimizer_state=optimizer_state,
            data_iterator=iter([(x, y)] * 10),
            first_batch=(x, y),
            total_steps=5,
            lr=0.1,
            ebm_mode="none",
            initial_coefficient=0.0,
            stress=0.0,
            prox_mu=0.0,
            weight_decay_coefficient=0.5,
        )
        norm_with_decay = float(tf.linalg.global_norm(params_with_decay).numpy())

        self.assertLess(norm_with_decay, norm_zero_decay)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine.Campaign4EngineTests.test_weight_decay_variable_updates_live_without_retrace -v`

Expected: FAIL with `TypeError: train() got an unexpected keyword argument 'weight_decay_coefficient'`.

- [ ] **Step 3: Make `weight_decay` a `tf.Variable` in `__init__`**

In `basil_core/campaign4_engine.py`, replace the `__init__` signature and optimizer construction (currently lines 369-389):

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
        self._weight_decay_var = tf.Variable(
            float(weight_decay), trainable=False, dtype=tf.float32
        )
        self._optimizer = tf.keras.optimizers.SGD(
            learning_rate=self._lr,
            momentum=float(momentum),
            weight_decay=self._weight_decay_var,
        )
        self._optimizer.build(self.model.trainable_weights)
```

Note: this removes the `if float(weight_decay) > 0.0 else None` guard from Phase A -- a `tf.Variable` initialized to `0.0` and never reassigned (Phase A's own path, once Task 4 preserves that behavior) produces the same `-variable*0*lr = 0` no-op `_apply_weight_decay` already computes for any zero decay value, so this is not a behavior change; it was only a construction-time detail, not a training-time one.

- [ ] **Step 4: Add the `weight_decay_coefficient` parameter to `train()`**

In `basil_core/campaign4_engine.py`, in the `train()` signature (currently lines 567-580), add the new parameter after `prox_mu: float,`:

```python
    def train(
        self,
        params,
        *,
        optimizer_state,
        data_iterator,
        first_batch,
        total_steps: int,
        lr: float,
        ebm_mode: str,
        initial_coefficient: float,
        stress: float,
        prox_mu: float,
        weight_decay_coefficient: float,
    ):
        self.load(params)
        for reference, weight in zip(self._ref_params, self.model.trainable_weights):
            reference.assign(tf.cast(weight, tf.float32))
        self._lr.assign(float(lr))
        self._weight_decay_var.assign(float(weight_decay_coefficient))
        self._coefficient.assign(float(initial_coefficient))
```

(The `self._weight_decay_var.assign(...)` line is new; everything else in this block is unchanged from the current code.)

- [ ] **Step 5: Update `run_campaign_four`'s two existing `.train(...)` call sites**

In `basil_core/campaign4_engine.py`, `run_campaign_four`'s `full_consensus` branch (`worker.train(...)` call, currently passing `prox_mu=0.0,` as its last argument) -- add immediately after:

```python
                    prox_mu=0.0,
                    weight_decay_coefficient=float(config.get("weightDecayCoefficient", 0.0)),
                )
```

And the pairwise branch's `worker.train(...)` call (currently ending `prox_mu=mu,`) -- add immediately after:

```python
                    prox_mu=mu,
                    weight_decay_coefficient=float(config.get("weightDecayCoefficient", 0.0)),
                )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine.Campaign4EngineTests.test_weight_decay_variable_updates_live_without_retrace -v`

Expected: PASS

- [ ] **Step 7: Run the full engine test suite, and Phase A's weight-decay test specifically, to confirm no regression**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine -v`

Expected: all tests PASS, including `test_weight_decay_reduces_final_model_norm` (Phase A's own test) -- confirms the `tf.Variable` refactor produces identical behavior to Phase A's original `float | None` construction-time approach.

- [ ] **Step 8: Commit**

```bash
git add basil_core/campaign4_engine.py tests/test_campaign4_engine.py
git commit -m "refactor: make SharedDeviceWorker.weight_decay a mutable tf.Variable

Serves both Phase A's static case (same value every train() call, behavior
unchanged -- verified by the existing test_weight_decay_reduces_final_model_norm
still passing) and Phase C's adaptive case (a new value each round) through
one mechanism, mirroring how _MutableLR already handles learning rate.
Regression test locks in that reassigning the Variable after tracing takes
effect on the next call, without retracing."
```

---

### Task 4: Wire the controller into the round loop, with telemetry

**Files:**
- Modify: `basil_core/campaign4_engine.py` (`LogicalNode` dataclass, `run_campaign_four`'s round loop and telemetry dict)
- Test: `tests/test_campaign4_engine.py`

**Interfaces:**
- Consumes: `_weight_decay_control_step` (Task 2), `SharedDeviceWorker.train(..., weight_decay_coefficient=...)` (Task 3).
- Produces: two new telemetry keys, `adaptive_weight_decay_coefficient` and `adaptive_weight_decay_growth_ratio`, shape `(rounds, nodes)`, matching every other per-node telemetry array in this file.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_campaign4_engine.py`, inside `Campaign4EngineTests` (uses the existing `self._config`/`self._data`/`self.TinyModel` helpers):

```python
    def test_adaptive_weight_decay_ramps_up_when_norm_grows(self):
        train, test, metadata = self._data(seed=55)
        config = self._config(
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=55,
            nRounds=6,
            adaptiveWeightDecayMode="adaptive",
            adaptiveWeightDecayTargetRatio=2.0,
            adaptiveWeightDecayGain=0.025,
            adaptiveWeightDecayCoefficientMin=1e-6,
            adaptiveWeightDecayCoefficientMax=0.05,
            adaptiveWeightDecayBeta=0.9,
            adaptiveWeightDecayMaxChangeFactor=2.0,
        )
        result = run_campaign_four(
            config=config,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        coefficient = result["telemetry"]["adaptive_weight_decay_coefficient"]
        growth_ratio = result["telemetry"]["adaptive_weight_decay_growth_ratio"]
        self.assertEqual(coefficient.shape, (6, 3))
        self.assertEqual(growth_ratio.shape, (6, 3))
        # Round 0 has no prior-round norm yet -- must start at the exact 0.0
        # default (not the coefficient_min floor, which would falsely claim
        # "evidence of growth" before any has been observed).
        self.assertTrue(np.all(coefficient[0] == 0.0))
        # From round 1 on, the controller is active -- every value must stay
        # within the configured bounds.
        self.assertTrue(np.all(coefficient[1:] >= 1e-6 - 1e-9))
        self.assertTrue(np.all(coefficient <= 0.05 + 1e-9))

    def test_adaptive_weight_decay_off_matches_mode_none_exactly(self):
        train, test, metadata = self._data(seed=56)
        base_kwargs = dict(
            environment="noise",
            mitigation="none",
            sigma=0.4,
            seed=56,
            nRounds=4,
        )
        config_none = self._config(**base_kwargs, adaptiveWeightDecayMode="none")
        result_none = run_campaign_four(
            config=config_none,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        self.assertTrue(
            np.all(result_none["telemetry"]["adaptive_weight_decay_coefficient"] == 0.0)
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine.Campaign4EngineTests.test_adaptive_weight_decay_ramps_up_when_norm_grows tests.test_campaign4_engine.Campaign4EngineTests.test_adaptive_weight_decay_off_matches_mode_none_exactly -v`

Expected: FAIL with `KeyError: 'adaptive_weight_decay_coefficient'` (telemetry key doesn't exist yet).

- [ ] **Step 3: Add new fields to `LogicalNode`**

In `basil_core/campaign4_engine.py`, in the `LogicalNode` dataclass (currently lines 340-360), add after the existing `adaptive_coefficient: float = 0.0,` line:

```python
    adaptive_coefficient: float = 0.0
    model_norm: float = 0.0
    model_norm_round0: float = 0.0
    weight_decay_growth_ratio_ema: float = 1.0
    adaptive_weight_decay_coefficient: float = 0.0
```

(`weight_decay_growth_ratio_ema` defaults to `1.0`, not `0.0` -- it represents a ratio, and a node that has never grown has ratio 1.0 by definition, i.e. `model_norm == model_norm_round0`. `adaptive_weight_decay_coefficient` defaults to `0.0` deliberately, distinct from the `coefficient_min` floor: round 0 has no prior-round norm to react to yet, so it must start at "no evidence of growth" rather than pre-seeded at the floor -- Step 5 below handles this explicitly.)

- [ ] **Step 4: Add the two new telemetry arrays**

In `basil_core/campaign4_engine.py`, in the telemetry dict initialization (currently around line 786-789, alongside `"model_norm": np.zeros(shape, dtype=np.float32),`), add:

```python
        "model_norm": np.zeros(shape, dtype=np.float32),
        "adaptive_weight_decay_coefficient": np.zeros(shape, dtype=np.float32),
        "adaptive_weight_decay_growth_ratio": np.zeros(shape, dtype=np.float32),
```

- [ ] **Step 5: Compute the coefficient before training, and record telemetry after**

In `basil_core/campaign4_engine.py`'s pairwise round loop (`run_campaign_four`, the `else:` branch handling non-`full_consensus` rounds), immediately before the `optimizer_state = (...)` block that precedes the `worker.train(...)` call (i.e., right after the CART block that currently ends around `telemetry["cart_seconds"][round_id, node_id] = (...)`), add:

```python
                adaptive_wd_mode = str(config.get("adaptiveWeightDecayMode", "none"))
                if adaptive_wd_mode == "adaptive" and round_id > 0:
                    applied_wd, node.weight_decay_growth_ratio_ema = _weight_decay_control_step(
                        previous_coefficient=max(
                            node.adaptive_weight_decay_coefficient,
                            float(config["adaptiveWeightDecayCoefficientMin"]),
                        ),
                        previous_smoothed_ratio=node.weight_decay_growth_ratio_ema,
                        model_norm=node.model_norm,
                        model_norm_round0=node.model_norm_round0,
                        target_ratio=float(config["adaptiveWeightDecayTargetRatio"]),
                        gain=float(config["adaptiveWeightDecayGain"]),
                        coefficient_min=float(config["adaptiveWeightDecayCoefficientMin"]),
                        coefficient_max=float(config["adaptiveWeightDecayCoefficientMax"]),
                        beta=float(config["adaptiveWeightDecayBeta"]),
                        max_change_factor=float(config["adaptiveWeightDecayMaxChangeFactor"]),
                    )
                    node.adaptive_weight_decay_coefficient = applied_wd
                elif adaptive_wd_mode != "adaptive":
                    node.adaptive_weight_decay_coefficient = float(
                        config.get("weightDecayCoefficient", 0.0)
                    )
                # else: round 0 under adaptive mode -- no prior-round norm yet,
                # node.adaptive_weight_decay_coefficient stays at its 0.0 default.
```

Then change the existing `worker.train(...)` call's `weight_decay_coefficient=float(config.get("weightDecayCoefficient", 0.0)),` line (added in Task 3 Step 5) to:

```python
                    weight_decay_coefficient=node.adaptive_weight_decay_coefficient,
```

Immediately after the existing `signature, model_norm = _state_diagnostics(node.params)` line, add:

```python
                signature, model_norm = _state_diagnostics(node.params)
                if round_id == 0:
                    node.model_norm_round0 = model_norm
                node.model_norm = model_norm
```

And in the telemetry recording block (currently the lines starting `telemetry["model_norm"][round_id, node_id] = model_norm`), add immediately after:

```python
                telemetry["model_norm"][round_id, node_id] = model_norm
                telemetry["adaptive_weight_decay_coefficient"][round_id, node_id] = (
                    node.adaptive_weight_decay_coefficient
                )
                telemetry["adaptive_weight_decay_growth_ratio"][round_id, node_id] = (
                    node.weight_decay_growth_ratio_ema
                )
```

**Note on the `full_consensus` branch:** that branch's `worker.train(...)` call (Task 3 Step 5) keeps `weight_decay_coefficient=float(config.get("weightDecayCoefficient", 0.0))` unchanged -- `adaptiveWeightDecayMode` only activates in the pairwise (non-full-consensus) path, matching how the parent spec's validation set and every existing adaptive-EBM diagnostic already only exercises the pairwise ring. Full consensus stays Phase-A-only, consistent with Campaign4Plan.md's existing scoping of the "ideal ceiling" reference as a separate, simpler protocol.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine.Campaign4EngineTests.test_adaptive_weight_decay_ramps_up_when_norm_grows tests.test_campaign4_engine.Campaign4EngineTests.test_adaptive_weight_decay_off_matches_mode_none_exactly -v`

Expected: PASS

- [ ] **Step 7: Run the full engine test suite and the full contract test suite to check for regressions**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_engine tests.test_campaign4_contracts tests.test_campaign4_execution tests.test_campaign4_worker tests.test_campaign4_performance_profiles -v`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add basil_core/campaign4_engine.py tests/test_campaign4_engine.py
git commit -m "feat: wire the adaptive weight-decay controller into the round loop

Computes each node's coefficient once per round, before its local training,
from the previous round's model_norm (the only value the engine actually
has -- computed at round end). Round 0 has no prior norm to react to and
starts at the safe 0.0 default, not the coefficient_min floor, since there
is no evidence of growth yet to justify even a minimal decay. New telemetry
(adaptive_weight_decay_coefficient, adaptive_weight_decay_growth_ratio)
mirrors the existing ebm_coefficient/stress_ema shape exactly."
```

---

### Task 5: Regenerate the config library

**Files:**
- Modify: `gui/configs/campaign4/**/*.json` (526 files), `gui/configs/campaign4/manifest.json`

- [ ] **Step 1: Confirm the library is out of date**

Run: `environment/basil-noise-env/bin/python scripts/sync_campaign4_configs.py --check`

Expected: `Campaign 4 config library: out of date (526 configs)`, nonzero exit code.

- [ ] **Step 2: Regenerate**

Run: `environment/basil-noise-env/bin/python scripts/sync_campaign4_configs.py`

Expected: `Wrote 526 Campaign 4 configs under .../gui/configs/campaign4`

- [ ] **Step 3: Confirm the library is now current**

Run: `environment/basil-noise-env/bin/python scripts/sync_campaign4_configs.py --check`

Expected: `Campaign 4 config library: current (526 configs)`, exit code 0.

- [ ] **Step 4: Run the full Campaign 4 test suite**

Run: `environment/basil-noise-env/bin/python -m unittest tests.test_campaign4_contracts tests.test_campaign4_engine tests.test_campaign4_execution tests.test_campaign4_worker tests.test_campaign4_performance_profiles -v`

Expected: all tests PASS, including `test_generated_manifest_matches_library`.

- [ ] **Step 5: Commit**

```bash
git add gui/configs/campaign4/
git commit -m "chore: regenerate Campaign 4 config library with adaptiveWeightDecay fields

Every config gains the new default-off fields; runId hashes change for all
526 configs (expected -- same effect as every prior provenance-file edit
this campaign). No config's semantic content beyond the new fields changes."
```

---

### Task 6: Build and queue the Phase C validation set

**Files:**
- None created (one-off queue construction, matching this session's established pattern).

**Interfaces:**
- Consumes: `make_config(..., adaptive_weight_decay_mode="adaptive")` (Task 1), `apply_performance_profile`/`load_worker_profile` (existing).

- [ ] **Step 1: Confirm the queue is empty before writing**

Run:
```bash
python3 -c "import json; q=json.load(open('gui/queue_state.json')); print('current queue:', len(q))"
```

Expected: `current queue: 0`. If nonzero, stop and ask the user before overwriting.

- [ ] **Step 2: Build the 3 configs and write the queue**

Run:
```python
import json
import sys
sys.path.insert(0, ".")
from gui.campaign4 import make_config, apply_performance_profile, is_completed
from gui.runtime_estimator import load_worker_profile

def phase_c_config(sigma):
    return make_config(
        split="nonIID",
        approach="merged",
        environment="noise",
        mitigation="none",
        sigma=sigma,
        seed=2025,
        phase="diagnostic",
        rounds=100,
        optimizer_state_mode="persistent",
        adaptive_weight_decay_mode="adaptive",
    )

configs = [phase_c_config(sigma) for sigma in (0.6, 0.4, 0.2)]

for c in configs:
    print(c["runId"], "|", c["conditionId"], "|", c["experimentName"], "| already completed:", is_completed(c))

profile = load_worker_profile("experiments/results4/campaign4/performance_profile.json")
profiled = [apply_performance_profile(c, profile) for c in configs]

with open("gui/queue_state.json", "w") as f:
    json.dump(profiled, f, indent=2)
print("\nwrote gui/queue_state.json with", len(profiled), "items")
```

Expected: 3 lines printed, all `already completed: False`, then `wrote gui/queue_state.json with 3 items`.

- [ ] **Step 3: Verify the queued configs are correct**

Run:
```bash
python3 -c "
import json, sys
sys.path.insert(0, '.')
from gui.campaign4 import is_completed
q = json.load(open('gui/queue_state.json'))
assert len(q) == 3
for c in q:
    assert c['adaptiveWeightDecayMode'] == 'adaptive'
    assert c['optimizerStateMode'] == 'persistent'
    assert c['mitigation'] == 'none'
    assert c['phase'] == 'diagnostic'
    assert c['nRounds'] == 100
    assert c['seed'] == 2025
    assert not is_completed(c)
    print(c['channelNoiseSigma'], c['runId'], c['experimentName'])
print('all 3 configs verified')
"
```

Expected: 3 lines (sigma 0.6, 0.4, 0.2), then `all 3 configs verified`.

- [ ] **Step 4: Hand off to the user**

Tell the user the queue is ready and ask them to run it via `python runGui.py` -> **Run Queue**. Do not create a git commit for this task -- `gui/queue_state.json` is a runtime artifact (see the Phase A plan's Task 4 for the same convention; note it is git-tracked despite `.gitignore:235`, a pre-existing housekeeping inconsistency documented in Phase A's ledger, not something to fix here).

---

## After Phase C completes

Once the user runs the 3 queued configs, pull `adaptive_weight_decay_coefficient`, `adaptive_weight_decay_growth_ratio`, `model_norm`, and final/peak accuracy for all three. Compare sigma=0.6's result against the fixed-3e-2 result (0.345 final) from Phase A's sweep -- does the controller reach and hold in the working range on its own? Compare sigma=0.2/0.4 against their own no-decay baselines -- does the controller correctly stay near the `coefficient_min` floor when growth is mild? Both of the spec's "Open questions" must be answered from this real data, not assumed, before considering any further sweep or combining this mechanism with EBM/SS.
