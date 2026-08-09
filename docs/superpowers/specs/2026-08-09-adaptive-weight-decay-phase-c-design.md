# Phase C: Adaptive Weight-Decay Controller

**Status:** design approved, not yet implemented
**Date:** August 9, 2026
**Scope:** Campaign 4, `merged` approach, nonIID, noise environments (extends
`docs/superpowers/specs/2026-08-09-weight-norm-control-design.md`)

## Background

Phase A (fixed weight decay) confirmed the mechanism: at sigma=0.6,
persistent-arm, a fixed `weightDecayCoefficient` directly counteracts the
model-weight-norm runaway that causes the noise-driven accuracy collapse
(see the parent spec for the full root-cause chain). An 8-point calibration
sweep at sigma=0.6 produced a clean, unimodal curve:

| coefficient | final accuracy | weight norm @ final round |
|---:|---:|---:|
| 0 | 0.100 | 346.6 |
| 1e-4 | 0.100 | 343.5 |
| 1e-3 | 0.101 | 318.5 |
| 3e-3 | 0.182 | 269.5 |
| 1e-2 | 0.291 | 151.7 |
| **3e-2** | **0.345** | **34.6** |
| 5e-2 | 0.293 | 14.2 |
| 1e-1 | 0.195 | 6.0 |

Below ~1e-3: too weak, no measurable effect, collapse unchanged. Above
5e-2: the decay itself becomes strong enough to prevent learning (weight
norm crushed toward zero, peak accuracy barely above random). The safe,
functional range is roughly 1e-2 to 5e-2, peaking at 3e-2 -- very stable
there (peak round 93/100, only a 0.005 peak-to-final drop).

This sweep only calibrated sigma=0.6. Rather than repeat a full sweep at
every sigma/environment combination, this phase builds a controller that
finds the right strength itself, using the sweep's bounds as the safety
envelope rather than guessing them (the mistake made on adaptive EBM's
first pass, corrected in the 2026-08-08 retune).

## Goals

- A weight-decay coefficient that adapts to observed weight-norm growth
  during training, without needing a fresh calibration sweep per
  sigma/environment/approach combination.
- Bounds grounded entirely in the Phase A sweep above -- never extrapolate
  past measured, confirmed-functional territory.
- Same separately-named, additive discipline as Phase A: does not touch
  EBM, SS, or the channel-noise definition. Activates only when channel
  noise is present (mirrors EBM's own `EBM is valid only when channel
  noise is active` rule) -- no reason to run a noise-countermeasure under
  clean or hidden-only conditions where the runaway-norm mechanism isn't
  in play.

## Non-goals

- Not combining with EBM or SS in this phase -- validated in isolation
  first, same as Phase A.
- Not retuning the channel-noise definition, SS, or static/adaptive EBM.
- Not claiming the controller works at every sigma without validation --
  Phase C's own validation set (below) checks it, the same discipline used
  for adaptive EBM.

## Mechanism

A `tf.Variable`-backed mutable weight-decay coefficient passed into
`tf.keras.optimizers.SGD(weight_decay=...)` at construction -- the same
pattern the codebase already uses for learning rate (`_MutableLR` in
`basil_core/basil.py`), just applied to a different optimizer parameter.
Verified directly this session: reassigning the Variable's value *after*
the training step has already been traced under `@tf.function` correctly
changes behavior on the next call, with no retrace (a 20-step smoke test:
10 steps at `weight_decay=0`, reassign the same Variable to `0.5`, 10 more
steps -- weight norm dropped further only after the reassignment, exactly
as expected of live per-round mutation).

**Update cadence: once per round, not per step.** Adaptive EBM updates its
coefficient every local training step because it needs the current
minibatch's gradient norm, which only exists mid-step. Weight-norm growth
is different: `model_norm` is computed once per node per round in this
engine (`_state_diagnostics`, called on that node's params right after its
local training for the round completes -- not before). The controller for
round `t` therefore uses the norm measured at the *end* of round `t-1`
(each node's most recently observed value, persisted on `LogicalNode`) to
set the coefficient used for round `t`'s training -- a one-round lag,
matching the lag adaptive EBM's own `stress_ema` already has (this round's
EBM coefficient is likewise set from the *previous* round's observed
snapshot distance). This adds no new per-step compute (no extra
`tf.linalg.global_norm` calls beyond what's already computed).

**Implementation is plain Python, not TF ops.** Because the control law
runs once per round in the round loop (not inside a `@tf.function`-traced
per-step function), it operates on the plain Python `float` that
`_state_diagnostics` already returns -- no `tf.constant`/`tf.Tensor`
wrapping needed, unlike `_bounded_adaptive_coefficient`'s TF-tensor
implementation (which must run in-graph because it depends on a
per-step gradient norm). Simpler to write and unit-test than the EBM
controller for exactly this reason.

## Control law

```
growth_ratio_t   = model_norm_t / model_norm_round0                  (per node)
smoothed_ratio_t = beta * smoothed_ratio_{t-1} + (1 - beta) * growth_ratio_t
desired_t        = clip(base + gain * (smoothed_ratio_t - target), coefficient_min, coefficient_max)
applied_t        = rate_limited(desired_t, applied_{t-1}, max_change_factor)
```

`rate_limited` mirrors `_bounded_adaptive_coefficient`'s existing pattern
exactly: `clip(desired, previous / max_change_factor, previous * max_change_factor)`.

**Correction found while mapping this onto the existing engine code (not
caught during the earlier design discussion):** the multiplicative rate
limiter breaks if `applied` is ever allowed to reach exactly `0.0` --
`0.0 / max_change_factor` and `0.0 * max_change_factor` are both `0.0`, so
`previous=0.0` would clamp `applied` to `0.0` forever, regardless of later
growth. Adaptive EBM avoids this with a small positive
`adaptiveEbmCoefficientMin` (`1e-6`), never exactly zero; Phase C adopts the
same pattern for the same reason, not a new idea -- `coefficient_min` is
now a real (tiny) floor rather than implied to be `0.0`. This does not
weaken the "do nothing absent evidence of growth" intent: Phase A's own
sweep already showed `1e-4` has no measurable effect (0.100 final, 343.5
norm, indistinguishable from the true zero baseline's 346.6) --
`coefficient_min=1e-6`, ten times smaller again, is negligible by the same
evidence, not a new assumption.

Bounds grounded in the sweep above, not guessed:

| constant | value | grounding |
|---|---:|---|
| `target` | 2.0 | growth ratio at sigma=0.6's own healthy peak round (108.8/56.4 approx 1.93), right before collapse begins |
| `coefficient_max` | 0.05 | highest coefficient confirmed still functional (0.293 final accuracy); 0.1 is excluded entirely -- demonstrated catastrophic (0.195 final, norm crushed to 6.0) |
| `coefficient_min` | 1e-6 | keeps the multiplicative rate limiter from getting stuck at a permanent 0.0 (see correction above); reused from adaptive EBM's own floor, and independently justified by Phase A's 1e-4 result showing no measurable effect at this order of magnitude |
| `base` | 0.0 | the desired-coefficient formula's baseline before the rate-limited floor is applied -- always-safe, always-tested target; low-growth conditions decay toward `coefficient_min` (negligible), not toward an untested value |
| `beta` | 0.9 | reused from adaptive EBM's proven-stable smoothing constant |
| `max_change_factor` | 2.0 | reused from adaptive EBM's proven-stable rate limit |
| `gain` | 0.025 | chosen so the controller reaches `coefficient_max` once `smoothed_ratio` hits 4.0 (`0.025*(4.0-2.0)=0.05`) -- well before sigma=0.6's fully-collapsed final ratio of 6.15, so the controller is already at full strength during the pre-collapse acceleration phase (observed starting around round 34-40) rather than only maxing out after collapse has already happened. At the fully-collapsed ratio of 6.15 it stays clipped at 0.05, never exceeding the confirmed-functional ceiling |

## Activation scope

`make_config` rejects `adaptiveWeightDecayMode="adaptive"` when
`useChannelNoise` is false, raising `ValueError` -- mirroring the existing
`use_ebm and not has_noise` check exactly (reject at construction, not a
silent no-op at runtime). A config that claims "adaptive" while doing
nothing would be a misleading, confusing provenance record; rejecting it
outright is the established codebase convention EBM already uses. When
active, every node's controller runs independently (each node has its own
`model_norm_round0` reference and its own smoothed ratio), consistent with
how adaptive EBM's per-node `stress_ema` already works.

## Config fields

New, orthogonal to `weightDecayCoefficient` (Phase A's static field, which
stays available as a control). Naming mirrors the existing `adaptiveEbm*`
convention:

```
adaptiveWeightDecayMode: "none" | "adaptive"      (default "none")
adaptiveWeightDecayTargetRatio: float              (default 2.0)
adaptiveWeightDecayGain: float                     (default 0.025)
adaptiveWeightDecayCoefficientMin: float           (default 1e-6)
adaptiveWeightDecayCoefficientMax: float           (default 0.05)
adaptiveWeightDecayBeta: float                     (default 0.9)
adaptiveWeightDecayMaxChangeFactor: float          (default 2.0)
```

`base` (0.0) is deliberately *not* a config field. Every other constant
above represents a genuine calibration choice; `base` specifically encodes
"do nothing absent evidence of growth," which is the mechanism's whole
safety rationale from the parent spec's own Decision 1. Exposing it as a
tunable field would let a future config accidentally weaken that guarantee.
It is hardcoded to `0.0` directly in the engine.

## Telemetry

Two new per-round, per-node fields (both cheap -- no new per-step compute):
`adaptive_weight_decay_coefficient` (the applied value) and
`adaptive_weight_decay_growth_ratio` (the smoothed ratio that drove it).
Mirrors the existing `ebm_coefficient`/`stress_ema` telemetry shape exactly.

## Testing

- **Unit test** for the control-law function in isolation, same intent as
  `test_adaptive_coefficient_is_bounded_and_rate_limited` (fixed inputs,
  assert the output lands within the expected bounded range) but on plain
  Python floats, not TF tensors -- per the plain-Python implementation
  decision above.
- **Regression test locking in the `tf.Variable`-reassignment mechanism**
  verified manually this session -- a small `@tf.function`-wrapped
  optimizer step, reassign the weight-decay Variable mid-stream, assert
  behavior changes on the next call. Protects against a future TF upgrade
  silently changing this (undocumented) behavior.
- **Contract test**: `adaptiveWeightDecayMode` defaults to `"none"` across
  every existing config-generation path, and is rejected by `make_config`
  when `useChannelNoise` is false (mirrors the existing
  `use_ebm and not has_noise` validation pattern).

## Validation plan

Small, isolated diagnostic set (no EBM, no SS -- same discipline as Phase
A), `merged`/nonIID/persistent/seed 2025/100 rounds:

1. Noise sigma=0.6, adaptive weight decay -- the primary question: does it
   reach and hold in the working 1e-2 to 5e-2 range without needing to be
   told the right value, matching or approaching the fixed-3e-2 result
   (0.345 final)?
2. Noise sigma=0.2 and sigma=0.4, adaptive weight decay -- regression
   check: does the controller correctly back off (stay near its `base=0.0`
   floor) when growth is mild, rather than over-regularizing conditions
   that don't need it? (The clean/no-noise case doesn't need its own
   diagnostic run: `make_config` rejects `adaptiveWeightDecayMode="adaptive"`
   there outright, per Activation scope above, and that rejection is
   covered by the contract test in Testing, not a training run.)

## Open questions this validation must answer before any further sweep

1. Does the controller reach the sigma=0.6 sweet spot on its own, or does
   `gain=0.025` need retuning (too slow/fast to reach the working range
   within the round budget)?
2. Does it correctly stay inert at low sigma, or does the `target=2.0`
   threshold trigger unnecessarily there?
3. If both hold, is the controller ready for the same isolate-then-combine
   sequence Phase A used (test alone first, then consider EBM/SS
   combination as separate future work)?
