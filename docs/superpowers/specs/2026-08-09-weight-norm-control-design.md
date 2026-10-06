# Weight-Norm Control: A Separately-Named Alternative to Adaptive EBM at High Sigma

**Status:** design approved, not yet implemented
**Date:** August 9, 2026
**Scope:** Campaign 4, `merged` approach, nonIID, noise-only environment (Phase A)

## Background

Campaign 4's adaptive EBM controller was retuned on 2026-08-08 after diagnostics
showed the original bounds drove the applied coefficient to 20-60x the static
schedule's calibrated magnitude, producing pre-clip gradient-norm spikes as
high as 1484 at sigma=0.6. The retune fixed that instability precisely
(pre-clip max dropped from 781.5 to 14.2 in the matched persistent-arm
sigma=0.6 comparison) but did not fix accuracy: no-mitigation, static EBM,
old adaptive EBM, and retuned adaptive EBM all converge to final accuracy
0.1000 (random) at sigma=0.6. See the paired comparison in conversation
history; also archived reasoning in
`experiments/results4/campaign4/superseded/README.md`.

Root-cause telemetry (persistent-arm, noise-only) isolated the actual
mechanism:

| | sigma=0.6, no-mit | sigma=0.4, no-mit (reference) |
|---|---:|---:|
| peak round | 34 (acc 0.301) | 92 (acc 0.332) |
| model weight norm: round0 -> peak -> final | 56 -> 109 -> **347** | 55 -> 109 -> **114** |
| momentum norm: round0 -> final | 0.14 -> 0.03 | 0.11 -> 0.03 |

At sigma=0.4 the model's weight norm plateaus after reaching a stable
operating point (no collapse). At sigma=0.6 it **triples** after the peak
round, and that is exactly when accuracy free-falls from 0.30 to 0.10.
Momentum norm shrinks monotonically at every sigma, ruling out momentum
accumulation as the driver.

Channel noise is defined as `relative_l2`: `noise_norm = sigma * ||model||`.
Because nothing in the current training loop bounds weight norm (no weight
decay anywhere in `basil_core/adaptive_experiment_engine.py`, no BatchNorm in
`CIFARModel`), the following feedback loop is unconstrained: bigger weights
-> proportionally bigger absolute noise -> more corrupted updates -> weights
pushed further off course -> still bigger norm. EBM regularizes
`||grad F(w)||^2` (gradient norm), not weight norm, so it does not intervene
on this loop -- which is consistent with EBM (in any tuning) failing to move
sigma=0.6 final accuracy at all.

The source paper (arXiv:1911.00251, EBM/WCM) uses fixed-variance absolute
noise (`E{DwDw^T} = sigma_e^2 I`), not noise scaled by model norm, plus a
star/full-aggregation topology and IID data -- conditions under which this
feedback loop cannot arise. Campaign 4 deliberately tests a harder,
decentralized non-IID setting with relative noise, so EBM's failure to
transfer is not surprising in hindsight. The channel-noise definition itself
stays untouched by this design (explicit decision, see Decisions below) --
the fix targets the model side of the loop, not the noise side.

## Goals

- Determine whether directly bounding weight-norm growth breaks the
  sigma=0.5/0.6 collapse, without touching the channel-noise definition,
  SS, or EBM.
- Keep it a separately-named, additive method per the project's Decision
  Rule 8 ("report the failure and evaluate a separately named alternative
  ... rather than silently changing EBM") -- static/adaptive EBM remain
  required controls, untouched, still comparable in every existing and
  future confirmation run.
- Validate cheaply (a fixed-coefficient version) before building anything
  adaptive.

## Non-goals

- Not replacing EBM in any existing or planned confirmation run.
- Not changing `channelNoiseSemantics` or how sigma is defined.
- Not (yet) combining this mechanism with EBM or SS -- Phase A isolates it
  alone to get a clean read; combinations are future work if Phase A and
  Phase C validate.
- Not building the adaptive controller (Phase C) in this pass -- its bounds
  will be grounded in Phase A's telemetry, not guessed upfront.

## Decisions (from brainstorming)

1. **Additive control, not a noise-model change.** Weight-norm control acts
   on the model/training side; `relative_l2` noise semantics are frozen as
   documented in ADAPTIVE_STUDY_PLAN.md.
2. **Separate, orthogonal config field**, not folded into the `mitigation`
   enum (`none`/`ss`/`ebm`/`ss_ebm`). This keeps every existing EBM/SS
   contract test and config hash untouched.
3. **Two-phase build**: Phase A (fixed coefficient) validates the mechanism
   cheaply; Phase C (adaptive, EBM-controller-pattern) is designed only
   after Phase A produces real telemetry to ground its bounds.

## Phase A design: fixed weight decay

### Config

New field `weightDecayCoefficient: float`, default `0.0`. Default-off means
every one of the 526 existing checked-in configs is byte-for-byte unchanged
-- this is purely additive.

Naming (mirrors the existing `optimizer=persistent` suffix precedent in
`_experiment_name`): when `weightDecayCoefficient > 0`,
`_experiment_name` appends `| weight_decay=<value>` and `_condition_id`
appends `_wd`, so the condition is unambiguously separate from
`no_mitigation` in every plot, table, and directory path.

### Engine

`basil_core/adaptive_experiment_engine.py`: pass `weight_decay=<coefficient>` directly
into the existing `tf.keras.optimizers.SGD(...)` constructor. TF's SGD
supports decoupled weight decay natively -- no new gradient computation, no
new backward pass, no measurable compute cost. This directly opposes
weight-norm growth at every local step.

### Validation set (3 configs, isolate the variable)

All: `approach=merged`, `nonIID`, `optimizerStateMode=persistent`,
`mitigation=none` (no EBM, no SS), `phase=diagnostic`, seed 2025, 100 rounds,
`weightDecayCoefficient=1e-4` (standard CNN/SGD literature default -- not
guessed aggressively, avoiding the mistake made tuning adaptive EBM's first
bounds).

| # | condition | question it answers |
|---|---|---|
| 1 | clean + weight decay | Does it hurt when there's nothing to defend against? |
| 2 | noise sigma=0.4 + weight decay | Does it preserve the already-fine case? |
| 3 | noise sigma=0.6 + weight decay | Does it fix the collapse? |

Evaluation: compare final/peak accuracy against the existing persistent-arm
baselines (clean 0.468, noise-0.4-none 0.374-ish reference range, noise-0.6-none
0.100) and pull `model_norm` telemetry (already an existing field, no schema
change needed) to directly confirm whether norm growth is actually
suppressed, not just infer it from accuracy alone.

## Testing

- **Unit test** (tiny synthetic model): weight decay measurably shrinks
  weight norm over N training steps vs. a `weight_decay=0` control, holding
  everything else fixed.
- **Contract test**: `weightDecayCoefficient` defaults to `0.0` across every
  existing config-generation path (`build_main_confirmation`,
  `build_static_controls`, `build_diagnostic`, `build_performance_benchmark`)
  -- regression guard proving this change touches nothing already run, and
  that `config_hash`/`runId` for every existing generated config is
  unchanged before/after this change lands.

## Phase C (future, not designed yet)

Same bounded/EMA-smoothed/rate-limited controller *pattern* already proven
safe for adaptive EBM (see `_adaptive_requested_coefficient` /
`_bounded_adaptive_coefficient` in `basil_core/adaptive_experiment_engine.py`), but
observing weight-norm growth rate instead of gradient-ratio stress, and
outputting a dynamically-scaled weight-decay coefficient instead of an EBM
coefficient. Its target ratio, min/max bounds, and smoothing constants will
be chosen from Phase A's measured norm-growth telemetry, not guessed --
avoiding a repeat of the adaptive-EBM first-pass over-aggressive-bounds
mistake. Full config/engine/telemetry/testing design deferred to its own
follow-up spec once Phase A data exists.

## Open questions Phase A must answer before Phase C is designed

1. Does `weightDecayCoefficient=1e-4` measurably suppress `model_norm`
   growth at sigma=0.6 (telemetry check), and does that translate to final
   accuracy above 0.10?
2. Does it cost any accuracy in the clean and sigma=0.4 conditions (a
   regression check on the cases that already work)?
3. Is 1e-4 in the right order of magnitude, or does the norm trajectory
   suggest a stronger/weaker coefficient is needed before Phase C's adaptive
   range is chosen?
