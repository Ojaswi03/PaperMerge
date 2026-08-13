# PaperMerge — Full Context Handoff

This document exists to let anyone picking up this project fresh, with no
memory of prior work sessions, get complete context quickly. It is written
as of **2026-08-10**, at the end of a working session that fixed several
real bugs in the Campaign 4 plotting pipeline and built the next batch of
experiments. Read this whole file before touching code.

## 1. What This Project Is

PaperMerge is the codebase behind Ojaswi Sinha's **MS CS (AI-ML) master's
thesis** (Ohio University, graduating December 2026), on **Byzantine-resilient
federated learning under noisy communication channels**. It is a
TensorFlow/Tkinter research framework, not a product — every design choice
downstream should serve "is this scientifically defensible and reproducible
for a thesis," not "is this good software architecture" for its own sake.

The framework compares four training approaches across CIFAR-10 (and
N-MNIST), IID and non-IID data splits:

| Approach | Topology | Purpose |
|---|---|---|
| BASIL | decentralized ring | Snapshot Selection (SS) for Byzantine resilience |
| Noisy Channel | FedAvg/server-client | EBM/WCM for channel-noise mitigation |
| Merged | decentralized ring | combines BASIL-style Byzantine defense with EBM noise mitigation — a project addition |
| CART | decentralized ring | class-aware ring training for non-IID data, with optional SS and EBM — a project addition |

Merged and CART are Ojaswi's original contributions built on top of two
source papers (BASIL and the noisy-channel/EBM paper). Read `README.md` in
full — it is long, detailed, and authoritative on setup, GUI usage, and
directory layout. This document does not repeat what's already there; it
covers **what changed and what was learned this session**, plus enough
architecture summary to navigate.

**Non-negotiable research rules** (violating these invalidates results):
- Clean environment = 0 Byzantine, no channel noise, no mitigation. Never use
  SS or EBM "just because it might help" in a clean run.
- SS / `useBasil=true` is Byzantine-attack mitigation **only**.
- EBM/WCM is channel-noise mitigation **only**.
- `weightDecayCoefficient`/`adaptiveWeightDecayMode` (see §3 below) is a
  newer, separate noise-mitigation mechanism, orthogonal to `mitigation`.
- Merged/CART no-SS clean/no-attack/no-noise configs use plain decentralized
  consensus averaging (`aggregationMode="consensus"`), not SS.

## 2. Two Parallel Campaigns — Don't Confuse Them

- **Campaign 3 (R2)**: the mature, previously-confirmed conference-paper
  study. Results in `experiments/results3/r2/`, plots in `plots3/r2/`. Not
  touched this session.
- **Campaign 4**: a separate, versioned, actively-evolving investigation
  (high-noise accumulation, then extended into weight-norm control and the
  accuracy-ordering hypothesis below). Results in
  `experiments/results4/campaign4/gui/`, plots in `plots4/campaign4/`,
  configs built by `gui/campaign4.py`. **All of this session's work is
  Campaign 4.**

Never let a plotting or config change for one campaign silently touch the
other; they have separate result roots, separate plot generators
(`plots/plotCampaign3.py` vs `plots/plotCampaign4.py`), and separate
`campaign_state.json` freeze gates.

## 3. The Adaptive Weight-Decay Controller (Weight-Norm Control)

Design docs: `docs/superpowers/specs/2026-08-09-weight-norm-control-design.md`
and `2026-08-09-adaptive-weight-decay-phase-c-design.md`. Plans:
`docs/superpowers/plans/2026-08-09-weight-norm-control-phase-a.md` and
`2026-08-09-adaptive-weight-decay-phase-c.md`.

**What it is:** a new noise-mitigation mechanism, added as **Phase A** (a
config field, `weightDecayCoefficient`, wired into the SGD optimizer,
orthogonal to `mitigation`) then **Phase C** (an adaptive controller,
`adaptiveWeightDecayMode="adaptive"`, that adjusts the coefficient online
each round — bounded, EMA-smoothed, rate-limited — reacting to observed
weight norm, following the same design pattern already proven for adaptive
EBM in `basil_core/campaign4_engine.py`).

**Critical implementation detail that caused two real bugs this session:**
for adaptive runs, the static `weightDecayCoefficient` field in the config
**always starts at 0.0** — the controller adjusts the live value online, but
the config's static field is just the starting point. Any code (plotting,
analysis, filtering) that checks `weightDecayCoefficient > 0` to decide "is
weight decay active" will silently return false for every adaptive run. You
must check `adaptiveWeightDecayMode != "none"` too. This is now handled
correctly in `plots/plotCampaign4.py`'s `RunRecord.effective_mitigation`
property (see §5) — if you write new code that needs to know whether weight
decay is active, use that pattern, don't reinvent it.

Phase A/C implementation commits: `c0b9500` through `afd6f69` (config fields,
control-law function, mutable `tf.Variable` refactor, round-loop wiring,
config library regen, target_ratio/gain retune). All on `main`, no separate
branch (matches Phase A's precedent).

## 4. The Accuracy-Ordering Diagnostic Matrix

**The hypothesis being tested:** for non-IID CIFAR-10, accuracy should order
as: `clean > single stressor + matched mitigation > joint stressors + partial
mitigation > joint stressors + full mitigation > joint stressors + no
mitigation (worst case)`, and CART should generally outperform Merged at each
tier.

**Config generation:** `gui/campaign4.py`'s `make_config()` (keyword args:
`split`, `approach`, `environment`, `mitigation`, `seed`, `sigma`, `ebm_mode`,
`phase`, `weight_decay_coefficient`, `adaptive_weight_decay_mode`, etc. — see
the function docstring and full signature for every field, it validates
combinations, e.g. SS requires the Hidden attack active, EBM requires noise
active). `runId` is a content hash (`v4-{sha256[:24]}`) of the canonical
config payload (`config_hash`/`_canonical_payload`, which excludes
`experimentName`/`runId`/`status`/timestamps) — so identical scientific
parameters always produce the same `runId`, and changing anything
scientifically meaningful (including `seed` or `adaptiveWeightDecayMode`)
produces a different one. `conditionId` (from `_condition_id`) is a coarser
grouping label used for display — it does **not** include
`adaptiveWeightDecayMode`, so two scientifically-different configs (e.g.
SS-only vs SS+adaptive-weight-decay) can share the same `conditionId` while
having different `runId`s. Don't confuse the two.

**The matrix as originally built** (commit `3df82b8`, "queue accuracy-ordering
diagnostic matrix"), per approach (merged, cart), all `phase="diagnostic"`,
`optimizer_state_mode="persistent"`, 100 rounds:
- 1 clean reference (0 Byzantine, no noise, no mitigation)
- 1 single stressor: hidden Byzantine attack + SS
- 3 single stressor: channel noise (sigma 0.2/0.4/0.6) + adaptive weight-decay
- 3 joint, "2 mitigations": hidden attack + noise, SS (attack) + adaptive
  weight-decay (noise) — **note this is SS+WD, not SS alone**
- 3 joint, "3 mitigations": hidden attack + noise, SS + EBM + adaptive
  weight-decay
- 3 joint, no mitigation: worst-case anchor

That's 14 conditions × 2 approaches = 28 configs, all seed=2025. All 28
completed successfully this session (no failures, no OOM) with `status:
"completed"` in their `run.json`.

**A design gap found and partially fixed this session:** the "SS" tier above
is not actually SS-only — it has adaptive weight decay engaged too, so it
doesn't isolate "attack mitigated, noise completely unmitigated" as a
baseline. A **new, genuinely SS-only tier** (`mitigation="ss"`,
`adaptiveWeightDecayMode="none"`, `hidden_noise` environment, sigma
0.2/0.4/0.6) was added this session — 12 configs (2 approaches × 2 seeds ×
3 sigmas) — to complete the real ladder: **SS-only → SS+WD → SS+EBM+WD →
worst-case**. This tier has run to completion (see §7, current state).

**Seed ablation:** the project's own seed convention already existed in
`gui/campaign4.py`: `DIAGNOSTIC_SEED = 2025`, `CONFIRMATION_SEEDS = (2026,
2027, 2028)`. This session built a full second copy of the 28-config matrix
at `seed=2026` (identical parameters otherwise) to get real seed-to-seed
variance for `seed_profiles.png` and error bars on every other figure — with
only one seed, every error bar in every figure is exactly zero-width
(`_mean_error` only computes real spread when `len(values) > 1`).

**Important, still-live finding:** now at n=6-per-cell (both seeds, 3 sigmas
averaged), the 3-mitigation tier (SS+EBM+WD) still does **not** beat the
2-mitigation tier (SS+WD) — 0.335 vs 0.339 (CART), 0.326 vs 0.345 (merged).
This is real data, not a bug — it directly tests (and so far does not
confirm) whether EBM adds value once adaptive weight decay is already
mitigating noise. The seed-2026 batch has now run and the finding held up
rather than overturning — see §7 for the current telemetry investigation
into whether this is a genuine EBM/weight-decay interaction effect. Do not
assume this is settled; only two seeds' worth of data exists so far, and
`CONFIRMATION_SEEDS` still has 2027/2028 unused if more evidence is wanted.

## 5. Two Real Bugs Found And Fixed In `plots/plotCampaign4.py` This Session

Both were caught by the user's direct scrutiny of "some plots don't look
right" — not by any automated check. Read this section carefully before
trusting any figure this pipeline has ever produced, or before writing new
analysis code against `RunRecord`.

### Bug 1 — Phase gate excluded every result from paper figures

`generate_campaign4_plots()`'s "paper" figure branch (evidence hierarchy,
defense composition, learning curves, noise robustness, average-vs-worst,
seed profiles, class retention, static-vs-adaptive, cart-lift) originally
filtered `record.phase == "confirmation"` only. **Every single Campaign 4
result in this project — all 50 records at the time — is `phase="diagnostic"`,
none are `"confirmation"`.** The loader (`_validated_record`) already treats
`"diagnostic"`, `"static_control"`, and `"confirmation"` identically for
data-quality purposes (`SCIENTIFIC_PHASES` — all three require the full
100-round history before being accepted at all), so the narrower
`"confirmation"`-only check in the paper-figure branch was an inconsistency,
not a deliberate rigor tier. **Fixed** by widening both the `confirmation_all`
filter and the `all_evidence` filter to `phase in SCIENTIFIC_PHASES`. Before
this fix, `merged/paper/` and `cart/paper/` directories were completely
empty — only `diagnostics/` figures (telemetry, gradient, CART, runtime
panels) existed anywhere.

### Bug 2 — weight-decay tier detection checked the wrong field, twice

First pass: `effective_mitigation` (a new property added to distinguish
weight-decay-augmented runs from plain ones in plotted bars/lines) checked
`weightDecayCoefficient > 0.0`. Because every weight-decay run in this
matrix uses the **adaptive** controller (§3), that field is always `0.0` in
the static config — so the check never triggered, and every "SS+EBM" bar
silently included weight-decay runs mislabeled as plain SS+EBM. Fixed by
also checking `adaptiveWeightDecayMode != "none"`.

Second pass (caught by the user directly, not by re-review): the fix above
only applied the weight-decay split to `mitigation="ss_ebm"`, not to
`mitigation="ss"`. But **every "SS" record in this matrix also has adaptive
weight decay engaged** (§4) — so the "Joint+SS" bar/line was *also*
mislabeled the whole time, showing SS+WD data under a plain "SS" label. This
made a real finding ("SS+EBM+WD doesn't beat SS") look like it was comparing
"3 mitigations vs 1 mitigation" when it was actually comparing "3 mitigations
vs 2 mitigations" — a very different and more defensible result. Fixed by
generalizing `effective_mitigation` to split **both** `"ss"` and `"ss_ebm"`
into `_wd` variants when weight decay is active:

```python
@property
def effective_mitigation(self):
    weight_decay_active = (
        float(self.config.get("weightDecayCoefficient", 0.0)) > 0.0
        or str(self.config.get("adaptiveWeightDecayMode", "none")) != "none"
    )
    if weight_decay_active and self.mitigation in ("ss", "ss_ebm"):
        return f"{self.mitigation}_wd"
    return self.mitigation
```

`MITIGATIONS`, `MITIGATION_LABELS`, `COLORS`, `MARKERS` now include `ss_wd`
and `ss_ebm_wd` as first-class tiers. Every accuracy-hierarchy figure
(`evidence_hierarchy`, `defense_composition`/`joint_challenge_profile` via
`_line_by_sigma`, `avg_worst`, `seed_profiles`, `learning_curves`,
`noise_robustness`, `class_retention`, `cart_lift`/`_paired_delta`) now keys
off `effective_mitigation`, not the raw `mitigation` field. `_static_adaptive`
(EBM static-vs-adaptive comparison) and the four diagnostics panels
(telemetry/gradient/CART/runtime) were deliberately left on raw `mitigation`
— they're not part of the accuracy-hierarchy narrative.

**Lesson for whoever picks this up:** if you add a new mitigation
combination, or a new orthogonal config axis like weight decay, you must
update `effective_mitigation` (or whatever plays its role) and re-audit
every figure that assumes a mitigation label is complete. A label that's
merely "not wrong syntactically" (a valid string in `MITIGATIONS`) can still
be semantically wrong (grouping data that shouldn't be grouped) in a way
that produces a plausible-looking, publication-quality chart with an
incorrect conclusion. This already happened twice in one session.

**How to regenerate plots correctly:** `generate_campaign4_plots()` in
`plots/plotCampaign4.py` resolves `RESULT_ROOT`/`PLOT_ROOT` relative to the
current working directory (`experiments/results4/campaign4/gui` and
`plots4/campaign4`). **Always run it from the repository root**, not from
inside `plots/` — running it from `plots/` silently writes a duplicate tree
to `plots/plots4/campaign4/` instead (this happened once this session; the
stray directory was deleted and `.gitignore` now has a defensive
`/plots/plots4/` entry in case it happens again). Correct invocation:

```bash
cd /home/ojaswi/PaperMerge
environment/basil-noise-env/bin/python -c "
import sys; sys.path.insert(0, 'plots')
from plotCampaign4 import generate_campaign4_plots
result = generate_campaign4_plots(mode='both', only_changed=False)
print(result['records'], len(result['generated']), result['errors'])
"
```

In normal GUI operation this is handled automatically and correctly by
`gui/experimentGui.py`'s `_scheduleCampaign4PlotRefresh`/queue-boundary hooks
— the cwd issue only bites when invoking the plotting module standalone from
a shell, as was done for the manual verification this session.

## 6. Execution Reliability Work (Earlier This Session)

This is now stable and shouldn't need revisiting unless something breaks
again. Full detail in `README.md`'s "Execution Reliability" subsection and in
the auto-memory file `campaign4_single_lane_only.md`. Summary:

- Campaign 4 runs isolated GPU subprocess workers (`gui/campaign_workers.py`'s
  `CampaignWorkerPool`), tracked via PID sidecar files for orphan detection.
- An earlier 2-lane run caused an unkillable hang with no diagnosable trace:
  GPU memory silently hit 11.8/12.28 GB against a configured 5.3 GB sum, and
  no crash log existed anywhere. Root-caused (via the
  `systematic-debugging` skill) to two things: (1) `TF_FORCE_GPU_ALLOW_GROWTH`
  being set unconditionally, fighting the per-lane hard memory cap — fixed in
  `scripts/run_campaign4_worker.py`'s `_configure_tensorflow` (only force
  growth when no cap is requested); (2) worker stdout/stderr wasn't persisted
  to disk — fixed in `gui/campaign_workers.py` (`.log` files next to PID
  sidecars, preserved on non-zero exit for post-mortem, cleaned up on success).
- **Despite both fixes being confirmed correct, 2-lane concurrency is a
  standing, non-negotiable decision to keep disabled**, based on the user's
  direct experience during the incident, re-confirmed explicitly when offered
  the option again later in the session. `experiments/results4/campaign4/performance_profile.json`
  (machine-local, gitignored) has `recommendedLanes: 1` in both the top-level
  and nested `concurrencyProfile` locations. **Do not suggest re-enabling
  2-lane unless the user raises it first.**
- A separate, smaller bug: the Manage Queue dialog's "Total remaining" banner
  and its per-row table used two different estimation code paths (a static
  `estimate()` vs. a live makespan-simulating `estimate_queue()`), so they
  drifted out of sync. Fixed in `gui/experimentGui.py` (commit `8160dc9`) by
  having the per-row table subtract live elapsed time and refresh on a 10s
  timer while the queue runs.
- GPU/CPU/RAM headroom investigation (not a bug, just an analysis): this
  workload is GPU-compute-bound, not CPU- or memory-bound (idle 30/32 CPU
  cores, ~3GB/12GB VRAM in use at 91% GPU utilization). There's no
  low-risk "use the idle hardware" lever available — the two real levers are
  a larger batch size (changes training dynamics, a research decision) or
  retrying XLA after a CUDA/TF stack upgrade (already flagged in
  `docs/Campaign4RuntimeResearch.md` as blocked on a stack upgrade, the
  single biggest documented pending speedup, 1.5-2x on EBM paths via kernel
  fusion). A GPU hardware upgrade (RTX 5080 recommended over 5090 for this
  workload's actual VRAM needs) would likely force that stack upgrade as a
  side effect.

## 7. Current State — Read This Before Doing Anything

**Queue** (`gui/queue_state.json`, git-ignored): empty — the 40-config batch
described below has finished running.

**Completed results:** 90 `run.json` records total (56 at `seed=2025`, 34 at
`seed=2026`), all `phase="diagnostic"`, all `status="completed"`, no
failures, no OOM. This is the original 50-record accuracy-ordering matrix
plus the 40-config batch that was queued:

- 28 configs: full accuracy-ordering matrix at `seed=2026` (mirrors the
  `seed=2025` batch exactly).
- 12 configs: the true SS-only tier (`mitigation="ss"`,
  `adaptiveWeightDecayMode="none"`, `hidden_noise`, sigma 0.2/0.4/0.6), at
  **both** seeds 2025 and 2026 (2 approaches × 2 seeds × 3 sigmas).

All single-lane, `optimizer_state_mode="persistent"`. Plots regenerated and
correct as of this session (with both bugs above fixed).

**Open research question, not yet resolved:** across both seeds (n=6 per
cell), `SS+WD` slightly outperforms `SS+EBM+WD` (CART 0.339 vs 0.335, merged
0.345 vs 0.326) — i.e. adding EBM on top of SS+WD does not help, and may
hurt slightly. Telemetry on one CART/hidden_noise/`ss_ebm_wd`/sigma=0.4/
seed=2025 run shows the adaptive EBM coefficient actively varying
(min/mean/max = 0.000194/0.00080/0.00179 against bounds 1e-6/0.0025) — so
EBM is genuinely engaging, not silently pinned off. This does not yet
explain the flat-to-negative result; it could be a real interaction effect
(EBM's gradient-norm regularizer and weight decay competing over the same
underlying quantity) or could resolve with more seeds. Not root-caused.

**A previously-flagged, still-unresolved anomaly** (from an earlier session,
investigated via `systematic-debugging` but not root-caused): the CART/Merged
clean-condition (nonIID, no attack, no noise, no mitigation) final accuracy
plateaus around 41-47%, not the 80-90% one would expect for CIFAR-10. This
was flagged before this session started and was not revisited this session —
it's still an open question and should not be assumed resolved just because
the accuracy-ordering matrix's numbers look internally consistent with each
other (they are consistent with each other, but the whole scale might still
be depressed for a reason not yet understood).

## 8. File Map — Where To Look

```
gui/campaign4.py            # make_config(), condition/experiment naming, config_hash/runId, seed constants
gui/campaign4_execution.py  # settings_from_profile(), select_next_config() concurrency scheduling
gui/campaign_workers.py     # CampaignWorkerPool: subprocess launch, PID sidecars, .log persistence, orphan cleanup
gui/experimentGui.py        # GUI: queue, manual runs, live plot scheduling, Manage Queue dialog
gui/runtime_estimator.py    # per-config and whole-queue ETA (estimate() vs estimate_queue())
basil_core/campaign4_engine.py  # SGD optimizer, adaptive EBM controller, adaptive weight-decay controller
basil_core/cart.py          # CART ring training
basil_core/basil.py         # BASIL ring / FedAvg loops
plots/plotCampaign4.py      # RunRecord, effective_mitigation, all Campaign 4 figures (paper + diagnostics)
scripts/run_campaign4_worker.py  # one isolated Campaign 4 run per subprocess; GPU memory-cap config
experiments/results4/campaign4/gui/   # Campaign 4 result artifacts (run.json, metrics.npz, telemetry.npz)
experiments/results4/campaign4/performance_profile.json  # machine-local, gitignored — lanes, memory caps
plots4/campaign4/           # Campaign 4 figures (paper/, diagnostics/) and run_summary.csv
docs/superpowers/specs/     # weight-norm-control and Phase C design docs
docs/superpowers/plans/     # Phase A / Phase C implementation plans
docs/Campaign4RuntimeResearch.md  # GPU benchmarking history, XLA status, runtime levers
docs/SessionHandoff.md       # this file
```

## 9. Verification Commands

```bash
source environment/basil-noise-env/bin/activate
environment/basil-noise-env/bin/python -m py_compile gui/experimentGui.py plots/plotGui.py plots/plotCampaign4.py \
    scripts/run_single_config.py scripts/run_campaign4_worker.py basil_core/basil.py basil_core/cart.py gui/campaign4.py
environment/basil-noise-env/bin/python -m unittest tests.test_campaign_workers -v
git diff --check
```

To regenerate Campaign 4 plots from the repo root (see §5 for why cwd
matters):

```bash
cd /home/ojaswi/PaperMerge
environment/basil-noise-env/bin/python -c "
import sys; sys.path.insert(0, 'plots')
from plotCampaign4 import generate_campaign4_plots
print(generate_campaign4_plots(mode='both', only_changed=False))
"
```
