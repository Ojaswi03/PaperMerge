# Superseded Campaign 4 records

Runs moved here are preserved as part of the research record (Campaign 4
decision rule 10) but are intentionally outside plot discovery, which reads
`experiments/results4/campaign4/gui` only.

| Run | Condition | Reason moved (2026-08-04) |
|---|---|---|
| `v4-f5977f96a1d758dc10012a82` | nonIID Merged clean_pairwise_no_mitigation seed 2025 | Stopped (incomplete) run from the retired `mb512_float32_no_xla` profile with stale source hashes. Superseded by completed `v4-5e51006cd80557728fbc52d7` under the validated `mb512_mixed_bfloat16_no_xla_cuda_malloc_async` profile. |

## 2026-08-05: adaptive EBM controller retune

All 66 diagnostic-phase runs completed under the pre-retune adaptive EBM
bounds (`adaptiveEbmTargetRatioBase=0.05`, `adaptiveEbmRatioMax=0.35`,
`adaptiveEbmCoefficientMax=0.01`) were moved to
`superseded/nonIID/cifar10/...` (mirroring their original relative paths).

**Why:** telemetry showed the applied EBM coefficient running 20-60x above
the static schedule's calibrated values (active ratio ~0.20-0.30 vs static's
incidental ~0.02-0.04), producing pre-clip gradient-norm spikes up to 1484 at
sigma=0.6 and no accuracy benefit over no mitigation at any tested sigma.
`gui/campaign4.py` was edited to tighten the adaptive bounds
(`adaptiveEbmTargetRatioBase=0.015`, `adaptiveEbmStressGain=0.05`,
`adaptiveEbmRatioMin=0.005`, `adaptiveEbmRatioMax=0.08`,
`adaptiveEbmCoefficientMax=0.0025`).

Since `gui/campaign4.py` is one of the 8 `PROVENANCE_FILES`, this edit
changes `current_source_hashes()` for every run regardless of environment,
so **all 66 diagnostics became stale for the freeze gate**, not just the
EBM-related ones. The full 66-run diagnostic sweep must be re-run on the
current source hashes before `Freeze Campaign 4 method` unlocks.
`plots4/campaign4/images` and `.plot_manifest.json` were cleared at the same
time since they only reflected the now-superseded data.

A 12-run validation set (clean, hidden+SS, noise x5 sigma + adaptive EBM,
hidden+noise x5 sigma + SS+adaptive EBM; merged/nonIID/persistent/seed 2025)
was queued separately as a fast sanity check on the retuned controller. It is
**not** a replacement for the 66-run diagnostic gate.

Notes from the same audit, for the record:

- `v4-33ad90c9…` (visit_reset) and `v4-16abcd7d…` (persistent) under
  `noise_sigma_0_4_ebm_static_ebm/seed_2025` are NOT duplicates; they are
  distinct Stage-B optimizer-state arms that share a condition folder.
- `v4-16abcd7d…`, `v4-67045f7b…`, and `v4-d1c0ff2f…` failed with
  `[Errno 32] Broken pipe` when their GUI parent died on 2026-08-03 (the
  orphaned-worker incident). They remain queued; re-running them overwrites
  the failed records in place under the same run IDs.
