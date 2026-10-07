# Fresh GPU-only A rerun

The user requested a full-GPU rerun of A after B/C/D completed. Preparation
does not start training. The persisted queue now contains **only A**; B/C/D
remain completed and are not queued again.

## Start manually

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

In an already-open GUI, select **Queue → Load Queue** and choose
`gui/queues/iid_a_100r_gpu_fresh.json`. Confirm one Pending A entry, GPU,
100 rounds, **0/100**, and **Fresh R0→R99 — GPU**, then press **Run Queue**.
Do not load the old CPU-continuation preset.

The config is
`gui/configs/IID/sequential_basil_100r/gpu/fresh/A_iid_basil_clean_no_attack_ce_100r_gpu.json`.
It omits `recoverySourceDirectory` and `deviceTransition`, so there is no CPU-state
reconstruction or loaded CPU-trained prefix. Canonical seeded initial weights
are generated on CPU, as for fresh B/C/D, and all local training runs on GPU.
Keyed image augmentation remains on CPU; GPU-only refers to training, not every
preprocessing or filesystem operation. GPU unavailability fails closed.

## Evidence preservation and routing

The previous A completed 100 rounds but had a 38-activation CPU prefix. Its
unchanged results are now archived at:

- `newResults/IID/reference_archive/A_clean_no_attack_ce_100r_gpu_mixed_device/`
- `newPlots/IID/reference_archive/A_clean_no_attack_ce_100r_gpu_mixed_device/`

Previous comparison results/plots are archived in the corresponding
`reference_archive/ABCD_100r_comparison_gpu_with_mixed_A/` folders. These are not
fresh all-GPU comparisons. The old A worker log/config are also preserved in
the result reference archive. Nothing was deleted.

Fresh A results will use `newResults/IID/A_clean_no_attack_ce_100r_gpu/`, and plots
will use `newPlots/IID/A_clean_no_attack_ce_100r_gpu/`. This replaces which run
occupies the canonical A path without overwriting the archived evidence.
The existing completed B/C/D result and plot directories are untouched.

After the rerun, the standard GPU comparison paths can reference fresh A with
the completed fresh B/C/D. Scientific analysis must use actual manifests and
remain deferred until the user's explicit completion/analysis trigger.

`gui/queues/iid_abcd_100r_gpu_fresh.json` additionally records all four fresh
configs. It is not the A-only start preset; use the A-only queue for this request.

## Fixed protocol

A remains clean IID BASIL with zero actual attackers, channel noise disabled,
ordinary CE, Snapshot Selection enabled and S=5. There is no EBM, anchor, CART,
momentum, clipping, or weight decay. Preserve 100 rounds, five complete epochs,
batch 512 including the final 392, SGD initial LR 0.05 with the approved round
decay, seed 2025, the paired partition and canonical initialization hashes.
The existing 4 GiB GPU cap, free-memory check, exclusive worker lock and finite
checks are unchanged.

## Preparation verification

Compile checks passed. The focused suite completed with 48 passed and two
display-dependent GUI tests skipped. A's no-training CLI validation reported
valid, fresh, GPU, 100 rounds, executionStarted=false. Configuration contrasts,
partition hash, and initialization hash match the actual completed B/C/D manifests.
All 5,759 preserved files retain matching SHA-256 manifest digests, including
the archived A/comparison outputs and untouched B/C/D evidence. The mathematical
source and GPU worker verification fingerprints are unchanged. See
`docs/IID_FRESH_A_GPU_PREPARATION.json` for machine-readable preparation evidence.
