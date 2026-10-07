# Fresh B/C/D GPU queue

Prepared at the user's request after stopping B's CPU reconstruction. Preparation
performs no research training. A's completed 100-round evidence is retained.

## Launch manually

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

If the GUI is already open, use **Queue → Load Queue** and select
`gui/queues/iid_bcd_100r_gpu_fresh.json`. Its in-memory queue does not automatically
reload when the persisted file changes. Confirm B → C → D, Pending, GPU,
100 rounds, **0/100**, and **Fresh R0→R99 — GPU** before pressing **Run Queue**.
Loading the preset never starts execution.

Fresh configurations are in `gui/configs/IID/sequential_basil_100r/gpu/fresh/`.
The previous `iid_abcd_100r_gpu.json` preset remains a CPU-continuation preset:
do not load it for this fresh study. The B/C/D GPU builder presets now load the
fresh configurations; A's preset remains unchanged.

## Evidence and output routing

B's stopped reconstruction is preserved under
`newResults/IID/recovery_archive/B_attack_clean_ce_100r_gpu_before_fresh/`.
Its checkpoint retains 660 verified CPU activations and 50 BASIL snapshots.
Nothing was deleted. The original CPU prefixes, completed A results/plots, and
historical 50-round references are unchanged. The pre-preparation persisted queue
is backed up at `gui/queues/iid_queue_before_fresh_bcd.json`.
The previous B worker log/config are archived alongside the reconstruction
directory, so the next B worker log does not mix fresh execution with old recovery.

Verification: 46 focused tests passed, two display-dependent tests skipped;
compile checks and three no-training CLI validations passed. SHA-256 manifest
digests match for all 5,585 preserved files. Machine-readable preparation evidence
is in `docs/IID_FRESH_GPU_PREPARATION.json`.

Fresh output paths remain:
- `newResults/IID/B_attack_clean_ce_100r_gpu/`
- `newResults/IID/C_attack_noise_ce_100r_gpu/`
- `newResults/IID/D_attack_noise_ebm_100r_gpu/`

Plots use the same folder names under `newPlots/IID/`. The four-way report can
still use completed A together with fresh B/C/D in
`newResults/IID/ABCD_100r_comparison_gpu/`; do not regenerate final analysis until
the user says **tests are complete**.

The fresh configurations omit both `recoverySourceDirectory` and
`deviceTransition`. They restore canonical initialization, not final CPU weights.
If these new runs are subsequently interrupted, their own GPU checkpoints may
be resumed; Fresh describes their scientific starting state, not permission to
overwrite existing research evidence.

## Unchanged protocol and qualification

All three retain seed 2025, the same expected IID partition and initialization
hashes, 100 rounds, five complete epochs, batch 512 including the final 392,
SGD with the existing round decay, S=5, receiver-local Snapshot Selection,
strict handoff, and attackers [0, 1, 5, 7] starting at Round 20.
B uses clean-channel CE; C uses absolute sigma_e=0.010 with CE;
D uses the same channel with CE + 0.0001 ||grad CE||², with no lambda or anchor.
The existing GPU memory cap, exclusive GPU worker lock, and finite-value checks
remain unchanged.

B/C/D will be GPU from initialization through completion. A still contains its
38-activation CPU prefix followed by GPU continuation. This qualification remains
visible in final comparisons; the study is not entirely all-GPU.
