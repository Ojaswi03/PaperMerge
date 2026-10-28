# Higher concurrency without changing CART training

For new launches use the [failure-isolating execution guide](CART_FAILURE_RECOVERY.md):
`run_cart_v2_resilient.py` retains these resource bounds and continues peer
workers after an individual failure. It also supports frozen-plan recovery.
The older scaled launcher retains its original fail-fast behavior; the commands
below now select the resilient launcher.

The separate `run_cart_v2_scaled.py` entry supports one through eight independent
workers per GPU. The active two-slot launcher remains unchanged. Each worker
still owns its model, ring state and random streams. Logical nodes within one
configuration remain sequential. The training fingerprint is unchanged.

## Measured local result

The [benchmark receipt](validation/CART_V2_SCALED_CONCURRENCY_20261028.json)
uses the same six full-TRAIN CIFAR-10 cases: IID/non-IID, three seeds, 100 nodes,
one round, early attack0, both mitigations and CART, on one RTX4070Ti.

| Workers on that GPU | Wall time | Throughput relative to two |
| --- | ---: | ---: |
| 2 | 173.37 s | 1.00x |
| 4 | 123.56 s | 1.40x |
| 6 | 77.50 s | 2.24x |

Final arrays, reference memory, telemetry and round metrics matched exactly.
The [multi-round equivalence receipt](validation/CART_V2_SIX_SLOT_EQUIVALENCE_20261028.json)
also compares twelve small real-data CART cases at two versus six workers,
including five accepted classifier corrections. These are bounded execution
and equivalence checks, not long-horizon efficacy or a guaranteed Blackwell
speedup. Eight-worker scheduling/resource bounds are tested; eight-worker
throughput was not measured locally.

Validation also passed 1,279 tests and 644 subtests (four skips). Independent
CPU verification passed all six full-TRAIN benchmark cases and all twelve
small multi-round cases. All 23 frozen training-source hashes match the
previous CART v2 validation index.

The local GPU/RAM limits required 512 MiB caps and an explicit controlled WSL
accounting mode because WSL returned `[N/A]` for per-process VRAM. The initial
full-cohort budget and per-launch free-memory floors remained enforced. Do
not enable that fallback on the HPC: its per-process VRAM reporting works,
and the deployment uses strict live accounting with 8 GiB caps.

## Conservative admission

For each GPU the required free capacity is:

$$
M_{\mathrm{required}}=J(M_{\mathrm{cap}}+M_{\mathrm{context}})+H_{\mathrm{GPU}}.
$$

Here J is concurrent workers on the GPU, M_cap is each TensorFlow allocation
cap, M_context is an additional allowance for CUDA/runtime context overhead,
and H_GPU is spare capacity beyond those allocations. The defaults are 8 GiB
per worker, 1 GiB per context and 8 GiB device headroom:

| Workers/GPU | Across four GPUs | Required free VRAM/GPU |
| --- | ---: | ---: |
| 4 | 16 | 44 GiB |
| 6 | 24 | 62 GiB |
| 8 | 32 | 80 GiB |

Host-RAM admission also requires the total number of workers times the
per-worker RAM budget, plus host headroom. The default is 32 GiB per worker
and 32 GiB spare host RAM: six workers on each of four GPUs requires 800 GiB
available host RAM. This is a conservative admission budget, not a per-process
RAM enforcement mechanism. The Machine's reported 1.5 TiB RAM can accommodate
that budget if enough remains available.

Before each new worker starts, live GPU allocations are rechecked. Only memory
attributed to this supervisor's own PIDs is credited to its budget. Unrelated
PaperMerge workers are still excluded by the inherited physical-GPU lease.
Other users are not bound by project locks; these checks cannot reserve the
shared machine against them. Fresh worker processes release their GPU contexts
on exit. Persistent CUDA compilation caches are retained.

## Keep the final horizon at 100 rounds

The 35-round campaign is a pilot, not the requested 100-round final study.
The scaled launcher defaults to 100 and the commands below specify it again.
Increasing concurrency reduces wall time; it does not reduce optimizer steps,
node count, data coverage or the attack-active horizon.

Moving from 35 to 100 rounds is approximately 2.86 times the training work.
If the measured local 2.24x gain over two workers transferred to HPC, a
100-round/six-worker study would take roughly 1.28 times the corresponding
35-round/two-worker study. This is conditional arithmetic, not an HPC timing
measurement. Use live measured rates rather than assuming linear scaling.

## HPC commands: fresh 100-round study

First preserve and stop the active 35-round campaign gracefully, then wait
for its workers to exit. The horizon and concurrency are frozen in each plan;
changing either is not an in-place recovery operation. Keep the old output
as pilot evidence. Pulling this update alone does not change its training.

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
tmux send-keys -t cart-v2-managed C-c
python scripts/hpc.py gpu-status
# Wait until your old workers have exited, then continue.
git pull --ff-only

RUN100="$PWD/.papermerge/local-tests/cart-v2-focused-72-100r-6slot-resilient-v1"
export MPLCONFIGDIR="$RUN100/matplotlib-cache"
COMMON100=(
  --profile hpc --gpus 0,1,2,3 --output "$RUN100"
  --jobs-per-gpu 6 --gpu-memory-mib 8192
  --gpu-headroom-mib 8192 --context-overhead-mib 1024
  --ram-mib-per-worker 32768 --ram-headroom-mib 32768
  --datasets mnist cifar10 --partitions iid dirichlet
  --seeds 2025 2026 2027 --sigmas 0.005 0.01 0.02 --only-sigma 0.01
  --conditions attack_noise_defense_rla attack_noise_defense_rla_cart
    clean attack_noise attack_noise_defense attack_noise_rla
  --nodes 100 --rounds 100 --local-steps 5
  --attack-start 20 --alpha 0.5 --model paper
)

python scripts/run_cart_v2_managed.py --test-notification --ntfy-topic papermerge-ojaswi &&
python scripts/run_cart_v2_resilient.py --preflight "${COMMON100[@]}" &&
python scripts/run_cart_v2_resilient.py --verify-gpus "${COMMON100[@]}" &&
python scripts/run_cart_v2_managed.py --execute --scheduler resilient "${COMMON100[@]}" \
  --ntfy-topic papermerge-ojaswi --tmux cart-v2-resilient-100r

python scripts/monitor_sharp_study.py --output "$RUN100" --timezone America/New_York
```

Preflight must show 72 configurations, 100 nodes, **100 rounds**, six workers
per GPU. Up to 24 configurations run concurrently. For only the 36 CIFAR-10
runs, use `--datasets cifar10` and a fresh output name. Use six initially;
eight is capacity-guarded but needs actual HPC throughput measurements before
claiming it is better. Do not change workers-per-GPU midway through a frozen
campaign.

The manager retains automatic three-seed line/bar figures, CSV averages and
ntfy completion/failure notifications. Full operational details are in the
[execution guide](CART_EXECUTION_AND_REPORTING.md). This runtime work does not
establish that CART improves accuracy or fixes previously observed collapse.
