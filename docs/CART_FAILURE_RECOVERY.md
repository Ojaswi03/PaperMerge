# Independent worker failures and filesystem recovery

Validation: 45 targeted tests passed, including real child-process failures,
launch failures, missing success manifests, selective retry and injected EIO.
Twelve small real-data six-slot GPU CART cases completed and passed independent
CPU verification. All arrays, telemetry, round metrics and reference metadata
matched the previous scheduler, including five accepted CART corrections.
The [validation receipt](validation/CART_V2_RESILIENT_EXECUTION_20261028.json)
records the source identities. This is execution correctness evidence, not
long-horizon accuracy evidence or proof that an HPC filesystem is healthy.

The recommended launcher is now `run_cart_v2_resilient.py`, selected by
`run_cart_v2_managed.py --scheduler resilient` (also the manager default).
The original parallel and scaled launchers remain unchanged so their frozen
plans and published timing receipts remain verifiable.

## Queue behavior

Six workers per GPU across four GPUs permits 24 concurrent configurations.
Each configuration still has 100 logical nodes and 100 complete rounds when
the new-study commands below are used. Logical nodes remain sequential inside
their configuration; only independent configurations run concurrently.

When one worker exits unsuccessfully:

1. Its log and last committed checkpoint are retained.
2. Its run is marked FAILED and listed in `failed_runs.json` and
   `campaign.json`. Its exit code, error and log path are recorded.
3. An event is appended to `worker_failure_events.jsonl`, retained across retries.
4. Its slot can start the next pending configuration. Other workers continue.
5. The manager sends a configuration-failure ntfy alert when it observes that event.

Each configuration gets one attempt per invocation; deterministic failures
cannot spin in an automatic retry loop. Once the queue is exhausted, the
campaign reports COMPLETED or COMPLETED_WITH_FAILURES. The latter is not a
successful full study. Final matched three-seed figures wait for missing runs
to succeed. Saved completed results remain available for read-only reporting.
The manager sends a final notification in either case.

An explicit later `--resume-plan` invocation skips completed configurations
and retries failed, interrupted or never-started configurations. It uses the
saved horizon, GPU slots, data/partition identities, runtime profiles and
source hashes. It cannot change a 35-round plan into 100 rounds or change
two workers/GPU to six. Original `plan.json` bytes are preserved; the new
orchestration provenance is stored separately in `resilient_execution.json`.

## What the reported Errno 5 means

The message `Input/output error: .../plan.json` reports a filesystem operation
failure. It does not establish a GPU OOM, an invalid learning objective or an
accuracy failure. The earlier manager could stop training when a progress read
raised that exception; worker logs are still needed to determine all failures.

Campaign JSON reads/writes now retry EIO, ESTALE, ETIMEDOUT, EAGAIN and EREMOTEIO
up to four times with bounded backoff. Repeated monitoring EIO is logged while
training continues. Worker-local failures are isolated as above. Persistent
storage failures during required checkpoint/orchestration operations, changed
source/data contracts, explicit user cancellation and other global problems
can still stop the campaign. Retries do not repair an unhealthy filesystem;
check disk/storage health with the HPC administrator if EIO recurs. No training
source or checkpoint format was changed by this operational update.

## Preserve the stopped pilot

The old 22 completed runs need not be repeated to finish that same 35-round
pilot. If desired, after confirming its old workers have exited:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
git pull --ff-only
OLD="$PWD/.papermerge/local-tests/cart-v2-focused-72-35r-managed-v1"
python scripts/hpc.py gpu-status

python scripts/run_cart_v2_managed.py --execute --scheduler resilient \
  --output "$OLD" --resume-plan --ntfy-topic papermerge-ojaswi \
  --tmux cart-v2-recover-35r
```

This deliberately remains **35 rounds and two workers per GPU**. Do not run
this pilot recovery simultaneously with the new study on the same GPUs.

## New study: 100 nodes, 100 rounds, six workers per GPU

First wait until old campaign workers have exited. If an old managed campaign
is still running, stop its named tmux session gracefully with Ctrl+C and wait
for its workers to release their GPU leases. A failed/stopped campaign does
not need another stop signal.

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
git pull --ff-only
python scripts/hpc.py gpu-status
df -h "$PWD"

RUN="$PWD/.papermerge/local-tests/cart-v2-focused-72-100r-6slot-resilient-v1"
export MPLCONFIGDIR="$RUN/matplotlib-cache"
COMMON=(
  --profile hpc --gpus 0,1,2,3 --output "$RUN"
  --jobs-per-gpu 6 --gpu-memory-mib 8192
  --gpu-headroom-mib 8192 --context-overhead-mib 1024
  --ram-mib-per-worker 32768 --ram-headroom-mib 32768
  --datasets mnist cifar10 --partitions iid dirichlet
  --seeds 2025 2026 2027
  --sigmas 0.005 0.01 0.02 --only-sigma 0.01
  --conditions attack_noise_defense_rla attack_noise_defense_rla_cart
    clean attack_noise attack_noise_defense attack_noise_rla
  --nodes 100 --rounds 100 --local-steps 5
  --attack-start 20 --alpha 0.5 --model paper
)

python scripts/run_cart_v2_managed.py --test-notification --ntfy-topic papermerge-ojaswi &&
python scripts/run_cart_v2_resilient.py --preflight "${COMMON[@]}" &&
python scripts/run_cart_v2_resilient.py --verify-gpus "${COMMON[@]}" &&
python scripts/run_cart_v2_managed.py --execute --scheduler resilient "${COMMON[@]}" \
  --ntfy-topic papermerge-ojaswi --tmux cart-v2-resilient-100r

python scripts/monitor_sharp_study.py --output "$RUN" --timezone America/New_York
```

Check preflight: **72 configurations, 100 nodes, 100 rounds, six workers/GPU,
workerFailurePolicy: continue**. This focused study uses only sigma 0.01;
the three seeds stay separate and are averaged at that same noise level.
For only 36 CIFAR-10 configurations, use `--datasets cifar10` and a fresh
output name. The [concurrency guide](CART_HIGH_CONCURRENCY.md) describes memory
budgets and the measured local speedup; HPC timing remains a live estimate.

After queue exhaustion, inspect failures and retry unfinished configurations:

```bash
python -m json.tool "$RUN/failed_runs.json"

python scripts/run_cart_v2_managed.py --execute --scheduler resilient \
  --output "$RUN" --resume-plan --ntfy-topic papermerge-ojaswi \
  --tmux cart-v2-retry-100r
```

Do not overwrite the saved training sources to repair a scientific/numerical
failure inside an existing frozen study. A method change needs a versioned
implementation and new study; temporary operational failures can resume the
existing checkpoint contract unchanged.
