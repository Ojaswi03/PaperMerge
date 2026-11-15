# Recover the isolated noise-only CUDA timeout

The themachine campaign reported seventeen completed runs and one failure:
`cifar10-dirichlet-2025-sharp_cart-3e2daad0801f`, physical GPU 3. The error was
`CUDA_ERROR_LAUNCH_TIMEOUT` during candidate logits evaluation. Archive CRC
checks passed for every checkpoint member. This establishes archive integrity,
not the underlying cause of the timeout or complete checkpoint validity.

`scripts/retry_noise_only_timeout.py` handles this failure outside the frozen
scientific producer. All ninety producer files remain unchanged. It requires
the original eighteen-case plan, seventeen completed other runs, the original
server/runtime/GPU profile, and no active experiment workers or controllers.

Before requeueing, it checks the complete saved checkpoint with the original
strict loader, ordered activation/SGD history, zero-attack evidence, retained
client hashes, validation metrics and optimization contract. The saved cursor
determines where training resumes. Reported progress can move backward if some
activations were never durably checkpointed.

It then acquires the original GPU lease and runs the existing device gate in a
fresh subprocess under the original GPU profile. The gate uses a separate
staging directory. An unsuccessful gate cannot change the target run or clear
its quarantine. A passed check must also match the retained runtime exactly.
Only then is the failed run requeued and that GPU's quarantine entry cleared.
Other quarantine entries and the seventeen completed runs remain untouched.

## Run on themachine

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

noise_run="/var/tmp/papermerge-1718278706/themachine_fedsharp-noise-only-500r-20261115-013857"

python scripts/retry_noise_only_timeout.py \
  --output "$noise_run" \
  --run-id cifar10-dirichlet-2025-sharp_cart-3e2daad0801f \
  --execute --background --ntfy-topic papermerge-ojaswi
```

Without `--execute`, the utility performs read-only checkpoint checks. With
`--execute`, it preserves the original failed row, log, checkpoint, device gate,
quarantine, campaign state, optimization receipt and round timings under
`operational_adapters/noise_timeout_retries/<runId>/<attempt>/`. The checkpoint
backup is a hard link; the existing atomic checkpoint writer preserves that
original inode when resumed training saves a new checkpoint. Completed result
measurement hashes are checked again after the device gate.

The command prints the exact saved activation and GPU-check log path before
starting the device check. Verification can take time. `--background` starts
the original campaign supervisor only after successful requeueing, with the
same output directory and original checkpoint. It skips the seventeen completed
runs. It also restarts the publisher through the original launcher and sends
the dashboard link through ntfy.

Monitor using the same directory:

```bash
python scripts/run_fedsharp_noise_only_auto.py --output "$noise_run" --monitor
```

If the GPU check fails, inspect the printed `gpu-check.log` and retain the
quarantine. The recovery does not reset GPUs, change driver settings, stop
other users' processes, move the checkpoint to another device or alter model
hyperparameters. A successful fresh check does not establish the root cause
or guarantee that another timeout cannot occur. A later failure is set aside
for diagnosis; this utility does not create an automatic retry loop.

After all eighteen runs complete, use the normal
[automatic-launcher export command](FEDSHARP_NOISE_ONLY_AUTO.md#export-completed-results).
Independent inference verification is still required before publishing results.
