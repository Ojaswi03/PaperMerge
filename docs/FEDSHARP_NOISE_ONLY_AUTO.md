# One noise-only launcher for either server

`scripts/run_fedsharp_noise_only_auto.py` reads the actual hostname and selects
the matching launcher and default configuration automatically.

| Hostname | Configuration | GPU policy |
| --- | --- | --- |
| hal9k | `hal9k_fedsharp_noise_only_500r.json` | GPUs 0 and 1 |
| themachine | `themachine_fedsharp_noise_only_500r.json` | Prefer 1 and 3; admit 0 and 2 when usable |

Fully qualified names such as `hal9k.cs.ohio.edu` are recognized. Unknown hosts
are refused. The dispatcher replaces itself with the concrete launcher, so the
real hostname, process ownership, background behavior and original checkpoint
contracts are preserved. It does not change the scientific experiment settings.

Both configurations contain the same eighteen fresh 500-round noise-only runs:
twelve FedSharp cases and six non-IID FedSharp+ cases, three seeds, sigma 0.01
and zero attackers. Use one server campaign. Running this block independently
on both servers repeats the same configurations; it does not divide the work
or create additional independent seeds.

## Start on the selected server

Use this exact block on either supported server. The output directory includes
that server's hostname. Run it once for a fresh campaign.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

noise_run="/var/tmp/papermerge-$(id -u)/$(hostname -s)_fedsharp-noise-only-500r-$(date +%Y%m%d-%H%M%S)"

python scripts/run_fedsharp_noise_only_auto.py \
  --output "$noise_run" --describe \
  --write-configs "$noise_run/configs.json" &&
python scripts/run_fedsharp_noise_only_auto.py \
  --output "$noise_run" --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

Description prints the chosen configuration before execution. Training still
requires the server's actual device gates and live resource admission checks.
The browser link is sent through ntfy after preparation.

Monitor from the same terminal:

```bash
python scripts/run_fedsharp_noise_only_auto.py --output "$noise_run" --monitor
tail -n 40 "$noise_run/noise_only_supervisor.log"
```

Ctrl+C closes only the monitor. In a new terminal, first set `noise_run` to the
exact path printed when starting the campaign. Resume on the same server and
with the same output, without a fresh timestamp or configuration overrides:

```bash
python scripts/run_fedsharp_noise_only_auto.py \
  --output "$noise_run" --resume --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

Automatic hostname selection is not checkpoint migration between servers.
Each concrete launcher preserves its own frozen source and runtime checks.

## Export completed results

The same entrypoint selects the matching CPU verifier:

```bash
python scripts/run_fedsharp_noise_only_auto.py \
  --output "$noise_run" --role export \
  --destination "docs/results/$(hostname -s)_fedsharp-noise-only-500r-20261114"
```

All eighteen runs and fresh independent inference must verify before publication.
On hal9k, the older automatic plot generator can produce
`COMPLETED_REPORT_FAILED` after successful training because it does not support
this new noise-only matrix. The exporter accepts that campaign status only
subject to the same full run, source, measurement, checkpoint and fresh inference
checks. It does not excuse incomplete or failed training. The themachine
launcher writes its own completion summary instead of invoking that plotter.

See the [hal9k guide](FEDSHARP_NOISE_ONLY_500R.md) and
[themachine guide](FEDSHARP_NOISE_ONLY_THEMACHINE.md) for resource policies,
verification scope, export contents and subsequent publication commands.
