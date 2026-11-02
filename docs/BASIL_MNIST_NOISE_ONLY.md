# MNIST BASIL noise-only controls

`configs/research/hal9k_mnist_basil_noise_only_100r.json` expands to six new
configurations: IID and nonIID, each with seeds 2025, 2026 and 2027.
All use MNIST, BASIL single-model selection with a 34-predecessor window,
100 clients, 100 complete ring rounds, five mini-batch SGD steps per activation,
and independent absolute Gaussian link noise with standard deviation 0.01.
Noise is active from the first available transmission through round 100.

The effective attacker count is zero and the attack switch is disabled for the
entire run. The inherited attack-type/onset fields are inactive metadata.
There are no ACDS, FedSharp or FedSharp+ configurations in this preset.
The original matrix defaults and completed study identities are preserved.
This adds 60,000 client activations and 300,000 SGD steps without rerunning the
completed clean or attacked studies.

The data, initialization, receiver-local selection, learning-rate schedule,
partitions and validation holdout follow the existing matched BASIL setup.
Evaluation is validation accuracy, not official benchmark TEST accuracy.
The resulting standalone plots have no attack-onset line or attack title.

## Launch on hal9k

Connect with `ssh os675524@hal9k.cs.ohio.edu`, then run:

```bash
(
set -e
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
nvidia-smi
free -h
paper_run="/var/tmp/papermerge-$(id -u)/hal9k_mnist-basil-noise-only-100r-$(date +%Y%m%d-%H%M%S)"
python scripts/run_basil_sharp_thesis.py \
  --config configs/research/hal9k_mnist_basil_noise_only_100r.json \
  --output "$paper_run" --describe
mkdir -p "$paper_run"
nohup setsid python -u scripts/run_basil_sharp_thesis.py \
  --config configs/research/hal9k_mnist_basil_noise_only_100r.json \
  --output "$paper_run" --execute --wait-for-resources \
  --ntfy-topic papermerge-ojaswi \
  > "$paper_run/adaptive_supervisor.log" 2>&1 < /dev/null &
echo $! > "$paper_run/supervisor.pid"
printf 'Run directory: %s\n' "$paper_run"
tail -f "$paper_run/adaptive_supervisor.log"
)
```

The description must report six configurations, `families: ["basil"]`,
`environments: ["noise_only"]`, `effectiveAttackers: [0]`,
`attacksEnabled: false`, channel sigma 0.01 and `trainingStarted: false`.
The subsequent launch verifies input fingerprints and both GPUs before starting
training, then uses the existing shared queue and RAM/VRAM admission checks.
`setsid` gives the supervisor and its workers a separate terminal session.
This matters because the supervisor installs its own interrupt handler:
`nohup` alone inside the subshell above does not isolate it from Ctrl+C sent
to the foreground log viewer. With the separate session, Ctrl+C closes the
log viewer while training continues.

## Connect the permanent browser monitor

The launcher's `--ntfy-topic` sends scheduler status notifications. To publish
the dashboard and send its link through ntfy, close the log viewer with Ctrl+C
and run the following on hal9k. It selects the newest directory created by the
noise-only launcher above and requires its frozen plan to exist. It starts or
retargets only the read-only publisher; training continues unchanged.

```bash
(
set -e
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
paper_run="$(python - <<'PY'
from pathlib import Path
import os
root = Path('/var/tmp') / f'papermerge-{os.getuid()}'
runs = sorted(root.glob('hal9k_mnist-basil-noise-only-100r-*'))
if not runs:
    raise SystemExit('No MNIST BASIL noise-only run directory found.')
if not (runs[-1] / 'plan.json').is_file():
    raise SystemExit('The newest run has no plan yet; check its preparation log.')
print(runs[-1])
PY
)"
python scripts/share_cart_monitor.py \
  --output "$paper_run" --background --interval 1 \
  --ntfy-topic papermerge-ojaswi
python scripts/share_cart_monitor.py --print-link
)
```

For monitoring, set `paper_run` to the exact directory printed by the launcher
in each new shell, then use
`python scripts/run_cart_comparison.py --monitor --output "$paper_run"`.
Never launch a second supervisor against that directory. Resume on hal9k uses
`python scripts/run_basil_sharp_thesis.py --output "$paper_run" --resume --execute --wait-for-resources --ntfy-topic papermerge-ojaswi`,
without a preset or changed settings.

If the campaign reports `SUPERVISOR_RECOVERY_NEEDED`, use the operational
handover rather than launching another original supervisor:

```bash
python scripts/rebalance_basil_sharp_live.py --output "$paper_run"
python scripts/rebalance_basil_sharp_live.py \
  --output "$paper_run" --execute --background --ntfy-topic papermerge-ojaswi
```

The first command reports the actual discoverable worker count and verifies
the saved plan/GPU contracts. The second starts a detached supervisor with
`start_new_session=True`, adopts surviving workers and loads original
checkpoints for unfinished configurations whose processes have exited.
Saved RUNNING rows alone do not prove that their worker processes still exist.
Inspect `adaptive_supervisor.log` and `live_scheduler.log` to distinguish an
interrupt from an operational exception. Preserve the frozen plan and sources.

After all six finish, independently verify the retained checkpoints with
`python scripts/verify_basil_sharp_thesis_results.py --output "$paper_run"`.
Keep the original checkpoints on hal9k. Compact export uses
`scripts/export_basil_sharp_thesis.py` with a new destination after verification;
the six measurements can then be added to conference comparisons using the
existing clean and attacked controls.

## Publish the verified compact results

After independent verification finishes, run the following on hal9k. It checks
the existing six-run verification receipt without rerunning inference, exports
only the compact evidence and PNG/SVG figures, and stages the explicitly
allowlisted result directory. The original checkpoints stay on hal9k.

```bash
(
set -e
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
paper_run="$(python - <<'PY'
from pathlib import Path
import json, os
root = Path('/var/tmp') / f'papermerge-{os.getuid()}'
runs = sorted(root.glob('hal9k_mnist-basil-noise-only-100r-*'))
if not runs:
    raise SystemExit('No matching run found.')
run = runs[-1]
plan = json.loads((run / 'plan.json').read_text())
receipt = json.loads((run / 'independent_validation.json').read_text())
assert len(plan['configs']) == 6
assert receipt['status'] == 'ALL_COMPARISON_MODELS_VERIFIED'
assert receipt['source'] == plan['source']
assert len(receipt['verified']) == 6
assert {v['runId'] for v in receipt['verified']} == {c['runId'] for c in plan['configs']}
assert all(c['dataset'] == 'mnist' and c['comparisonCase']['arm'] == 'basil_noise_only'
           and c['comparisonCase']['effectiveAttackers'] == 0
           and c['comparisonCase']['effectiveSigma'] == 0.01 for c in plan['configs'])
print(run)
PY
)"
result_dir="docs/results/hal9k_mnist-basil-noise-only-100r-20261102"
python scripts/export_basil_sharp_thesis.py \
  --output "$paper_run" --destination "$PWD/$result_dir"
git add -- "$result_dir"
git diff --cached --check
python scripts/check_publication_policy.py
git diff --cached --stat
git commit --only -m "Add verified MNIST BASIL noise-only results" -- "$result_dir"
git push origin main
)
```

The exporter validates complete run budgets, saved accuracy arrays and source
snapshots, then writes the SHA256 manifest. It refuses conflicting existing
exports. Do not force-add ignored paths or upload the original run directory.
