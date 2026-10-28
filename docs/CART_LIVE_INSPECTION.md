# More information during an active campaign

The inspector reads saved JSON telemetry and does not modify training files,
load models, send notifications, or stop/restart workers. It is outside the
frozen training and compiled-transfer runtime fingerprints, so it can be pulled
and used while the HPC campaign continues.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
RUN="$PWD/.papermerge/local-tests/cart-v2-focused-72-100r-6slot-resilient-v1"

python scripts/inspect_cart_campaign.py --output "$RUN" \
  | tee /tmp/papermerge-live-inspection.txt
```

For machine-readable output:

```bash
python scripts/inspect_cart_campaign.py --output "$RUN" --json \
  > /tmp/papermerge-live-inspection.json
```

The text report includes:

- Current manager and supervisor status, update times in UTC, recorded PIDs
  and whether those PIDs exist when run on the same host.
- Every selected configuration, including completed and pending runs, with
  mean/worst accuracy, runtime adapter, checkpoint-file presence and ETA source.
- Measured seconds per activation since the runtime switch when that worker
  declares switch markers. A missing rate is shown explicitly.
- The queue slot limiting the forecast and how much of its remaining time uses
  assumed speeds. This explains why an ETA can stay high while active jobs speed up.
- CART versus the matched control at each seed's latest common completed round.
- Three-seed means and paired accuracy differences at **one common round across
  all three seeds**, keeping each noise level separate. Negative effects remain.
- A clean reference at that same round only when matching histories and source/
  data/initialization identities are available; otherwise it reports waiting.

An INTERRUPTED phone message can belong to the intentional pre-upgrade stop.
Check the current manager/supervisor statuses and timestamps before treating it
as a new failure. If a manager is stopped while campaign metadata says RUNNING,
the inspector flags that discrepancy rather than claiming everything is healthy.
PID existence alone is not proof that counters are advancing; compare snapshots
and consult worker logs when progress appears stalled.

The snapshot is not atomic: workers may finish another activation or round while
files are being read. Comparisons therefore use saved completed rounds and
validate their node counts and identities. They are exploratory progress checks,
not the prespecified final/last-ten-round endpoint, independent inference,
statistical significance, or proof of a general CART advantage. The displayed
sample SD belongs to the three paired accuracy differences; it is not a
confidence interval. A checkpoint-file presence flag does not validate its
contents; the worker validates it on recovery.

Validation: 20 targeted tests passed, including read-only behavior, unequal-round
alignment, noise separation, missing-seed handling, invalid identity/metric
rejection, and manager-status disagreement. The inspector was also run against
the saved 72-case tiny real-data workflow (12 seed comparisons/four groups).
Those small workflow metrics are not results of the active HPC campaign.

## When protected CIFAR-10 stays near chance

Do not confuse zero failed processes with successful learning. If protected
accuracy is still around 10–15% after many rounds, use the checkpoint audit to
separate pre-attack learning problems from attack-active degradation. This
audit supports the CART v2 survivor guard and reports local training CE, mean
selection count, direct attacked-copy admission, and CART gate/correction counts.

```bash
for AUDIT_CONDITION in attack_noise_defense attack_noise_defense_rla attack_noise_defense_rla_cart
do
  python scripts/audit_sharp_selection.py --output "$RUN" \
    --dataset cifar10 --partition iid --seed 2025 --sigma 0.01 \
    --condition "$AUDIT_CONDITION" --all-rounds
done | tee /tmp/papermerge-cifar-selection-audit.txt
```

It reads only the JSON metadata member of each atomic checkpoint, not its
model arrays. Attacker labels are used post hoc for auditing, never as inputs
to training. With zero-based attackStart=20, the first attack-active displayed
round is 21. Poor learning before then cannot be attributed to the active
Hidden transformation. A larger CART-versus-control difference while both
models remain near chance is not evidence of near-clean recovery. Preserve
the current experiment when developing a separate CART revision.
