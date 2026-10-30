# Share the live CART monitor in Chrome

The [browser dashboard](https://ojaswi03.github.io/PaperMerge/)
is public and read-only. A campaign-specific link is created when the HPC
publisher first connects. Share that complete `?feed=...` link with your professor;
the base site alone has no live campaign attached.

The share page uses the repository's `github.io` address. The progress-storage
API remains on the original hosted service; the browser reads it with an
explicit read-only cross-origin policy. Existing feed IDs, private write keys
and running publishers remain valid. The old `chatgpt.site` webpage returns
HTTP 410 (retired); it no longer displays the monitor. The update API remains
available independently of that retired webpage. To convert an existing link, keep its
`?feed=...` portion and put it after the GitHub address, or pull the changes and
use `--print-link` below. No training or publisher restart is required.

Run on HPC while training continues:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
git pull --ff-only

python scripts/share_cart_monitor.py --temporary \
  --output "$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1" \
  --ntfy-topic papermerge-ojaswi
```

The command prints **Share this link** followed by the campaign URL. It also
sends the link to your ntfy topic after the first successful update. Your
professor can open it in Chrome without SSH, VPN or a ChatGPT account. Only the
publisher needs outbound HTTPS access from HPC; no inbound HPC port is opened.
Keep this command running in the foreground while you want to share. **Ctrl+C
deletes the temporary feed**, and viewers see “Sharing session ended” on their
next refresh. Training does not stop. Each invocation creates a new link, with
its write credential held only in memory. If cleanup cannot reach the server,
the temporary feed expires within 45 seconds of its last successful update.
Do not use `--tmux` for this temporary mode. Training does not need restarting.

The publisher uploads small allowlisted progress snapshots every second by default.
Successful uploads are silent after the initial link; terminal messages appear
only for issues or when sharing ends.
The dark page refreshes in place every second, showing the same column order
and formatted values as the terminal: GPU, STATUS, DATA/SPLIT, SEED, SIGMA, ENV,
CONDITION, ROUND, ACTIVATIONS, ACC, ELAPSED, ETA. Header cards show overall wall
time, measured activations per second, remaining time, expected finish in Eastern
Time, active workers per GPU, pending/failed counts, clients, and rounds.
All configuration rows are available in a scrollable table with sticky headings.
It warns when updates are more than 45 seconds old. Completion
preserves the final snapshot while your temporary publisher stays open. No historical curves are
invented from a late connection; the existing experiment reports remain the
source for full round-by-round results.

Checkpoint files, model weights, data examples, source files, logs, usernames,
absolute paths, process IDs and raw error messages are excluded. A private
per-feed write credential stays in memory in temporary mode. In optional
persistent mode it is stored with mode 0600 in the ignored campaign output as
`browser_share.json`. It is never included in the share link or a
public API response. Viewers cannot update the feed or control training. Anyone
can create a separate feed, but cannot modify another feed without its private
credential. The link is unlisted rather than an authenticated access boundary.

## Optional persistent sharing

For a link that continues after SSH disconnect instead, use the older persistent
mode. Ctrl+C on a terminal monitor does not revoke a persistent feed:

```bash
python scripts/share_cart_monitor.py \
  --output "$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1" \
  --tmux cart-browser-monitor --ntfy-topic papermerge-ojaswi
```

To retrieve a persistent link later:

```bash
python scripts/share_cart_monitor.py \
  --output "$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1" \
  --print-link
```

To inspect the publisher, independently of training:

```bash
tmux capture-pane -p -t cart-browser-monitor -S -30
tail -n 30 "$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1/browser_monitor.log"
```

After retrying an interrupted training campaign, restart the publisher command
with the same output to retain the same link. Do not start a second publisher
while its tmux session still exists. Network errors are retried and do not stop
training. The browser view can lag the terminal by its upload/poll interval.
Upload time is subtracted from the requested interval; slow requests never
overlap. Workers retain their existing reporting cadence, so one-second polling
does not invent additional accuracy or activation measurements.

## Current hal9k run without tmux

These monitor changes are outside the frozen training fingerprint. Pulling them
does not require retraining or changing the original plan. If the supervisor
stopped with the `VmRSS` error, use
[the checkpoint recovery command](BASIL_SHARP_SUPERVISOR_RECOVERY.md) first.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
thesis_run=/var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030

nohup python -u scripts/share_cart_monitor.py \
  --output "$thesis_run" --interval 1 --ntfy-topic papermerge-ojaswi \
  > "$thesis_run/browser_monitor.log" 2>&1 < /dev/null &
echo $! > "$thesis_run/browser_publisher.pid"

tail -f "$thesis_run/browser_monitor.log"
```

The log prints the campaign-specific GitHub Pages link. Ctrl+C stops only this
log viewer. The publisher and training continue in the background. The publisher
also sends the link through ntfy after its first successful update. Start only
one publisher for this output; its lock prevents simultaneous uploads by another
persistent publisher. After a GitHub Pages update, reload the same feed URL.
