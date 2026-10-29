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

python scripts/share_cart_monitor.py \
  --output "$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1" \
  --tmux cart-browser-monitor \
  --ntfy-topic papermerge-ojaswi
```

The command prints **Share this link** followed by the campaign URL. It also
sends the link to your ntfy topic after the first successful update. Your
professor can open it in Chrome without SSH, VPN or a ChatGPT account. Only the
publisher needs outbound HTTPS access from HPC; no inbound HPC port is opened.
The publisher stays in its own tmux session after SSH disconnect. Training does
not need restarting.

The publisher uploads small allowlisted progress snapshots every ten seconds.
The page refreshes in place every five seconds, showing GPU, dataset/partition,
seed, status, completed rounds, activations, mean TRAIN-holdout accuracy, elapsed
time and ETA. It warns when updates are more than 45 seconds old. Completion
preserves the final snapshot and stops the publisher. No historical curves are
invented from a late connection; the existing experiment reports remain the
source for full round-by-round results.

Checkpoint files, model weights, data examples, source files, logs, usernames,
absolute paths, process IDs and raw error messages are excluded. A private
per-feed write credential is stored with mode 0600 in the ignored campaign
output as `browser_share.json`; it is never included in the share link or a
public API response. Viewers cannot update the feed or control training. Anyone
can create a separate feed, but cannot modify another feed without its private
credential. The link is unlisted rather than an authenticated access boundary.

To retrieve the same link later:

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
