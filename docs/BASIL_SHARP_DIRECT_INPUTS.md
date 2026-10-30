# Direct training inputs and measured completion time

The thesis runner now reads the original MNIST training archive and five
CIFAR-10 training batches directly into process memory. It never reads or
repairs `experiments/cache/*/x_train_float32.npy`. The exact normalization,
image order, label order, and all data fingerprints match the independently
verified completed study. A mismatch in the raw sources stops execution. No
normalized cache files are created. The new loader is included in the producer
source fingerprint and is used for preparation, both GPU checks, every worker,
and independent final inference.

MNIST discovery checks the original archive in `~/.keras/datasets/mnist.npz`,
then the raw archive at `experiments/cache/mnist/mnist.npz`. If neither exists,
the original approximately 11 MB compressed archive is downloaded once to
`/var/tmp/papermerge-<numeric-uid>/raw-data/mnist.npz`, with a lock and atomic
publication. Normalized arrays are still produced only in memory and verified
against the same full-data fingerprints. Interrupted downloads are removed.
The preflight below reports the actual MNIST location and verifies both complete
datasets without creating any experiment jobs.

The scientific matrix is unchanged: 84 configurations, 100 clients, 100 rounds,
five local mini-batch updates, and three seeds. Earlier failed plans remain
untouched; use a fresh output directory. On hal9k, write checkpoints and outputs
to its local disk rather than the slow shared home filesystem.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

thesis_run=/var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030
if python scripts/check_basil_sharp_inputs.py
then
  mkdir -p "$thesis_run"
  nohup python -u scripts/run_basil_sharp_thesis.py \
    --gpu-count 2 --ram-mib-per-worker 8192 --ram-headroom-mib 16384 \
    --execute --wait-for-resources --output "$thesis_run" \
    --ntfy-topic papermerge-ojaswi \
    > "$thesis_run/adaptive_supervisor.log" 2>&1 < /dev/null &
  echo $! > "$thesis_run/supervisor.pid"
fi
```

Both server GPUs must pass their execution checks before training starts. The
8 GiB host reservation per worker and 16 GiB host headroom permit about nine
workers initially given the reported 90 GiB available RAM. Live admission may
queue more cases. Do not start another cache repair or installation.

```bash
tail -f /var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030/adaptive_supervisor.log

python scripts/run_cart_comparison.py --monitor \
  --output /var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030
```

For the overall ETA, the monitor divides activations completed during this
supervisor invocation by its measured elapsed wall time. It divides all remaining
activations, including queued cases, by that overall speed. It reports activations
per second, time remaining, and the predicted finish date/time in America/New_York.
No completion forecast is shown before observed activation progress. A saved
baseline excludes activations completed before a resume. The estimate updates as
the mix of datasets, methods, concurrency, and checkpoint cost changes; it is a
forecast rather than a completion guarantee. Final writes can continue after the
activation count reaches its total. The browser uses the delivered overall ETA
and displays its finish time in America/New_York.

After completion, run independent inference and export from this same absolute
output path. Copy the compact export into the shared repository before removing
local results. The local output directory is specific to hal9k; it is not shared
with themachine or retained in Git automatically.

[Local validation receipt](https://github.com/Ojaswi03/PaperMerge/blob/a731e45b08c4fc66c613fa33342d900a65bb7eda/docs/validation/BASIL_SHARP_DIRECT_INPUTS_READY_20261030.json):
65 Python tests, eight browser checks, and 28 GPU checks passed. Direct loading
of both complete datasets matched their archived fingerprints; the loader test
used about 1.16 GiB peak host RAM and the full GPU checks about 2.50 GiB.
Server read speed is not inferred from the local timing. The actual two server
GPUs must still pass their own execution checks before training.

[Archive-discovery follow-up](validation/BASIL_SHARP_DIRECT_PATHS_READY_20261030.json)
adds tests for an existing repository archive, a missing-archive download,
download reuse, and cleanup after a connection failure. The preflight must
report `INPUTS_VERIFIED` on the actual server before background execution begins.
