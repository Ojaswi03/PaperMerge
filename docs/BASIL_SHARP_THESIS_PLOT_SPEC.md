# Requested BASIL–SHARP thesis plots

**Current launch procedure:** [direct input loading and measured ETA](BASIL_SHARP_DIRECT_INPUTS.md).
This supersedes the cache repair/restart commands below. The damaged shared
normalized cache is left alone; original training archives are loaded into RAM.

The author requests six curves for IID (three BASIL and three SHARP) and eight
curves for non-IID (four BASIL and four SHARP). The author explicitly removed
the duplicated fourth impairment case.

| Case within each family | Byzantine attack | Channel noise | Additional distinction |
| --- | --- | --- | --- |
| 1 | No | No | Clean execution of the stated algorithm |
| 2 | Yes | No | Attack without channel noise |
| 3 | Yes | Yes | Joint attack and channel noise |
| 4, non-IID only | Yes | Yes | BASIL adds ACDS; SHARP adds CART |

The duplicated joint attack/noise case is omitted. The acronym ADSC in the
request is interpreted as the existing ACDS data-sharing mechanism. Sync and
audit the completed 66-case HPC export before deciding which additional matched
experiments are necessary.

## Algorithm identity and matching

BASIL selects one complete predecessor model by local loss, using its b+1
predecessor window. With 100 clients and budget 33, this is 34 predecessors.
The repository's guarded SHARP uses a 66-predecessor screen/rank/average rule.
The recent pooling check is a separately identifiable SHARP selection revision;
its use must be fixed across the SHARP conditions and disclosed in captions.
The completed `ACDS checked` configurations combine ACDS with SHARP selection.
They are not evidence for the requested BASIL + ACDS curve.

Clean algorithm runs retain their declared selection mechanisms: zero actual
attackers does not mean silently disabling the algorithm's model selection.
Identical data partitions, holdout identities, seed groups, architectures,
initializations, communication-noise semantics, attack onset and local SGD
budgets must be checked before any archived curves are reused. A condition
called clean in an older study does not establish those matches by itself.
CART means the full declared CART method; the better admission-only ablation
must not silently replace it under the same legend label.

The preceding experiments used 100 clients, 100 rounds, five mini-batch steps,
seeds 2025–2027, Hidden attack onset at displayed round 21, sigma 0.01 for noisy
conditions, and Dirichlet concentration 0.5. Continuing those settings on MNIST
and CIFAR-10 is the working scope. With the duplicate omitted, that scope has
84 configurations: 2 datasets × 3 seeds × (6 IID + 8 non-IID). Four-GPU worker
admission must respect the final verified memory budgets; case count alone does
not authorize overcommitting devices or RAM.

## Figure layout and review

Prepare four principal line figures: MNIST IID, MNIST non-IID, CIFAR-10 IID,
and CIFAR-10 non-IID. Keep corresponding conditions visually consistent across
figures. Use condition colors and distinguish BASIL from SHARP with line style;
place the six/eight-entry legend outside the plotting area. Mean client accuracy
is averaged within each run before calculating the mean and sample SD across
three seeds. Plot every recorded round without smoothing. Label the vertical
axis according to the actual recorded evaluation split. Titles and legends
should describe methods and conditions, without internal campaign/version names
or test-split banners. SVG is the thesis-folder deliverable; PDF and high-DPI PNG
can remain with the research export.

After the HPC results are pushed and pulled locally, verify complete case/seed
coverage, saved sources, data contracts, metric shapes, independent final
inference, uncertainty calculations and any discrepancy with matched existing
runs. Inspect attack onset, plateaus, collapse, seed variation and method gaps,
including negative outcomes. Then visually inspect each figure for clipped
labels, overlapping legends, distinguishable lines and honest axes. Report that
assessment before the thesis-folder replacement stage.

The requested destination is `/mnt/c/Users/ojasw/Desktop/THESIS/Images`.
It currently contains seven SVG files. Their hashes are inventoried locally in
`.papermerge/thesis-plots/request_20261029.json`; none has been deleted or copied
over. Preserve a recoverable copy of the old images before replacing them with
the reviewed new SVGs.

## New matched run

The separate runner uses genuine BASIL single-model selection and checked SHARP
pooling. For BASIL, ordinary SGD and no proximal/noise penalty are retained; for
SHARP, its proximal and noise-aware objective are retained. Clean cases set the
actual attacker count and channel noise to zero while keeping each algorithm's
selection mechanism active. The non-IID extension adds 5% ACDS sharing to BASIL
and the full CART procedure to the same checked-pooling SHARP base. It does not
substitute the better incoming-only CART variant. This checked-pooling CART
recipe has its own explicit source identity and requires a new run.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

python scripts/run_basil_sharp_thesis.py \
  --gpu-count 4 --execute --wait-for-resources \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-100r-20261029" \
  --tmux basil-sharp-thesis --ntfy-topic papermerge-ojaswi
```

There are 84 cases, with at most 21 workers per GPU on four devices. The default
3 GiB TensorFlow cap plus 1 GiB context reservation and 8 GiB device headroom
requires at most 94,208 MiB per GPU. Host reservation is 16 GiB per worker plus
32 GiB headroom (1,376 GiB for all cases). Live admission may queue workers when
other activity reduces available memory; allocations are never pooled across
GPUs. Every selected GPU must pass its own bound-profile execution checks.

### Two-GPU server with 93 GiB host RAM

On `hal9k.cs.ohio.edu`, the reported two idle GPUs have ample device memory,
but available host RAM is approximately 90 GiB. Use a fresh output directory
and an 8 GiB host reservation per worker with 16 GiB host headroom:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

python scripts/run_basil_sharp_thesis.py \
  --gpu-count 2 --ram-mib-per-worker 8192 --ram-headroom-mib 16384 \
  --execute --wait-for-resources \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030" \
  --tmux basil-sharp-hal9k --ntfy-topic papermerge-ojaswi

python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030"
```

At the reported available RAM, this admits approximately nine concurrent
workers across the two GPUs, with the remaining cases queued. Live memory
checks can lower concurrency. Swap is not counted as available host RAM.
Scientific configurations remain 84 cases, 100 clients, and 100 rounds;
only the resource reservations change. The 3 GiB TensorFlow cap, 1 GiB CUDA
context reserve, and 8 GiB device headroom remain unchanged.

Stop the waiting `basil-sharp-thesis` supervisor on `themachine` before launching
here. Plans are bound to their original host, GPU identities, source, and output
directory; do not resume the old server's plan on this host. Use the new output
path in browser-monitor, verification, and export commands as well.

[The two-GPU readiness receipt](validation/BASIL_SHARP_HAL9K_READY_20261030.json)
records 89 passing local tests, 28 passing GPU checks with the 8 GiB host
profile, and the measured memory checks. These measurements inform the
reservation, rather than imposing a hard per-process host-memory limit. Each
actual server GPU still runs its own checks before training starts.

For a read-only description of the requested matrix, add `--describe` and omit
`--execute`, `--wait-for-resources`, `--tmux`, and `--ntfy-topic`. For interrupted
recovery, use `--resume --execute --output ... --tmux ...` with the original
source, host and numerical profiles; do not supply new scientific settings.

Monitor in another terminal:

```bash
python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-100r-20261029"

python scripts/share_cart_monitor.py \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-100r-20261029" \
  --tmux basil-sharp-browser
```

After completion:

```bash
python scripts/verify_basil_sharp_thesis_results.py \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-100r-20261029"

python scripts/export_basil_sharp_thesis.py \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-100r-20261029" \
  --destination "$PWD/docs/results/basil-sharp-thesis-100r-20261029"
```

The supervisor generates four plain-labelled SVG/PDF/PNG figures under
`thesis_figures/`. After the export is pushed and pulled locally, those figures
and the full measurements still require review before the Windows thesis image
folder is replaced.

Local validation is recorded in
[the readiness receipt](validation/BASIL_SHARP_THESIS_READY_20261029.json): 101
Python tests and eight browser tests passed, 28 GPU checks passed under the
3 GiB cap, and four short 100-client CIFAR-10 cases passed fresh CPU inference
for all 400 final models with zero accuracy discrepancy. These are execution
checks; each actual HPC GPU must still pass its own saved-profile gate.

BASIL is evaluated under the common training budget in this simulator; the
plots are not presented as reproduction of a published BASIL accuracy table.
The ACDS simulation implements fitting-example sharing and does not establish
a cryptographic anonymity guarantee.

## Recovering the changed CIFAR-10 cache on hal9k

The first hal9k verification attempt exited before training. A read-only audit
found CIFAR-10 image values as high as 1.6549019813537598 and a different image
checksum from the already saved plan. MNIST and both label caches still matched.
Earlier checks had loaded the images successfully. This establishes a cache
change; it does not identify which process or storage condition caused it.

After the supervisor has exited, rebuild from the existing raw training batches:

```bash
python scripts/repair_basil_sharp_cache.py \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030" \
  --apply
```

The tool stages a separate rebuild and requires exact agreement with the saved
image and label checksums. It preserves the damaged image file and a JSON
receipt in a `cache-recovery-*` directory under the output folder, then replaces
only the image file atomically and makes it read-only. It refuses to proceed if
the plan is in use, labels differ, the raw rebuild differs, or the cache changes
again during recovery. Omitting `--apply` checks the rebuild without replacing
the shared cache. No clipping, rescaling of damaged data, plan editing, or
scientific source changes are involved. Checksums are rechecked after install.

Only after `REPAIRED_AND_VERIFIED` or `ALREADY_VERIFIED`, resume the existing plan
without new hardware/scientific flags. Since hal9k has no tmux installation:

```bash
nohup python -u scripts/run_basil_sharp_thesis.py \
  --resume --execute --wait-for-resources \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030" \
  --ntfy-topic papermerge-ojaswi \
  >> .papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030/adaptive_supervisor.log \
  2>&1 < /dev/null &

echo $! > .papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030/supervisor.pid
```

[Local recovery validation](validation/CIFAR_CACHE_RECOVERY_READY_20261030.json)
records 16 passing tests and restoration of an isolated, deliberately damaged
50,000-image cache from raw batches. This is local tool validation, not a claim
that the shared server cache has already been repaired.

### Follow-up: the failed plan also differs from the verified archive

The server rebuild completed and passed range/label validation. Its image
identity is `f8d4e2cda7d5b07dab87dc18de073ebd26513ca2a097b6b7b8b97b991cc23462`,
which matches the independently verified completed study. However, the failed
hal9k plan recorded `a2561bfa25520b58dd61adfbfc0a09d8dfcad0d4a63f053b249b198e656243a8`.
The repair correctly refused to install data that differed from that plan.
Do not change its checksum or resume it with different data.

Instead, preserve the failed plan, reuse the already-written rebuild, and verify
it against the checksum-protected completed archive:

```bash
python scripts/install_verified_cifar_cache.py \
  --output "$PWD/.papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030" \
  --rebuild "$PWD/.papermerge/local-tests/basil-sharp-thesis-hal9k-100r-20261030/cache-recovery-z4gg04hr" \
  --reference-plan "$PWD/docs/results/cart-contribution-100r-20261029/plan.json"
```

This operation checks the archive's plan checksum, rejects active supervisors
and any training evidence in the failed plan, checks the existing rebuild,
preserves the damaged file with a hard link, then atomically renames the verified
file into place. It does not rewrite either 600 MB payload. The failed plan and
receipt remain intact. A fresh plan is mandatory.

The server reports 7.2 TB free on its local `/dev/sda2` filesystem under
`/var/tmp`, while its shared home showed very slow writes. Put the new outputs
on that local disk:

```bash
mkdir -p /var/tmp/papermerge-os675524/basil-sharp-thesis-verified-100r-20261030
nohup python -u scripts/run_basil_sharp_thesis.py \
  --gpu-count 2 --ram-mib-per-worker 8192 --ram-headroom-mib 16384 \
  --execute --wait-for-resources \
  --output /var/tmp/papermerge-os675524/basil-sharp-thesis-verified-100r-20261030 \
  --ntfy-topic papermerge-ojaswi \
  > /var/tmp/papermerge-os675524/basil-sharp-thesis-verified-100r-20261030/adaptive_supervisor.log \
  2>&1 < /dev/null &
echo $! > /var/tmp/papermerge-os675524/basil-sharp-thesis-verified-100r-20261030/supervisor.pid
```

Launch only after the installation command succeeds. Use this new absolute
output path for monitoring, independent verification, and export. Output files
are local to hal9k; export the completed results to the shared repository before
removing them. Input arrays continue to be read from the checked shared cache.
[Local validation](validation/CIFAR_ARCHIVE_INSTALL_READY_20261030.json)
records 21 passing tests and reuse of an existing full 50,000-image rebuild,
with no payload copying, exact archive agreement, and preservation of the failed
plan. Scientific producer fingerprints remain unchanged.
