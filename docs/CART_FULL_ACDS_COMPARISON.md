# Full ACDS and updated CART comparison

The `full` HPC launcher evaluates two methods under the joint Hidden attack and
absolute Gaussian model channel: SHARP with 5% ACDS fitting-data sharing, and
SHARP with CART training-time local-prior adjustment plus consistent peer
eligibility when the SHARP survivor guard falls back. The methods are separate
arms, `acds` and `prior_fallback`; old experiments are preserved as their original
versions and are not substituted into this comparison.

## Frozen experiment matrix

| Setting | Value |
| --- | --- |
| Datasets | MNIST and CIFAR-10 |
| Partitions | IID and Dirichlet alpha 0.5 |
| Seeds | 2025, 2026, 2027 |
| Methods | SHARP + ACDS, SHARP + updated CART |
| Clients and rounds | 100 clients, 100 complete ring rounds |
| Local optimization | Five mini-batch SGD steps per activation |
| Per-run budget | 10,000 activations, 50,000 SGD steps |
| Attack | 33 configured identities, retained Hidden transform from zero-based round 20 |
| Channel | Absolute coordinate Gaussian sigma 0.01 |
| ACDS data | Per-owner 5% fitting-data circulation, five batches |
| CART | Local fitting counts, raw-logit MCC ingress and commit, eight peer candidates with guard-consistent fallback |
| Total | 24 complete configurations |

The primary endpoint remains the mean client accuracy over rounds 91--100.
Report final-round, per-class, honest-client and worst-client outcomes as
additional measurements. Report sample SD across seeds, not across the 100
correlated client models as if they were independent repetitions.

ACDS receives additional raw fitting examples; CART does not. The experiment
matches computation steps and physical impairments, not information availability.
This is the documented ACDS data-availability adaptation integrated with SHARP,
not a claim to reproduce all anonymity/transport mechanisms in BASIL.

The new CART version uses established training-time logit adjustment, retains
the original MCC/private-probe gate, and makes peer eligibility follow the same
fallback already used by SHARP. Correcting fallback behavior is engineering
work, not an independent novelty claim. The candidate research contribution is
the combined receiver-local admission and complete-update commitment design.
The full comparison assesses usefulness; it cannot establish priority over the
literature by itself. Component ablations remain necessary for attribution.

## Start on the four-GPU server

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

bash scripts/launch_cart_comparison_hpc.sh full \
  --gpu-count 4 --jobs-per-gpu 2 \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029"
```

The launcher checks resources, verifies each selected GPU and the prior-adjusted
objective, freezes a new plan and source snapshots, then starts a detached tmux
supervisor. At most eight configurations run concurrently with these settings;
clients within each experiment still activate sequentially around the ring.
Use a fresh output if changing GPU layout, method source or runtime.

```bash
python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029"

python scripts/share_cart_monitor.py \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029" \
  --tmux cart-full-browser --ntfy-topic papermerge-ojaswi
```

The first command closes independently of training. The second prints the
GitHub Pages link for this new run and continues after SSH disconnects.

After all cases complete, independently infer the final checkpoints and export
the verified records using `verify_cart_comparison_results.py` and
`export_cart_comparison.py`, with this run's output path. Full results are not
available merely because these launch instructions exist.
