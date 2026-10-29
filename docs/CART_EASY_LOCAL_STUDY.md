# Short local CART diagnosis before longer experiments

The 50-round prior-adjusted CART run stayed near 27.7% accuracy and used the
receiver-owned model in 94 of its last 100 activations. The following tests
separate missed candidate opportunities, class penalties, and local-prior
training effects before allocating more time to long trajectories.

## Frozen checkpoint audit

`scripts/audit_cart_peer_window.py` evaluates the saved boundary at 50 completed
rounds for receiver IDs 0, 5, ..., 95. It uses only the private fitting probes,
performs no SGD, and verifies the input checkpoint hash before and after.
All receivers see that same frozen boundary. This is not a replay of round 51
or a new end-to-end learning result.

| Candidate set | Receivers with a peer above the owned score | Within the original 0.01 band |
| --- | ---: | ---: |
| Eight recent screened received peers | 0/20 | 1/20 |
| Full screened received window | 2/20 | 3/20 |
| Full screened pre-link window | 3/20 | 4/20 |

The median best-peer minus owned score improves from -0.08475 with eight
received peers to -0.05587 with the full screened window. Before the newest
link perturbation, the full-window median is -0.05374. Removing the retention
penalty gives macro-score improvements over the owned model in only 3/20
full-window received cases. Thus freshness truncation misses some opportunities,
while neither truncation, the newest noise draw, nor the class penalty alone
explains the sampled receivers' strong preference for their own models.

The pre-link copy is the saved outbound state. Byzantine transforms and older
channel/training ancestry are still present; it is not an uncorrupted oracle.
Sender identities are recorded for diagnosis and do not affect candidate scores.
The audit holds the current receiver's class targets and calibrated allowance
fixed when comparing received and pre-link evidence. The figures are local
probe scores, not benchmark accuracies or confidence bounds.

## Eight-case local training screen

The `easy` preset explicitly selects four mechanisms:

| Arm | Change |
| --- | --- |
| `cart` | Existing CART, ordinary local CE, eight recent screened peers |
| `prior_cart` | Training-only local-prior adjustment, eight peers |
| `prior_macro` | Same prior-adjusted worker; macro-only ingress and commit scoring |
| `prior_wide` | Same prior-adjusted worker and class scoring; full screened peer window |

Each is tested with no physical impairments and with joint Hidden attack plus
absolute sigma 0.01 noise. These are defended-algorithm environments, not the
legacy untreated clean controls. All cases use CIFAR-10, 100 Dirichlet clients
at alpha 0.5, seed 2025, five rounds, and five SGD steps per activation. Hidden
starts at zero-based round 3, so only two rounds are attack-active. Fitting data,
holdout, initialization, optimizer schedule and runtime are matched.

The wider arm preserves peak screening and the original class-quality gate.
It expands the coherent peer shortlist up to the configured predecessor window;
it does not average all peer models or use known attacker labels for selection.
It adds inference work, not SGD steps. New scalar telemetry separates macro
recognition from the class penalty and records candidate ages and probe counts.
The default eight-peer arm retains its existing numerical decisions.

Start or recover locally:

```bash
bash scripts/launch_cart_local_easy.sh
```

Monitor after sourcing `environment/basil-noise-env/bin/activate`:

```bash
python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/cart-easy-dirichlet-20261029"
```

Compare paired clean/joint trajectories, class recall, worst-client accuracy,
ownership routes, zero update scales, runtime and score decomposition. Five
rounds and one seed screen mechanisms; they do not establish long-run superiority
or robustness. Select the next same-runtime 50-round comparisons after inspecting
these outcomes. Heavy runs are not automatically queued by this launcher.

## Completed local outcomes

All eight cases completed in 27 minutes 18 seconds with no failures. Each
completed 500 activations and 2,500 SGD steps. Fresh CPU inference reproduced
the exported final accuracies for all 800 client models exactly. Results and
producer snapshots are archived in `docs/results/cart-easy-dirichlet-20261029/`.

| Method | No physical impairments | Joint attack and noise |
| --- | ---: | ---: |
| Original CART | 10.593% | 21.889% |
| Prior-adjusted CART | 11.936% | 26.654% |
| Prior-adjusted macro only | 12.049% | 26.823% |
| Prior-adjusted full peer window | 11.936% | 25.858% |

Prior adjustment helps this short joint comparison. The 0.169 percentage-point
macro advantage is small and one seed does not establish a reliable difference.
The wider window is 0.796 points below the eight-peer prior-adjusted case here,
so the checkpoint opportunity audit did not translate into an accuracy gain.
The poor no-impairment results require further diagnosis of the defended
training/selection path. These cases keep guards, proximal anchoring and CART
active and are not the older unprotected clean reference. Comparing them with
joint conditions does not establish that channel noise itself improves learning.
The planned impairment-isolation experiments can separate the effects of attack
and noise under the same defended method before costly confirmation studies.
