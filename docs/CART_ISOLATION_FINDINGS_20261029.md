# Verified impairment-isolation findings and the next CART revision

The HPC export in `docs/results/cart-isolation-32-10r-20261029/` contains 32
completed ten-round cases, one seed, and 100 clients per case. All 199 recorded
file checksums match. Its independent CPU inference receipt covers 3,200 final
client models with zero difference in their saved accuracy values. Attack onset
is zero-based round 3 in this diagnostic, not round 20 used for the full study.

| Environment | IID CART | IID macro only | Dirichlet CART | Dirichlet macro only |
| --- | ---: | ---: | ---: | ---: |
| Clean physical channel, no attack | 20.224% | 20.994% | 11.279% | 11.609% |
| Attack only | 11.951% | 11.706% | 10.733% | 10.988% |
| Noise only | 29.670% | 29.496% | 24.397% | 24.363% |
| Joint | 29.454% | 29.351% | 23.990% | 23.748% |

This does not establish a consistent incremental benefit from the class penalty.
It also does not show that physical channel noise improves learning. Sigma is
used both by the communication simulation and by the RLA coefficient. Changing
sigma changes the optimization objective and can change screening and admission
decisions along the whole learning trajectory.

## Fallback inconsistency found in code and local checkpoints

The SHARP survivor guard returns to full-window ranking when too few snapshots
pass screening. The CART coherent-peer shortlist independently reconstructed
only the passing set. When all copies fail, the SHARP pool is still defined,
but CART previously offered no individual peer candidate. This could make a
receiver keep its own model or the pooled candidate even when the guard had
already deliberately fallen back to the full window.

In the preserved local five-round Dirichlet screen, both ordinary and
prior-adjusted clean CART have all-fail screens at all 100 activations in rounds
2 through 5, and zero coherent peer candidates. Joint CART has approximately
65--66 raw survivors and eight coherent choices at those stages. These are
direct local checkpoint findings. The compact HPC exports do not contain the
original per-activation screen flags; their clean q=1 versus noisy q=33 SHARP
summaries are consistent with this mechanism but do not independently establish
all HPC fallback counts.

The `cart_fallback` and `prior_fallback` arms follow the SHARP guard's eligibility
when it falls back. They retain the private MCC decision rule, score band,
eight-peer cap, full-update check, head repair, and original SGD count. The
historical screen-only rule remains the default for its control arms. New round
summaries record screen/all-fail/guard counts and coherent candidate availability.

## Matched five-round fallback pilot

| Method | Existing local control | Consistent fallback |
| --- | ---: | ---: |
| Ordinary CART, clean | 10.593% | 11.678% |
| Prior-adjusted CART, clean | 11.936% | 20.015% |
| Ordinary CART, joint | 21.889% | 21.889% |
| Prior-adjusted CART, joint | 26.654% | 26.654% |

The joint configurations have no full-window guard fallback in this pilot,
and their endpoints remain unchanged. The clean prior-adjusted result is
8.079 percentage points higher. This is a bounded one-seed diagnostic and
does not establish general superiority or literature novelty. The new full
comparison uses prior adjustment plus consistent fallback under joint attack
and noise, with 100 rounds, 100 clients, both datasets/partitions, and three seeds.

## Negative counterfactual

`docs/validation/CART_SCREEN_COUNTERFACTUAL_20261029.json` keeps the saved local
clean outbound states fixed and adds fresh synthetic channel perturbations at
three receivers. All candidates still fail the screen at sigma 0, 0.001, 0.003,
and 0.01. The threshold is about 12.08 while observed peak ratios are much
larger. Adding the newest link draw to an already degraded clean checkpoint
does not reproduce the successful noisy trajectory. The difference involves
earlier training/selection history and cannot be assigned to one fresh noise draw.

Next noise-mechanism studies should independently control channel perturbation,
regularizer strength, and fallback behavior. The requested full ACDS/CART run
is a matched method comparison, not that factorial causal experiment.
