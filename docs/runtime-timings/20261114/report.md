# Recorded training and evaluation runtimes

These are historical observations from the current 150-run validation matrix, not a controlled speed benchmark.

One ring round contains 100 sequential client activations, each with five mini-batch SGD steps. The reported round time is each completed run's recorded protocol elapsed time divided by 500. It is not a directly measured median of 500 individual round durations.

The recorded loop time includes training, selection, calibration, repair, regular validation of all client models, and checkpoint/status work within that loop. It excludes environment startup, queue waiting, GPU admission gates, cross-server transfers, final independent verification, and discarded work not retained in a resumed checkpoint. It is not GPU kernel time or calendar time.

Continuation totals include the original 100-round elapsed time. Hosts, concurrent worker load and execution optimizations differ. Some runs changed host after round 100. Clean controls were fresh later runs. Consequently, ratios in this table must not be claimed as intrinsic algorithmic speedups.

The ring is simulated inside each experiment worker; these are not measurements of communication among 100 physical machines.

## Reference comparison

Impaired methods below use Hidden attack + noise, sigma 0.01. Clean uses the fresh FedSharp direct-transfer control (mu zero, no attack/noise/defense). BASIL clean controls are retained separately in the CSV, not pooled as extra seeds.

| Dataset/split | Method | Average round, median across seeds | 500 rounds, median | 500-round range | Hosts over trajectory |
| --- | --- | ---: | ---: | --- | --- |
| mnist / IID | Clean (FedSharp control) | 4.39 s | 00:36:37 | 00:36:22 to 00:37:19 | themachine |
| mnist / IID | BASIL | 17.80 s | 02:28:22 | 01:54:30 to 03:33:23 | hal9k; hal9k -> themachine |
| mnist / IID | FedSharp | 39.36 s | 05:28:01 | 04:18:03 to 05:36:25 | hal9k -> themachine |
| mnist / non-IID | Clean (FedSharp control) | 4.39 s | 00:36:37 | 00:36:34 to 00:36:46 | themachine |
| mnist / non-IID | BASIL | 13.19 s | 01:49:56 | 01:45:50 to 01:54:07 | hal9k |
| mnist / non-IID | BASIL + ACDS | 19.77 s | 02:44:44 | 02:33:17 to 02:46:32 | hal9k -> themachine |
| mnist / non-IID | FedSharp | 37.59 s | 05:13:15 | 04:17:45 to 05:40:08 | hal9k -> themachine |
| mnist / non-IID | FedSharp+ | 52.14 s | 07:14:28 | 06:21:14 to 07:26:51 | hal9k; hal9k -> themachine |
| cifar10 / IID | Clean (FedSharp control) | 7.18 s | 00:59:51 | 00:54:50 to 01:00:44 | themachine |
| cifar10 / IID | BASIL | 23.25 s | 03:13:45 | 03:07:23 to 03:57:53 | hal9k; hal9k -> themachine |
| cifar10 / IID | FedSharp | 56.24 s | 07:48:41 | 06:58:23 to 09:31:07 | hal9k -> themachine |
| cifar10 / non-IID | Clean (FedSharp control) | 7.26 s | 01:00:29 | 00:55:04 to 01:00:37 | themachine |
| cifar10 / non-IID | BASIL | 23.04 s | 03:12:00 | 02:47:29 to 03:15:19 | hal9k -> themachine |
| cifar10 / non-IID | BASIL + ACDS | 23.38 s | 03:14:48 | 03:12:37 to 03:16:58 | hal9k |
| cifar10 / non-IID | FedSharp | 57.32 s | 07:57:40 | 06:47:33 to 09:32:42 | hal9k -> themachine |
| cifar10 / non-IID | FedSharp+ | 82.56 s | 11:28:00 | 10:49:33 to 12:32:51 | hal9k; hal9k -> themachine |

BASIL + ACDS and FedSharp+ have no IID cases in the selected study. They must not be assigned invented IID timing values.

## Existing noise-only runs

Only BASIL MNIST noise-only runs existed when this report was prepared. The new 18-case FedSharp/FedSharp+ noise-only campaign has no measured runtime yet.

| Dataset/split | Method | Average round | 500 rounds |
| --- | --- | ---: | ---: |
| mnist / dirichlet | BASIL | 17.79 s | 02:28:14 |
| mnist / iid | BASIL | 12.47 s | 01:43:53 |

## Separate official-test evaluation timing

These saved local CPU evaluation times cover 100 client models, each classifying all 10,000 test images in batches of 256 with four TensorFlow intra-op threads. They include prediction, metric calculation, progress writes and compressed prediction export after the models are loaded. They exclude checkpoint loading, are not single-image latency, and cannot predict HPC training time. Only the non-IID, Hidden attack + noise, seed-2025 final models were tested.

| Dataset | Method | Evaluation loop seconds for 1,000,000 client-image predictions |
| --- | --- | ---: |
| mnist | BASIL | 2.556 |
| mnist | BASIL + ACDS | 2.578 |
| mnist | FedSharp | 2.888 |
| mnist | FedSharp+ | 3.784 |
| cifar10 | BASIL | 26.525 |
| cifar10 | BASIL + ACDS | 25.672 |
| cifar10 | FedSharp | 25.512 |
| cifar10 | FedSharp+ | 26.847 |

## Reproduce

Run `python3 scripts/report_training_timings.py --destination <new-empty-directory>`. The source manifests, selected run IDs, completion counters and input checksums are checked. Original source results are read only.

All CSV timing quantities are seconds. Clock formatting is hours:minutes:seconds. The new campaign also saves per-round wall intervals in `round_timings.jsonl`, explicitly identifying partial rounds after recovery.
