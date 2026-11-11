# Published local test evaluations

Start with [the four-method report](four-method-official-test-20261110/report.md).
This package preserves the completed local evaluation files, including saved
per-image predictions, original receipts, CSV summaries, figures and checksums.
Absolute local paths inside original receipts record provenance; use the
[model manifest](../../../models/manifest.json) for portable model downloads.

- `full-official-test-20261110`: four FedSharp/FedSharp+ cases, four million predictions.
- `basil-full-test-20261110`: four BASIL/BASIL + ACDS cases, four million predictions.
- `four-method-official-test-20261110`: combined summary and four PNG/SVG figures.
- `local-single-image-20261110-index0`: earlier single-image demonstration, not test-set accuracy.
- `eurosat-ood-20261110`: fixed 2,000-image unrelated-data confidence diagnostic, not EuroSAT classification accuracy.

All official-test runs are final round 500, non-IID, Hidden attack + noise,
seed 2025. This one-seed study is distinct from the three-seed validation study.
Image 0 was demonstrated earlier. No models were trained or tuned afterward;
metrics for the other 9,999 images are also saved. See the research index for
interpretation, runtime differences and reproduction commands.
