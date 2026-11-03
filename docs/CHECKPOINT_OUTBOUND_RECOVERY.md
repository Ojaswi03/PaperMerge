# Recovery of an isolated outgoing-array CRC failure

The final hal9k scan found 34 valid archives and one repeatable CRC failure in
`outbound.npy` for `cifar10-dirichlet-2025-basil_acds-03863eb17a93`. Its other
four archive members passed CRC. This identifies repeatable read failure, not
the underlying hardware/storage/runtime cause.

`scripts/recover_fedsharp_outbound.py` can recover this derived array without
training. It requires the explicitly identified damaged archive SHA256, local
DONE ownership, no active training worker, matching frozen source/configuration
and checkpoint contract, a complete activation/SGD history, valid trained-model
hashes, and exact saved checkpoint/metric agreement. It applies the original
seeded attack transform to the retained final models (or copies honest models),
then requires every reconstructed vector to match its recorded outbound SHA256.
Any mismatch aborts the recovery.

All intact `.npy` members are copied byte-for-byte with normal CRC verification.
The candidate passes all ZIP CRCs and the original strict checkpoint loader.
No CRC check is suppressed, no damaged array is accepted by changing its CRC,
and no recorded model hash or metric is changed. Before installation, the
original damaged archive is retained through a hard link under the host-local
`operational_adapters/outbound_recovery/` directory. The replacement and saved
recovery receipt record original/repaired archive hashes, preserved-member
hashes, matched-vector counts and the recovery script hash. The script also
retains its own source snapshot. A PREPARED receipt indicates an incomplete
installation and cannot be exported as a recovered result.

Use `--execute` to install a successfully validated candidate. Without it, the
script builds and validates a temporary candidate, then discards it. Both modes
leave training records and metrics intact; temporary candidate preparation
uses local scratch space. The command does not request GPU execution.

After recovery, use `scripts/export_recovered_fedsharp_fleet.py` instead of the
ordinary exporter for hal9k. It first invokes the unchanged fleet exporter with
fresh CPU inference enabled, then validates and includes the recovery receipt
and source snapshot, regenerates the complete SHA256 manifest and installs the
final export atomically. It never publishes the preserved damaged checkpoint.
This separate wrapper leaves themachine's already-running exporter unchanged.
The usual combined 150-run export check still applies before Git publication.

This recovers one deterministic derived artifact with evidence from recorded
hashes. It does not establish that every historical training computation was
error-free or remove the requirement for final independent inference.
