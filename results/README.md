# Result directories

This directory separates working measurements from published evidence:

- Files directly under `results/` are mutable local outputs produced by legacy/default script paths. They are
  useful for development, but they are not the source of current headline claims.
- `results/runs/<UTC>-<experiment>/` contains timestamped candidate runs created by `scripts/repro.sh`.
- `results/published/<snapshot>/` contains immutable reviewed snapshots. Existing snapshot contents must not be
  edited in place because their manifest hashes are part of the evidence contract.

The current headline snapshot is
[`ascend910_9382-migration-20261013`](published/ascend910_9382-migration-20261013/).
Its scope, protocol limitations, and known metadata errata are indexed in
[`results/published/README.md`](published/README.md). The reader-facing interpretation is in
[`docs/benchmarks/results.md`](../docs/benchmarks/results.md).

Create a new timestamped run for new measurements; after review, publish it under a new snapshot ID rather than
overwriting an existing directory.
