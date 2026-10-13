# Published result snapshots

Published snapshot directories are immutable evidence bundles. The current headline source is
`ascend910_9382-migration-20261013`; earlier directories are retained for provenance and must not be mixed into
the current aggregate.

| Snapshot | Role |
|---|---|
| `ascend910_9382-migration-20261013` | Current headline C2C, E2E, application-shape, real-transform, and scale evidence |
| `ascend910_9382-migration-20261012` | Superseded migration snapshot retained for audit |
| `ascend910_9382-revalidation-20261010` | Earlier revalidation snapshot retained for audit |
| `ascend910_9382-cann9.0.0-v2` | Legacy-unverified evidence |
| `ascend910_9382-cann9.0.0` | Legacy-unverified evidence |

## Metadata errata for `20261013`

The immutable files are not rewritten when a label is later found to be imprecise. These errata affect metadata
wording, not the stored timings, correctness status, geometric means, or win/tie/loss counts.

1. `scale_slice.json` labels the statistic as a per-trial median. The actual and intended device statistic is the
   minimum `device_chain` span within each trial, followed by the median across three trials. The raw 36 trial
   rows and the 12 summary values are internally consistent; the current collector now records this definition.
2. In `e2e.json`, legacy field `bare_dev_mean` stores the minimum sample (`dev_min`), despite its name. Headline
   R2C/ACL calculations use `bare_dev`, which is the min-of-means value, so published performance claims are not
   affected. New collector output emits a correctly defined `bare_dev_mean` plus an explicit `bare_dev_min`.
3. The E2E command records `--reps 10`, while the collector enforces 20 repetitions for native and 10 for
   self/ACL. All sides use three-round min-of-means, but they were executed in grouped order rather than randomly
   interleaved. Current documentation discloses these exact counts and order.

Use `python3 scripts/publish_results.py --check --snapshot ascend910_9382-migration-20261013` to verify the
immutable bundle against its manifest.
