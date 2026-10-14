# Adaptive PostgreSQL filtered ANN

`gsvector-filter` builds a native L2 graph and a GIN label index. Every ANN SQL
query contains the complete label-AND predicate. Small candidate sets use GIN
plus exact ranking; otherwise ANN candidate budgets expand until k valid hits
are found or a bounded ceiling triggers an explicitly counted exact fallback.

Counters in `pg_filter_stats_json` distinguish small exact, successful ANN and
fallback queries. A successful ANN branch must have an `ann_idx` execution plan.
No graph-inline label filter is claimed. All attempts for a query use the same
repeatable-read, read-only transaction snapshot. Query-argument sweeps reuse
the same index and preserve immutable event snapshots.

The exact baseline remains available as `pg-filter-exact`. Submission image
dispatch remains pending the organizer's final interface.
