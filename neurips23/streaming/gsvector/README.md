# gsvector-pg streaming adapter

Native PostgreSQL dense streaming adapter for `run.py --nodocker`. It requires the
paired gsvector/gsvector-pg PQ layout fix for MSTuring's 100 dimensions.
MSTuring configuration uses L2. The shared adapter also supports native IP with
gsvector-pg 0.1.1 and is reused by `neurips23/ood/gsvector`.
Sparse vectors, filtered search, and a competition Docker image are not
implemented by this native ANN adapter. Separate explicitly named PostgreSQL
exact baselines are available for the Filter/Sparse tracks.

## Connection and safety

Install `requirements.txt` into the benchmark Python environment. Supply
`GSVECTOR_PG_DSN` explicitly; the adapter does not connect to a default database.
Use a **dedicated** PostgreSQL instance with `shared_preload_libraries='gsvector'`.
The adapter creates a uniquely named schema and never reuses user tables.
`GSVECTOR_KEEP_SCHEMA=1` retains that schema for diagnosis; otherwise `done()`
drops only schemas owned by this adapter instance.

From the benchmark directory:

```bash
export GSVECTOR_PG_DSN='host=/path/to/private/socket port=55488 dbname=postgres user=YOUR_USER'
export GSVECTOR_ARTIFACTS='/path/to/run/events'
export BIGANN_RESULTS_DIR='/path/to/run/framework-results'
export BIGANN_DATA_DIR='/path/to/existing/big-ann-data'

python run.py --nodocker --algorithm gsvector --dataset msturing-30M-clustered \
  --neurips23track streaming --runbook_path neurips23/runbooks/final_runbook.yaml \
  --count 10 --runs 1 --rebuild --force
```

This is the complete runbook command, **not a claim that all 1280 steps have
already passed**. Use the parent workspace's `scripts/run-bigann-streaming.sh`
to create/stop a private instance automatically and collect evaluation artifacts.
Its default diagnostic run uses a separately named `msturing-60k-streaming`
subset (60K accessible vectors, 100 queries, at most 50K live vectors).

`BIGANN_LOCAL_DATASETS` optionally names a JSON manifest for local dense subsets:
each entry supplies absolute `base`/`queries` bin paths, `dtype`, `rows`,
`query_count`, `metric`, and writable `groundtruth_dir`. Existing official
dataset names cannot be overridden. This prevents reduced query sets from
silently masquerading as official full runs.

## Mutation and query semantics

- Input IDs are benchmark tags, preserved through replacement; results contain
  tags, not the row number of the replacement's source vector.
- First insertion bulk-loads its actual vectors and builds the initial index.
  Setup does not preload the dataset or anticipated inserts.
- Later inserts use bounded binary COPY chunks; replacement uses SQL UPDATE;
  deletion uses SQL DELETE. Mutation chunks commit separately, and an error
  aborts the benchmark rather than being ignored.
- In synchronous maintenance mode (`gsvector.worker=off`), maintenance explicitly
  uses `VACUUM (PARALLEL 0)`. Parallel VACUUM skips inline drain and can otherwise
  leave the alternate memtable slot occupied.
- Full compaction occurs at `compact_runs` runs. This is a simple host baseline,
  not a claim of an optimized streaming compaction policy.
- Each query batch uses `threads` independent PG connections. Parameters use the
  extension's binary vector format. EXPLAIN must select `ann_idx`; no sequential
  scan or exact-search fallback is used. Small graphless runs remain possible and
  are reported in the event sidecar.
- Query-argument sweeps start fresh workloads instead of replaying insertions
  into the previous argument group's populated index.

### Experimental maintenance and query controls

`release_input_cache` is an opt-in boolean index parameter (default `false`).
After a successful mutation COPY/UPDATE commits, the adapter can advise away
whole pages of the consumed, contiguous, **read-only numpy memmap** input slice,
before index maintenance. It does not close the caller's map or change any file
bytes; a later reread loads the same data again. Heap arrays, strided views,
writable/COW maps, partial boundary pages and unsupported platforms are skipped.
The policy uses no future operations, dataset IDs, query results or ground truth.

This is drop-behind for input I/O, not an exclusion from memory accounting:
input reads, COPY, advice and index work remain within the same timed resource
scope. `input_cache_eligible_bytes`, `input_cache_advised_bytes`,
`input_cache_calls`, `input_cache_errors` and `input_cache_s` are recorded in
mutation maintenance totals. Advised bytes are **not** a measurement of freed
RAM; validate residency and cgroup pressure separately. Advisory errors do not
invalidate an already committed mutation. Use private benchmark inputs: file
cache is shared with any other process reading the same inode, so drop-behind
can slow those readers. `gsvector-streaming-retire-input-release` is the explicit
experimental configuration; it is not a claim of full-run admission.

`vacuum_memtables` is a separate opt-in boolean (default `false`). After normal
drain/compaction/reclamation it runs PostgreSQL `VACUUM (PARALLEL 0)` on both
internal heap slots belonging to this index. Base-table VACUUM only attempts
explicit truncation of the non-active slot; a logically empty active slot can
still have many physical pages, and every query scans both slots. Vacuuming the
internal heaps can remove that repeated empty-page work. The adapter records
`vacuum_mt_calls` and `vacuum_mt_s`, already included in mutation elapsed time.

This is **not SQL TRUNCATE** and never skips a source based on estimated row
counts. PostgreSQL retains versions required by old snapshots or in-flight
transactions; truncation may be deferred by concurrent readers. It requires
permission to vacuum the index's internal heaps. The combined experimental
configuration is `gsvector-streaming-retire-input-mt`; its performance must be
measured over both mutations and queries, not just the faster query phase.

`reclaim_backoff_max` is an opt-in index parameter (default `0`, disabled).
With explicit tiered reclamation enabled, an `insufficient_reclaimable` receipt
doubles that run's retry interval from `reclaim_cooldown` up to this maximum,
measured in maintenance rounds. The maximum must be at least the base cooldown
and no greater than 1,000,000. Successful/retired runs do not transfer backoff
history to their replacements; removed runs and new workloads discard it.
The event receipt includes `retry_after_rounds`.

This only schedules calls to the unchanged server-side MVCC-safe reclamation
API. It never infers safety from benchmark IDs or anticipated operations.
Retries continue without new deletes, allowing old snapshots to release.
Conversely, new deletes can wait up to the maximum interval before a target
is eligible again; resource contention can delay execution further. It can
reduce futile scans but can also increase query cost or reclamation debt.

`grouped_rerank` is an optional boolean query parameter forwarding the existing
`gsvector.grouped_rerank` GUC. Omitting it inherits the connection default;
removing an override resets it on reused connections. It changes candidate
grouping, not search breadth. Older extension versions lacking this GUC will
reject an explicit setting rather than silently ignore it.

`gsvector-streaming-reclaim-control` and `gsvector-streaming-reclaim-backoff`
are diagnostic 8-thread configurations differing only in `reclaim_backoff_max`.
They do not enable grouped reranking, reduce queries, or assert 8GB/one-hour
admission. The original configurations and default behavior remain unchanged.

`federate_k` is an optional numeric query parameter in `1..8`, forwarding the
per-source candidate multiplier (server default `1.5`). It is independent of
`search_l`; removing an override resets the session default, including on
reused single-query connections. Changing it can change recall and result IDs.

`gsvector-streaming-retire-only` is an **experimental, not full-run-validated**
configuration. Its extra single-run reclamation uses
`reclaim_min_dead=1, reclaim_min_dead_fraction=1.0`, so only a wholly
MVCC-reclaimable run may retire; a partly live run is not rebuilt by that API.
Normal insertion/drain and tiered compaction still build graphs: "retire-only"
does not describe all maintenance. Input budget is 1M rows per maintenance
round, at most two runs, with bounded no-benefit backoff. Its query setting
`federate_k=2.0` restored sampled recall in a final-state diagnostic. Removing
dead runs can change adaptive deepening; neither the quality nor the overall
performance is guaranteed for every checkpoint. Queries and mutation semantics
are not shortened or anticipated.

Two additional **unvalidated full-run controls**, not new defaults:

- `gsvector-streaming-retire-reconnect` changes only `reuse_connections=false`.
  Each worker still runs its entire query chunk on one connection, then closes
  it. This is not one connection per query. It tests cross-checkpoint backend
  and handle-cache lifetime; it is not a proven fix.
- `gsvector-streaming-retire-codeoff` changes only `build_neighbor_codes=false`.
  It tests the existing compact graph layout through normal construction.
  Fixed-topology diagnostics showed lower memory/cold I/O but slower warm
  searches, so it must not be assumed faster overall. Verify the emitted run
  layouts too: this is a build-session GUC, not a persisted index-wide policy
  automatically inherited by every background maintenance connection.

## Measurement boundaries

`cache_session_setup` is an optional query argument, default `false`; `true`
requires `reuse_connections=true`. On an owned persistent worker connection,
it reuses the vector adapter, session settings, and the result of a previous
successful ANN `EXPLAIN` check only while its complete query configuration,
schema, operator, vector type, k, backend and local maintenance generation match.
It does **not** cache or pin the PostgreSQL execution plan: PostgreSQL still
plans/executes ordinary `ORDER BY ... LIMIT` statements. An initial EXPLAIN
does not prove later statements select the same plan if schema or statistics
change. This mode requires exclusive ownership of worker sessions and a stable
index and planner configuration during a measured run. After external DDL,
ANALYZE, session changes, or any uncertain external state, call
`adapter.invalidate_query_sessions()` before the next query; arbitrary changes
by other clients cannot be detected without another database round trip. The
adapter invalidates its own cache on index build/maintenance and drops failed
cached sessions without retrying their queries. Each SQL statement keeps its
normal autocommit snapshot; this option does not use batch SRF or pipeline.

The official runner's `search_times` includes each full query batch, connection
setup and diagnostic overhead. It is not pure graph-search latency.
Streaming index construction belongs to the first insertion step, not setup.
The framework's `index_size` is therefore only the setup-time memory delta.
`GSVECTOR_PG_PIDFILE` includes the server process tree in PSS snapshots; this is
not a peak-memory measurement or a competition resource cap.

Per-operation JSONL events contain elapsed time, live row counts and run/graph
statistics. HDF5 stores their path and SHA-256 rather than an oversized JSON
attribute. The host/no-Docker runner does not enforce competition CPU, memory or
wall-time limits. A successful run does not imply a recall or throughput target
has been met.

## Tests

```bash
python -m unittest discover -s tests -p test_gsvector_streaming.py -v
```

Binary framing and input validation run offline; the mutation/query/replay test
requires `GSVECTOR_PG_DSN` and creates/removes its own schema.
