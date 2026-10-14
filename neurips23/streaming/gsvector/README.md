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

## Measurement boundaries

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
