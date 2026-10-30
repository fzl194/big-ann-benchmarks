"""Streaming L2/IP adapter; connect only to an explicitly supplied benchmark PG.

No dataset preloading, anticipated inserts, padding of input dimensions, or
sequential-scan fallback. The first insertion builds the initial index; later
mutations use the index's memtable/drain paths and optional full compaction.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import numpy as np
import psycopg
from psycopg import sql
from psycopg.adapt import Dumper
from psycopg.pq import Format, TransactionStatus
import psutil

from neurips23.streaming.base import BaseStreamingANN
from neurips23.streaming.gsvector.maintenance import ReclaimPlanner, ReclaimThreshold
from neurips23.streaming.gsvector.input_cache import release_consumed_input


class QueryCardinalityError(RuntimeError):
    def __init__(self, found, expected, completed_queries=0):
        super().__init__(f"ANN returned {found} results, expected {expected}")
        self.found, self.expected = found, expected
        self.completed_queries = completed_queries
        self.completed_queries_total = completed_queries


def gather_query_chunks(futures):
    """Wait for all chunks; never hide SQL/corruption errors behind a shortfall."""
    results, errors = [], []
    completed = 0
    for future in futures:
        try:
            result = future.result()
            results.append(result)
            completed += len(result)
        except Exception as exc:
            errors.append(exc)
            results.append(None)
            completed += getattr(exc, "completed_queries", 0)
    if errors:
        for exc in errors:
            if not isinstance(exc, QueryCardinalityError):
                raise exc
        errors[0].completed_queries_total = completed
        raise errors[0]
    return results


class VectorBytes(bytes):
    """Already encoded vector_send-compatible binary parameter."""


def encode_vector(vector):
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not 1 <= len(values) <= 8000 or not np.isfinite(values).all():
        raise ValueError("expected a finite 1..8000-dimensional vector")
    return VectorBytes(struct.pack("!hh", len(values), 0) + values.astype(">f4").tobytes())


def binary_copy_blocks(vectors, ids, batch_size=4096):
    """Bounded binary COPY payload: bigint ID + extension vector (network order)."""
    yield b"PGCOPY\n\xff\r\n\0" + struct.pack("!ii", 0, 0)
    dim = vectors.shape[1]
    row_type = np.dtype([
        ("nfields", ">i2"), ("id_size", ">i4"), ("id", ">i8"),
        ("vector_size", ">i4"), ("dim", ">i2"), ("unused", ">i2"),
        ("values", ">f4", (dim,)),
    ])
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        rows = np.empty(end - start, dtype=row_type)
        rows["nfields"] = 2
        rows["id_size"] = 8
        rows["id"] = ids[start:end]
        rows["vector_size"] = 4 + dim * 4
        rows["dim"] = dim
        rows["unused"] = 0
        rows["values"] = vectors[start:end]
        if not np.isfinite(rows["values"]).all():
            raise ValueError("non-finite vector component")
        yield rows.tobytes()
    yield struct.pack("!h", -1)


def has_index_plan(plan, name="ann_idx"):
    if isinstance(plan, dict):
        return plan.get("Index Name") == name or any(has_index_plan(v, name) for v in plan.values())
    if isinstance(plan, list):
        return any(has_index_plan(v, name) for v in plan)
    return False


class Gsvector(BaseStreamingANN):
    def __init__(self, metric, index_params):
        metric = {"angular": "cosine", "normalized": "cosine", "manhattan":"l1"}.get(metric, metric)
        if metric not in ("euclidean", "ip", "cosine", "l1"):
            raise ValueError("gsvector-pg adapter supports native L2, IP, Cosine or L1")
        self.metric = metric
        self.operator, self.opclass = {
            "euclidean": ("<->", "vector_l2_ops"),
            "ip": ("<#>", "vector_ip_ops"),
            "cosine": ("<=>", "vector_cosine_ops"),
            "l1": ("<+>", "vector_l1_ops"),
        }[metric]
        self.params = dict(index_params)
        allowed = {"R", "L", "alpha", "build_threads", "batch_rows", "compact_runs",
                   "min_run_rows", "compaction", "compact_max_rows", "build_beam_first",
                   "build_neighbor_codes", "build_neighbor_code_bytes", "pq_m", "reclaim_max_rows",
                   "reclaim_max_runs", "reclaim_cooldown", "reclaim_min_dead",
                   "reclaim_min_dead_fraction", "reclaim_backoff_max", "release_input_cache",
                   "vacuum_memtables", "trim_allocator", "reconnect_writer"}
        if set(self.params) - allowed:
            raise ValueError(f"unknown index parameters: {set(self.params) - allowed}")
        self.release_input_cache = self.params.get("release_input_cache", False)
        if not isinstance(self.release_input_cache, bool):
            raise ValueError("release_input_cache must be boolean")
        self.vacuum_memtables = self.params.get("vacuum_memtables", False)
        if not isinstance(self.vacuum_memtables, bool):
            raise ValueError("vacuum_memtables must be boolean")
        self.trim_allocator = self.params.get("trim_allocator", False)
        if not isinstance(self.trim_allocator, bool):
            raise ValueError("trim_allocator must be boolean")
        self.reconnect_writer = self.params.get("reconnect_writer", False)
        if not isinstance(self.reconnect_writer, bool):
            raise ValueError("reconnect_writer must be boolean")
        self.R = int(self.params.get("R", 32))
        self.L = int(self.params.get("L", 100))
        self.alpha = float(self.params.get("alpha", 1.2))
        self.pq_m = self.params.get("pq_m", 0)
        if isinstance(self.pq_m, bool) or not isinstance(self.pq_m, int) or not 0 <= self.pq_m <= 65528:
            raise ValueError("pq_m must be an integer in 0..65528")
        self.build_threads = int(self.params.get("build_threads", 2))
        self.build_beam_first = self.params.get("build_beam_first", False)
        if not isinstance(self.build_beam_first, bool):
            raise ValueError("build_beam_first must be boolean")
        self.build_neighbor_codes = self.params.get("build_neighbor_codes", True)
        if not isinstance(self.build_neighbor_codes, bool):
            raise ValueError("build_neighbor_codes must be boolean")
        self.build_neighbor_code_bytes = self.params.get("build_neighbor_code_bytes", -1)
        if (type(self.build_neighbor_code_bytes) is not int or
                not -1 <= self.build_neighbor_code_bytes <= 2147483647):
            raise ValueError("build_neighbor_code_bytes must be an integer in -1..2147483647")
        self.batch_rows = int(self.params.get("batch_rows", 32768))
        self.compact_runs = int(self.params.get("compact_runs", 8))
        self.compact_max_rows = int(self.params.get("compact_max_rows", 0))
        if not 0 <= self.compact_max_rows <= 2147483647:
            raise ValueError("compact_max_rows must be in 0..2147483647")
        self.compaction = self.params.get("compaction", "full")
        if self.compaction not in ("full", "tiered"):
            raise ValueError("compaction must be full or tiered")
        reclaim_defaults = {"reclaim_max_rows": 0, "reclaim_max_runs": 1,
                            "reclaim_cooldown": 8, "reclaim_min_dead": 1}
        for name, default in reclaim_defaults.items():
            value = self.params.get(name, default)
            upper = {"reclaim_max_runs": 256, "reclaim_cooldown": 1000000}.get(name, 2147483647)
            lower = 0 if name == "reclaim_max_rows" else 1
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in {lower}..{upper}")
            setattr(self, name, value)
        self.reclaim_min_dead_fraction = self.params.get("reclaim_min_dead_fraction", 0)
        self.reclaim_threshold = ReclaimThreshold(
            self.reclaim_min_dead, self.reclaim_min_dead_fraction)
        self.reclaim_backoff_max = self.params.get("reclaim_backoff_max", 0)
        if (type(self.reclaim_backoff_max) is not int or
                not 0 <= self.reclaim_backoff_max <= 1000000 or
                (self.reclaim_backoff_max and self.reclaim_backoff_max < self.reclaim_cooldown)):
            raise ValueError("reclaim_backoff_max must be zero or in reclaim_cooldown..1000000")
        if self.reclaim_max_rows and self.compaction != "tiered":
            raise ValueError("explicit reclamation requires compaction=tiered; no full fallback")
        self.reclaim_planner = (ReclaimPlanner(self.reclaim_max_rows, self.reclaim_max_runs,
                                             self.reclaim_cooldown, self.reclaim_backoff_max)
                                if self.reclaim_max_rows else None)
        self._reclaim_dirty = False
        self.min_run_rows = int(self.params.get("min_run_rows", 32768))
        if not (1 <= self.R <= self.L <= 4096 and 1 <= self.alpha <= 4):
            raise ValueError("invalid R/L/alpha")
        if self.batch_rows <= 0 or self.build_threads <= 0 or self.min_run_rows < 0:
            raise ValueError("invalid batch size, thread count or graphless threshold")
        if self.compact_runs < 2:
            raise ValueError("compact_runs must be >= 2")
        self.dsn = os.environ.get("GSVECTOR_PG_DSN")
        if not self.dsn:
            raise ValueError("GSVECTOR_PG_DSN must explicitly select a dedicated benchmark instance")
        self.name = "gsvector"
        self.query_args = {"search_l": 200, "threads": 4}
        self.keep_schema = os.environ.get("GSVECTOR_KEEP_SCHEMA", "0") == "1"
        self.artifacts = Path(os.environ.get("GSVECTOR_ARTIFACTS", "results/gsvector-artifacts"))
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.schemas = []
        self.conn = None
        self.pool = None
        self._query_local = threading.local()
        self._query_connections = []
        self._query_connections_lock = threading.Lock()
        self.schema = None
        self.events = []
        self.maintenance_totals = {}
        self.search_number = 0
        self.mutated = False
        self.live_count = 0
        self.index_ready = False
        self.vector_oid = None
        self.run_id = uuid.uuid4().hex

    def setup(self, dtype, max_pts, ndims):
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("uint8"), np.dtype("int8")):
            raise ValueError(f"unsupported source dtype: {dtype}")
        if not 1 <= ndims <= 8000:
            raise ValueError("this DiskANN adapter requires a dimension in 1..8000")
        self.dim, self.max_pts = int(ndims), int(max_pts)
        if self.max_pts <= 0:
            raise ValueError("max_pts must be positive")
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS gsvector")
        if self.reclaim_planner is not None and not self.conn.execute(
                "SELECT to_regprocedure('gsvector_reclaim_run(regclass,bigint,integer,integer)')"
                " IS NOT NULL").fetchone()[0]:
            self.conn.close()
            self.conn = None
            raise ValueError("reclamation requires gsvector 0.1.8 API; upgrade explicitly")
        self.vector_oid = self.conn.execute("SELECT 'vector'::regtype::oid").fetchone()[0]
        if self.trim_allocator and not self.conn.execute(
                "SELECT to_regprocedure('public.gs_trim_allocator()') IS NOT NULL").fetchone()[0]:
            self.conn.close()
            self.conn = None
            raise ValueError("trim_allocator requires the explicitly installed allocator helper")
        # A separate private schema per workload, including each query-arg sweep.
        self._new_workload()

    def _new_workload(self):
        self._close_query_pool()
        self.schema = "annbench_" + uuid.uuid4().hex
        self.events = []
        self.maintenance_totals = {}
        self.search_number = 0
        self.mutated = False
        self.live_count = 0
        self.index_ready = False
        self._reclaim_dirty = False
        if self.reclaim_planner is not None:
            self.reclaim_planner.reset()
        self.conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.schemas.append(self.schema)
        self.conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
        self.conn.execute(sql.SQL(
            "CREATE TABLE base (id bigint PRIMARY KEY, embedding vector({}) NOT NULL)"
        ).format(sql.Literal(self.dim)))
        self._configure_write_connection(self.conn, set_search_path=False)
        self._record("setup", dimension=self.dim, max_pts=self.max_pts,
                     metric=self.metric, parameters=self.params)

    def _configure_write_connection(self, conn, set_search_path=True):
        if set_search_path:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS incoming "
                          "(id bigint PRIMARY KEY, embedding vector NOT NULL) ON COMMIT DELETE ROWS")
        for name, value in {
            "gsvector.build_threads": self.build_threads,
            "gsvector.min_run_rows": self.min_run_rows,
            "gsvector.flush_threshold": min(self.batch_rows, 1000000),
            "gsvector.merge_runs": min(self.compact_runs,1024),
        }.items():
            conn.execute("SELECT set_config(%s,%s,false)", (name, str(value)))
        if "compact_max_rows" in self.params:
            conn.execute("SELECT set_config('gsvector.compact_max_rows',%s,false)",
                              (str(self.compact_max_rows),))
        if "build_beam_first" in self.params:
            conn.execute("SELECT set_config('gsvector.build_beam_first',%s,false)",
                              ("on" if self.build_beam_first else "off",))
        if "build_neighbor_codes" in self.params:
            conn.execute("SELECT set_config('gsvector.build_neighbor_codes',%s,false)",
                              ("on" if self.build_neighbor_codes else "off",))
        if "build_neighbor_code_bytes" in self.params:
            registered = conn.execute("SELECT 1 FROM pg_settings WHERE name = %s",
                                      ("gsvector.build_neighbor_code_bytes",)).fetchone()
            if registered is None:
                raise RuntimeError("a matching structural-optimization extension is required "
                                   "for build_neighbor_code_bytes")
            conn.execute("SELECT set_config('gsvector.build_neighbor_code_bytes',%s,false)",
                              (str(self.build_neighbor_code_bytes),))

    def _reconnect_writer(self):
        if not self.reconnect_writer:
            return
        old = self.conn
        if old.info.transaction_status != TransactionStatus.IDLE:
            raise RuntimeError("writer reconnect requires a committed, idle connection")
        started = time.perf_counter()
        new = psycopg.connect(self.dsn, autocommit=True)
        try:
            self._configure_write_connection(new)
        except BaseException:
            new.close()
            raise
        self.conn = new
        old.close()
        # A new writer also needs fresh single-query adapter/dumper settings.
        for name in list(vars(self)):
            if name.startswith("_single_"):
                delattr(self, name)
        for key, value in (("writer_reconnect_s", time.perf_counter() - started),
                           ("writer_reconnect_calls", 1)):
            self.maintenance_totals[key] = self.maintenance_totals.get(key, 0) + value

    def _close_query_pool(self):
        # Workers must stop before their backend handles are closed. Autocommit
        # sessions retain no snapshot across mutation/checkpoint boundaries.
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        for connection in self._query_connections:
            connection.close()
        self._query_connections.clear()
        self._query_local = threading.local()

    @contextmanager
    def _query_connection(self):
        if not self.query_args.get("reuse_connections", False):
            with psycopg.connect(self.dsn, autocommit=True) as connection:
                yield connection
            return
        connection = getattr(self._query_local, "connection", None)
        if connection is None:
            connection = psycopg.connect(self.dsn, autocommit=True)
            self._query_local.connection = connection
            with self._query_connections_lock:
                self._query_connections.append(connection)
        # Do not retry failed queries: that would hide correctness failures and
        # alter measured workload semantics.
        yield connection

    def _record(self, operation, **data):
        event = {"operation": operation, "live_rows": self.live_count, **data}
        if operation in ("insert", "replace", "delete"):
            event["maintenance_totals"] = dict(self.maintenance_totals)
        self.events.append(event)
        with (self.artifacts / f"{self.run_id}-{self.schema}.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def _validate_ids(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("IDs must be a one-dimensional integer array")
        if len(ids) and (ids.min() < 0 or ids.max() > np.iinfo(np.int32).max):
            raise ValueError("IDs exceed the framework's signed-int32 result format")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate IDs in mutation")
        return ids.astype(np.int64, copy=False)

    def _validate_vectors(self, X, ids):
        X = np.asarray(X)
        if X.shape != (len(ids), self.dim):
            raise ValueError(f"expected {(len(ids), self.dim)} vectors, got {X.shape}")
        # Validate all input before the first chunk commits.
        for start in range(0, len(X), self.batch_rows):
            block = np.asarray(X[start:start + self.batch_rows], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError("non-finite vector component")
        return X

    def _copy(self, table, X, ids):
        statement = sql.SQL("COPY {} (id,embedding) FROM STDIN (FORMAT BINARY)").format(
            sql.Identifier(table))
        with self.conn.cursor().copy(statement) as copy:
            for block in binary_copy_blocks(X, ids):
                copy.write(block)

    def _release_input(self, X):
        if not self.release_input_cache:
            return
        started = time.perf_counter()
        receipt = release_consumed_input(X)
        for name in ("eligible_bytes", "advised_bytes"):
            key = "input_cache_" + name
            self.maintenance_totals[key] = self.maintenance_totals.get(key, 0) + receipt[name]
        for key, value in (("input_cache_calls", 1),
                           ("input_cache_errors", int(receipt["error"] is not None)),
                           ("input_cache_s", time.perf_counter() - started)):
            self.maintenance_totals[key] = self.maintenance_totals.get(key, 0) + value

    def _build_index(self):
        if self.pq_m and (self.dim <= 0 or self.dim % self.pq_m):
            raise ValueError("pq_m must divide the indexed dimension")
        # Keep the default SQL compatible with installations predating pq_m.
        pq_option = (sql.SQL(", pq_m={}").format(sql.Literal(self.pq_m))
                     if self.pq_m else sql.SQL(""))
        self.conn.execute(sql.SQL(
            "CREATE INDEX ann_idx ON base USING gsvector (embedding {}) "
            "WITH (engine='diskann', r={}, l={}, alpha={}{})"
        ).format(sql.Identifier(self.opclass), sql.Literal(self.R),
                 sql.Literal(self.L), sql.Literal(self.alpha), pq_option))
        self.index_ready = True
        self.conn.execute("ANALYZE base")
        self._trim_allocator()
        self._reconnect_writer()

    def _trim_allocator(self):
        if not self.trim_allocator:
            return
        started = time.perf_counter()
        released = self.conn.execute("SELECT public.gs_trim_allocator()").fetchone()[0]
        if not isinstance(released, bool):
            raise ValueError("unexpected allocator helper result")
        for key, value in (("allocator_trim_s", time.perf_counter() - started),
                           ("allocator_trim_calls", 1),
                           ("allocator_trim_reported_releases", int(released))):
            self.maintenance_totals[key] = self.maintenance_totals.get(key, 0) + value

    def _runs(self):
        if not self.index_ready:
            return []
        idx = f"{self.schema}.ann_idx"
        oid = self.conn.execute("SELECT %s::regclass::oid", (idx,)).fetchone()[0]
        return self.conn.execute(sql.SQL(
            "SELECT run_id,n,coalesce((stats->>'graph')::int,0) AS graph FROM gsvector.{} "
            "ORDER BY run_id"
        ).format(sql.Identifier(f"runs.{oid}"))).fetchall()

    def _maintenance(self):
        if not self.index_ready:
            return
        # All calls autocommit: run-list publication occurs at transaction commit.
        # With gsvector.worker=off, parallel VACUUM deliberately skips inline
        # drain. Serialize maintenance before reusing the other memtable slot.
        index = f"{self.schema}.ann_idx"
        def execute(stage, statement, args=None):
            started = time.perf_counter()
            try:
                return self.conn.execute(statement, args)
            finally:
                self.maintenance_totals[stage + "_s"] = (
                    self.maintenance_totals.get(stage + "_s", 0.0)
                    + time.perf_counter() - started)
                self.maintenance_totals[stage + "_calls"] = (
                    self.maintenance_totals.get(stage + "_calls", 0) + 1)

        # VACUUM also recycles drained internal memtable storage. Omitting it
        # made a measured streaming prefix slower despite cheaper inserts.
        execute("vacuum", "VACUUM (PARALLEL 0) base")
        # VACUUM may itself seal on row/time thresholds. Drain that slot before
        # explicitly sealing again: the original sequence failed at step 21 of
        # the official runbook. Calls commit individually; no hidden retries.
        execute("drain", "SELECT gsvector_drain(%s::regclass)", (index,))
        execute("flush", "SELECT gsvector_flush(%s::regclass)", (index,))
        execute("drain", "SELECT gsvector_drain(%s::regclass)", (index,))
        execute("vacuum", "VACUUM (PARALLEL 0) base")
        runs = self._runs()
        if len(runs) >= self.compact_runs:
            if self.compaction == "full":
                execute("compact", "SELECT gsvector_compact(%s::regclass)", (index,))
            else:
                # Each successful tier pass reduces run count; never rewrite a
                # lone large run merely because small runs accumulated.
                for _ in range(len(runs)):
                    if not execute("compact", "SELECT gsvector_compact(%s::regclass,false)",
                                   (index,)).fetchone()[0]:
                        break
        # Independent of the merge-run count gate. A successful delete/update
        # latches potential debt; cooldown permits retries after old snapshots
        # release even when no further deletes arrive. Inserts alone do not
        # activate scans on an otherwise clean workload.
        if self.reclaim_planner is not None and self._reclaim_dirty:
            after = self._runs()
            fresh_ids = {int(r[0]) for r in after} - {int(r[0]) for r in runs}
            jobs, oversized = self.reclaim_planner.plan(after, fresh_ids)
            outcomes = []
            for run_id, rows in jobs:
                minimum = self.reclaim_threshold.minimum(rows)
                result = execute(
                    "reclaim", "SELECT gsvector_reclaim_run(%s::regclass,%s,%s,%s)",
                    (index, run_id, rows, minimum)).fetchone()[0]
                if (not isinstance(result, dict) or
                        result.get("outcome") not in {
                            "target_missing", "input_budget", "insufficient_reclaimable",
                            "reclaimed", "retired"}):
                    raise ValueError("unexpected reclamation result")
                receipt = result.get("receipt")
                if type(receipt) is not int or receipt < 0:
                    raise ValueError("invalid reclamation receipt")
                if (result["outcome"] in {"reclaimed", "retired"}) != (receipt > 0):
                    raise ValueError("reclamation receipt/outcome disagreement")
                self.reclaim_planner.record_outcome(run_id, result["outcome"])
                if result["outcome"] == "reclaimed":
                    self.reclaim_planner.defer(receipt)
                outcomes.append({"target_run_id": run_id, "planned_input_rows": rows,
                                 **result, "min_reclaimable_rows": minimum,
                                 "retry_after_rounds": max(
                                     0, self.reclaim_planner.due.get(run_id, 0)
                                     - self.reclaim_planner.tick)})
            self._record("maintenance_reclaim", round=self.reclaim_planner.tick,
                         planned_input_rows=sum(n for _, n in jobs),
                         outcomes=outcomes, oversized_runs=oversized)
        if self.vacuum_memtables:
            # Logically empty is not physically empty. The index's base-table
            # VACUUM cleanup only attempts to truncate the non-active MT slot.
            # A reused active slot can retain a large empty heap which every
            # ANN query must scan for MVCC-visible entries.
            #
            # Use PostgreSQL VACUUM, not SQL TRUNCATE, row-count shortcuts or
            # catalog statistics: old snapshots and in-flight writers must
            # retain their tuples. Each command commits before the next query.
            oid = self.conn.execute("SELECT %s::regclass::oid", (index,)).fetchone()[0]
            for slot in (0, 1):
                execute("vacuum_mt", sql.SQL("VACUUM (PARALLEL 0) {}.{}").format(
                    sql.Identifier("gsvector"), sql.Identifier(f"mt.{oid}.{slot}")))
        self._trim_allocator()
        self._reconnect_writer()

    def insert(self, X, ids):
        start = time.perf_counter()
        ids = self._validate_ids(ids)
        X = self._validate_vectors(X, ids)
        if self.live_count + len(ids) > self.max_pts:
            raise ValueError("insertion exceeds runbook max_pts")
        for offset in range(0, len(ids), self.batch_rows):
            end = min(offset + self.batch_rows, len(ids))
            with self.conn.transaction():
                self._copy("base", X[offset:end], ids[offset:end])
            self.live_count += end - offset
            self._release_input(X[offset:end])
            if self.index_ready:
                self._maintenance()
        if len(ids) and not self.index_ready:
            self._build_index()
        self.mutated = True
        self._record("insert", rows=len(ids), elapsed_s=time.perf_counter() - start, runs=self._runs())

    def delete(self, ids):
        start = time.perf_counter()
        ids = self._validate_ids(ids)
        for offset in range(0, len(ids), self.batch_rows):
            batch = ids[offset:offset + self.batch_rows]
            with self.conn.transaction():
                deleted = self.conn.execute("DELETE FROM base WHERE id = ANY(%s)",
                                            (batch.tolist(),)).rowcount
                if deleted != len(batch):
                    raise ValueError("delete contains inactive IDs")
            self.live_count -= deleted
            if deleted:
                self._reclaim_dirty = True
        self._maintenance()
        self.mutated = True
        self._record("delete", rows=len(ids), elapsed_s=time.perf_counter() - start, runs=self._runs())

    def replace(self, X, ids):
        start = time.perf_counter()
        ids = self._validate_ids(ids)
        X = self._validate_vectors(X, ids)
        for offset in range(0, len(ids), self.batch_rows):
            end = min(offset + self.batch_rows, len(ids))
            with self.conn.transaction():
                self._copy("incoming", X[offset:end], ids[offset:end])
                replaced = self.conn.execute(
                    "UPDATE base b SET embedding=i.embedding FROM incoming i WHERE b.id=i.id"
                ).rowcount
                if replaced != end - offset:
                    raise ValueError("replace contains inactive IDs")
            self._reclaim_dirty = True
            self._release_input(X[offset:end])
            self._maintenance()
        self.mutated = True
        self._record("replace", rows=len(ids), elapsed_s=time.perf_counter() - start, runs=self._runs())

    def set_query_arguments(self, query_args):
        self._set_query_arguments(query_args, reset_workload=True)

    def _set_query_arguments(self, query_args, reset_workload):
        args = dict(query_args)
        if set(args) - {"search_l", "threads", "random_io", "reuse_connections", "fed_run_pool_max",
                       "grouped_rerank", "federate_k", "query_order", "query_dispatch",
                       "query_batch_size", "max_k", "pq_rerank_frontier", "refine_sq8", "refine_expanded_pool", "refine_u8_simd"}:
            raise ValueError(f"unknown query arguments: {args}")
        if args.get("query_order", "input") not in ("input", "locality"):
            raise ValueError("query_order must be input or locality")
        if args.get("query_dispatch", "contiguous") not in ("contiguous", "interleaved"):
            raise ValueError("query_dispatch must be contiguous or interleaved")
        batch_size = args.get("query_batch_size", 1)
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("query_batch_size must be an integer in 1..32")
        if "max_k" in args and (type(args["max_k"]) is not int or not 1 <= args["max_k"] <= 32768):
            raise ValueError("max_k must be an integer in 1..32768")
        if "refine_expanded_pool" in args:
            p = args["refine_expanded_pool"]
            if type(p) is not int or not 0 <= p <= 32768:
                raise ValueError("refine_expanded_pool must be an integer in 0..32768")
            if p and not args.get("refine_sq8", False):
                raise ValueError("refine_expanded_pool requires refine_sq8")
        for name in ("random_io", "reuse_connections", "grouped_rerank", "pq_rerank_frontier", "refine_sq8", "refine_u8_simd"):
            if name in args and not isinstance(args[name], bool):
                raise ValueError(f"{name} must be boolean")
        if args.get("refine_sq8", False) and args.get("pq_rerank_frontier", False):
            raise ValueError("refine_sq8 and pq_rerank_frontier are mutually exclusive")
        search_l = int(args.get("search_l", 200))
        threads = int(args.get("threads", 4))
        if search_l < 1 or threads < 1:
            raise ValueError("invalid search_l or threads")
        if "fed_run_pool_max" in args and not 0 <= int(args["fed_run_pool_max"]) <= 1024:
            raise ValueError("fed_run_pool_max must be in 0..1024")
        if "federate_k" in args:
            value = args["federate_k"]
            if type(value) not in (int, float) or not 1 <= value <= 8 or not math.isfinite(value):
                raise ValueError("federate_k must be a finite number in 1..8")
        # Official runner replays the runbook for each argument group.
        if self.mutated and reset_workload:
            self._new_workload()
        elif (threads != self.query_args["threads"] or
              args.get("reuse_connections", False) != self.query_args.get("reuse_connections", False)):
            self._close_query_pool()
        self.query_args = {"search_l": search_l, "threads": threads}
        for name in ("random_io", "reuse_connections", "grouped_rerank", "pq_rerank_frontier", "refine_sq8", "refine_u8_simd"):
            if name in args:
                self.query_args[name] = args[name]
        if "fed_run_pool_max" in args:
            self.query_args["fed_run_pool_max"] = int(args["fed_run_pool_max"])
        if "federate_k" in args:
            self.query_args["federate_k"] = args["federate_k"]
        for name in ("query_order", "query_dispatch", "query_batch_size", "max_k", "refine_expanded_pool"):
            if name in args:
                self.query_args[name] = args[name]

    def _apply_pq_rerank_frontier(self, conn):
        explicit = "pq_rerank_frontier" in self.query_args
        reset = ((conn is self.conn and getattr(self, "_single_pq_frontier_overridden", False)) or
                 (conn is not self.conn and self.query_args.get("reuse_connections", False)))
        if not explicit and not reset:
            return
        # Old PostgreSQL accepts arbitrary custom-GUC placeholders. Require
        # the registered boolean before claiming that this experiment ran.
        setting = conn.execute("SELECT vartype FROM pg_settings WHERE name = %s",
                               ("gsvector.pq_rerank_frontier",)).fetchone()
        if setting is None or setting[0] != "bool":
            if explicit:
                raise RuntimeError("pq_rerank_frontier requires a matching gsvector extension")
            return  # An omitted feature must still work with an old server.
        enabled = explicit and self.query_args["pq_rerank_frontier"]
        conn.execute("SELECT set_config('gsvector.pq_rerank_frontier',%s,false)",
                     ("on" if enabled else "off",))
        if conn is self.conn:
            self._single_pq_frontier_overridden = explicit

    def _apply_refine_sq8(self, conn):
        explicit = "refine_sq8" in self.query_args
        reset = ((conn is self.conn and getattr(self, "_single_refine_sq8_overridden", False)) or
                 (conn is not self.conn and self.query_args.get("reuse_connections", False)))
        if not explicit and not reset:
            return
        setting = conn.execute("SELECT vartype FROM pg_settings WHERE name = %s",
                               ("gsvector.refine_sq8",)).fetchone()
        if setting is None or setting[0] != "bool":
            if explicit:
                raise RuntimeError("refine_sq8 requires a matching gsvector extension")
            return
        enabled = explicit and self.query_args["refine_sq8"]
        conn.execute("SELECT set_config('gsvector.refine_sq8',%s,false)",
                     ("on" if enabled else "off",))
        if conn is self.conn:
            self._single_refine_sq8_overridden = explicit

    def _apply_refine_u8_simd(self, conn):
        explicit = "refine_u8_simd" in self.query_args
        reset = ((conn is self.conn and getattr(self, "_single_refine_u8_simd_overridden", False)) or
                 (conn is not self.conn and self.query_args.get("reuse_connections", False)))
        if not explicit and not reset:
            return
        setting = conn.execute("SELECT vartype FROM pg_settings WHERE name = %s",
                               ("gsvector.refine_u8_simd",)).fetchone()
        if setting is None or setting[0] != "bool":
            if explicit:
                raise RuntimeError("refine_u8_simd requires a matching gsvector extension")
            return
        enabled = explicit and self.query_args["refine_u8_simd"]
        conn.execute("SELECT set_config('gsvector.refine_u8_simd',%s,false)",
                     ("on" if enabled else "off",))
        if conn is self.conn:
            self._single_refine_u8_simd_overridden = explicit

    def _apply_refine_expanded_pool(self, conn):
        explicit = "refine_expanded_pool" in self.query_args
        reset = ((conn is self.conn and getattr(self, "_single_expanded_pool_overridden", False)) or
                 (conn is not self.conn and self.query_args.get("reuse_connections", False)))
        if not explicit and not reset:
            return
        setting = conn.execute("SELECT vartype FROM pg_settings WHERE name = %s",
                               ("gsvector.refine_expanded_pool",)).fetchone()
        if setting is None or setting[0] != "integer":
            if explicit:
                raise RuntimeError("refine_expanded_pool requires a matching gsvector extension")
            return
        value = self.query_args["refine_expanded_pool"] if explicit else 0
        conn.execute("SELECT set_config('gsvector.refine_expanded_pool',%s,false)", (str(value),))
        if conn is self.conn:
            self._single_expanded_pool_overridden = explicit

    def _apply_random_io(self, conn):
        self._apply_pq_rerank_frontier(conn)
        self._apply_refine_sq8(conn)
        self._apply_refine_u8_simd(conn)
        self._apply_refine_expanded_pool(conn)
        # Omitted option inherits the connection default (including PGOPTIONS).
        # Persistent single-query sessions must undo a previous explicit value.
        if "random_io" in self.query_args:
            conn.execute("SELECT set_config('gsvector.query_random_io',%s,false)",
                         ("on" if self.query_args["random_io"] else "off",))
        elif conn is self.conn and getattr(self,"_single_random_io_overridden",False):
            conn.execute("RESET gsvector.query_random_io")
        elif conn is not self.conn and self.query_args.get("reuse_connections", False):
            conn.execute("RESET gsvector.query_random_io")
        if conn is self.conn:
            self._single_random_io_overridden = "random_io" in self.query_args
        if "fed_run_pool_max" in self.query_args:
            conn.execute("SELECT set_config('gsvector.fed_run_pool_max',%s,false)",
                         (str(self.query_args["fed_run_pool_max"]),))
        elif ((conn is self.conn and getattr(self, "_single_pool_max_overridden", False)) or
              (conn is not self.conn and self.query_args.get("reuse_connections", False))):
            conn.execute("RESET gsvector.fed_run_pool_max")
        if conn is self.conn:
            self._single_pool_max_overridden = "fed_run_pool_max" in self.query_args
        if "grouped_rerank" in self.query_args:
            conn.execute("SELECT set_config('gsvector.grouped_rerank',%s,false)",
                         ("on" if self.query_args["grouped_rerank"] else "off",))
        elif ((conn is self.conn and getattr(self, "_single_grouped_overridden", False)) or
              (conn is not self.conn and self.query_args.get("reuse_connections", False))):
            conn.execute("RESET gsvector.grouped_rerank")
        if conn is self.conn:
            self._single_grouped_overridden = "grouped_rerank" in self.query_args
        if "federate_k" in self.query_args:
            conn.execute("SELECT set_config('gsvector.federate_k',%s,false)",
                         (str(self.query_args["federate_k"]),))
        elif ((conn is self.conn and getattr(self, "_single_federate_overridden", False)) or
              (conn is not self.conn and self.query_args.get("reuse_connections", False))):
            conn.execute("RESET gsvector.federate_k")
        if conn is self.conn:
            self._single_federate_overridden = "federate_k" in self.query_args
        if "max_k" in self.query_args:
            conn.execute("SELECT set_config('gsvector.max_k',%s,false)",
                         (str(self.query_args["max_k"]),))
        elif ((conn is self.conn and getattr(self, "_single_max_k_overridden", False)) or
              (conn is not self.conn and self.query_args.get("reuse_connections", False))):
            conn.execute("RESET gsvector.max_k")
        if conn is self.conn:
            self._single_max_k_overridden = "max_k" in self.query_args

    def _query_chunk(self, queries, k):
        with self._query_connection() as conn:
            class VectorDumper(Dumper):
                oid = self.vector_oid
                format = Format.BINARY

                def dump(self, obj):
                    return obj

            conn.adapters.register_dumper(VectorBytes, VectorDumper)
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
            conn.execute("SET enable_seqscan=off")
            conn.execute("SET enable_bitmapscan=off")
            conn.execute("SET max_parallel_workers_per_gather=0")
            conn.execute("SET gsvector.query_beam_first=on")
            conn.execute("SELECT set_config('gsvector.search_l',%s,false)",
                         (str(self.query_args["search_l"]),))
            conn.execute("SELECT set_config('gsvector.scan_target_k',%s,false)", (str(k),))
            self._apply_random_io(conn)
            statement = f"SELECT id FROM base ORDER BY embedding {self.operator} %s LIMIT %s"
            if len(queries):
                plan = conn.execute("EXPLAIN (FORMAT JSON) " + statement,
                                    (encode_vector(queries[0]), k)).fetchone()[0]
                if not has_index_plan(plan):
                    raise RuntimeError(f"ANN index plan required: {plan}")
            result = np.full((len(queries), k), -1, dtype=np.int32)
            expected = min(k, self.live_count)
            batch_size = self.query_args.get("query_batch_size", 1)
            if batch_size > 1:
                # Experimental SRF explicitly installed in the owned test DB.
                # Uses a statement snapshot and the normal scan correctness
                # path; collection and exact rerank are scheduled run-major.
                for start in range(0, len(queries), batch_size):
                    batch = queries[start:start + batch_size]
                    slots = ",".join(["%s"] * len(batch))
                    statement_batch = (
                        "SELECT b.query_no,t.id FROM public.gs_ann_batch("
                        f"%s::regclass,ARRAY[{slots}]::vector[],%s) b "
                        "JOIN base t ON t.ctid=b.tid "
                        "ORDER BY b.query_no,b.dist,b.tid")
                    rows = conn.execute(statement_batch, (
                        f"{self.schema}.ann_idx", *[encode_vector(q) for q in batch], k)).fetchall()
                    grouped = [[] for _ in batch]
                    for ordinal, identifier in rows:
                        if not 0 <= ordinal < len(batch):
                            raise RuntimeError("batch ANN returned an invalid query ordinal")
                        grouped[ordinal].append(identifier)
                    for i, values in enumerate(grouped):
                        if len(values) < expected:
                            raise QueryCardinalityError(len(values), expected, start + len(batch))
                        if len(values) != expected or len(set(values)) != len(values):
                            raise RuntimeError("batch ANN cardinality/uniqueness mismatch")
                        result[start + i, :len(values)] = values
                return result
            for i, query in enumerate(queries):
                rows = conn.execute(statement, (encode_vector(query), k)).fetchall()
                values = [row[0] for row in rows]
                if len(values) < expected:
                    raise QueryCardinalityError(len(values), expected, i + 1)
                if len(values) != expected or len(set(values)) != len(values):
                    raise RuntimeError(f"ANN returned {len(values)} results, expected {expected}")
                result[i, :len(values)] = values
            return result

    def query(self, X, k):
        start = time.perf_counter()
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.dim or not np.isfinite(X).all():
            raise ValueError("invalid query vectors")
        if k <= 0 or k > self.query_args["search_l"]:
            raise ValueError("require 0 < k <= search_l")
        if not self.index_ready:
            self._build_index()
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=self.query_args["threads"])
        if (self.query_args.get("query_order", "input") == "input" and
                self.query_args.get("query_dispatch", "contiguous") == "contiguous"):
            # Preserve the original path, including its zero-copy query slices.
            chunks = [chunk for chunk in np.array_split(X, self.query_args["threads"]) if len(chunk)]
            futures = [self.pool.submit(self._query_chunk, chunk, k) for chunk in chunks]
            self.res = (np.concatenate(gather_query_chunks(futures)) if chunks else
                        np.empty((0, k), dtype=np.int32))
        else:
            from .query_order import query_partitions
            partitions = query_partitions(
                X, self.query_args["threads"], self.query_args.get("query_order", "input"),
                self.query_args.get("query_dispatch", "contiguous"))
            futures = [self.pool.submit(self._query_chunk, X[indices], k)
                       for indices in partitions]
            self.res = np.empty((len(X), k), dtype=np.int32)
            for indices, result in zip(partitions, gather_query_chunks(futures)):
                self.res[indices] = result
        self.search_number += 1
        evidence = {}
        if os.environ.get("GSVECTOR_SAVE_CHECKPOINTS") == "1":
            directory = self.artifacts / f"{self.run_id}-{self.schema}-checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"search-{self.search_number:04d}.npy"
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("xb") as stream:
                np.save(stream, self.res, allow_pickle=False)
            temporary.replace(path)
            evidence["checkpoint_file"] = str(path.resolve())
        elapsed = time.perf_counter() - start  # diagnostic persistence is included
        self._record("search", checkpoint=self.search_number, queries=len(X), k=k, elapsed_s=elapsed,
                     query_args=self.query_args, runs=self._runs(), **evidence)

    def get_memory_usage(self):
        # Sum PSS, not RSS (PostgreSQL shared pages must not be counted per backend).
        pids = {os.getpid()}
        pidfile = os.environ.get("GSVECTOR_PG_PIDFILE")
        if pidfile and Path(pidfile).exists():
            postmaster = psutil.Process(int(Path(pidfile).read_text().splitlines()[0]))
            pids.add(postmaster.pid)
            pids.update(child.pid for child in postmaster.children(recursive=True))
        total = 0
        for pid in pids:
            try:
                memory = psutil.Process(pid).memory_full_info()
                total += getattr(memory, "pss", memory.rss)
            except psutil.NoSuchProcess:
                pass
        return total / 1024

    def get_additional(self):
        events_path = self.artifacts / f"{self.run_id}-{self.schema}.jsonl"
        # Static query sweeps append to the working log after earlier HDF5
        # files have been stored. Hash an immutable snapshot per checkpoint.
        snapshot = events_path.with_name(events_path.stem + f"-snapshot-{self.search_number}.jsonl")
        snapshot.write_bytes(events_path.read_bytes())
        return {
            "pg_schema": self.schema,
            "pg_live_rows": self.live_count,
            # A full 1280-step runbook exceeds HDF5's small-attribute limit.
            # Keep detailed telemetry in the durable sidecar, not a giant attr.
            "pg_events_file": str(snapshot.resolve()),
            "pg_events_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "pg_event_count": len(self.events),
            "pg_pss_kb_at_end": self.get_memory_usage(),
            "memory_accounting": (
                "Python and local PG process-tree PSS, not peak memory"
                if os.environ.get("GSVECTOR_PG_PIDFILE") else
                "Python client only; external PostgreSQL memory is NOT measured"),
            "index_size_scope": "framework index_size is setup-time delta, not peak streaming memory",
        }

    def done(self):
        self._close_query_pool()
        if self.conn is not None:
            try:
                if not self.keep_schema:
                    for schema in self.schemas:
                        self.conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
            finally:
                self.conn.close()

    def __str__(self):
        return f"gsvector(R={self.R},L={self.L},queries={self.query_args})"
