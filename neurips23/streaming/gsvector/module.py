"""Streaming L2/IP adapter; connect only to an explicitly supplied benchmark PG.

No dataset preloading, anticipated inserts, padding of input dimensions, or
sequential-scan fallback. The first insertion builds the initial index; later
mutations use the index's memtable/drain paths and optional full compaction.
"""
import json
import hashlib
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
from psycopg.pq import Format
import psutil

from neurips23.streaming.base import BaseStreamingANN


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
                   "min_run_rows", "compaction", "compact_max_rows"}
        if set(self.params) - allowed:
            raise ValueError(f"unknown index parameters: {set(self.params) - allowed}")
        self.R = int(self.params.get("R", 32))
        self.L = int(self.params.get("L", 100))
        self.alpha = float(self.params.get("alpha", 1.2))
        self.build_threads = int(self.params.get("build_threads", 2))
        self.batch_rows = int(self.params.get("batch_rows", 32768))
        self.compact_runs = int(self.params.get("compact_runs", 8))
        self.compact_max_rows = int(self.params.get("compact_max_rows", 0))
        if not 0 <= self.compact_max_rows <= 2147483647:
            raise ValueError("compact_max_rows must be in 0..2147483647")
        self.compaction = self.params.get("compaction", "full")
        if self.compaction not in ("full", "tiered"):
            raise ValueError("compaction must be full or tiered")
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
        self.vector_oid = self.conn.execute("SELECT 'vector'::regtype::oid").fetchone()[0]
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
        self.conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.schemas.append(self.schema)
        self.conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
        self.conn.execute(sql.SQL(
            "CREATE TABLE base (id bigint PRIMARY KEY, embedding vector({}) NOT NULL)"
        ).format(sql.Literal(self.dim)))
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS incoming "
                          "(id bigint PRIMARY KEY, embedding vector NOT NULL) ON COMMIT DELETE ROWS")
        for name, value in {
            "gsvector.build_threads": self.build_threads,
            "gsvector.min_run_rows": self.min_run_rows,
            "gsvector.flush_threshold": min(self.batch_rows, 1000000),
            "gsvector.merge_runs": min(self.compact_runs,1024),
        }.items():
            self.conn.execute("SELECT set_config(%s,%s,false)", (name, str(value)))
        if "compact_max_rows" in self.params:
            self.conn.execute("SELECT set_config('gsvector.compact_max_rows',%s,false)",
                              (str(self.compact_max_rows),))
        self._record("setup", dimension=self.dim, max_pts=self.max_pts,
                     metric=self.metric, parameters=self.params)

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

    def _build_index(self):
        self.conn.execute(sql.SQL(
            "CREATE INDEX ann_idx ON base USING gsvector (embedding {}) "
            "WITH (engine='diskann', r={}, l={}, alpha={})"
        ).format(sql.Identifier(self.opclass), sql.Literal(self.R),
                 sql.Literal(self.L), sql.Literal(self.alpha)))
        self.index_ready = True
        self.conn.execute("ANALYZE base")

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
            self._maintenance()
        self.mutated = True
        self._record("replace", rows=len(ids), elapsed_s=time.perf_counter() - start, runs=self._runs())

    def set_query_arguments(self, query_args):
        self._set_query_arguments(query_args, reset_workload=True)

    def _set_query_arguments(self, query_args, reset_workload):
        args = dict(query_args)
        if set(args) - {"search_l", "threads", "random_io", "reuse_connections", "fed_run_pool_max"}:
            raise ValueError(f"unknown query arguments: {args}")
        for name in ("random_io", "reuse_connections"):
            if name in args and not isinstance(args[name], bool):
                raise ValueError(f"{name} must be boolean")
        search_l = int(args.get("search_l", 200))
        threads = int(args.get("threads", 4))
        if search_l < 1 or threads < 1:
            raise ValueError("invalid search_l or threads")
        if "fed_run_pool_max" in args and not 0 <= int(args["fed_run_pool_max"]) <= 1024:
            raise ValueError("fed_run_pool_max must be in 0..1024")
        # Official runner replays the runbook for each argument group.
        if self.mutated and reset_workload:
            self._new_workload()
        elif (threads != self.query_args["threads"] or
              args.get("reuse_connections", False) != self.query_args.get("reuse_connections", False)):
            self._close_query_pool()
        self.query_args = {"search_l": search_l, "threads": threads}
        for name in ("random_io", "reuse_connections"):
            if name in args:
                self.query_args[name] = args[name]
        if "fed_run_pool_max" in args:
            self.query_args["fed_run_pool_max"] = int(args["fed_run_pool_max"])

    def _apply_random_io(self, conn):
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
            for i, query in enumerate(queries):
                rows = conn.execute(statement, (encode_vector(query), k)).fetchall()
                values = [row[0] for row in rows]
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
        chunks = [chunk for chunk in np.array_split(X, self.query_args["threads"]) if len(chunk)]
        futures = [self.pool.submit(self._query_chunk, chunk, k) for chunk in chunks]
        self.res = (np.concatenate([future.result() for future in futures]) if chunks else
                    np.empty((0, k), dtype=np.int32))
        elapsed = time.perf_counter() - start
        self.search_number += 1
        self._record("search", checkpoint=self.search_number, queries=len(X), k=k, elapsed_s=elapsed,
                     query_args=self.query_args, runs=self._runs())

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
