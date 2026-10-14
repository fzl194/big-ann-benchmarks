"""Explicit PostgreSQL exact baselines, not gsvector ANN implementations.

Filter: GIN label containment followed by exact L2 ranking.
Sparse: term-indexed postings and exact nonnegative sparse inner products.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psycopg
from psycopg import sql
from psycopg.adapt import Dumper
from psycopg.pq import Format

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS
from neurips23.streaming.gsvector.module import Gsvector, VectorBytes, encode_vector, has_index_plan

COPY_HEADER = b"PGCOPY\n\xff\r\n\0" + struct.pack("!ii", 0, 0)
COPY_END = struct.pack("!h", -1)


class PGExactBase(BaseANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.params = dict(index_params)
        if set(self.params) - {"batch_rows"}:
            raise ValueError("exact baseline accepts only batch_rows")
        self.batch_rows = int(self.params.get("batch_rows", 4096))
        if self.batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self.dsn = os.environ.get("GSVECTOR_PG_DSN")
        if not self.dsn:
            raise ValueError("GSVECTOR_PG_DSN must select a dedicated benchmark instance")
        self.keep = os.environ.get("GSVECTOR_KEEP_SCHEMA", "0") == "1"
        self.artifacts = Path(os.environ.get("GSVECTOR_ARTIFACTS", "results/pg-exact"))
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.schema = "pgbench_" + uuid.uuid4().hex
        self.log = self.artifacts / (self.schema + ".jsonl")
        self.conn = None
        self.schema_created = False
        self.vector_oid = None
        self.threads = 4
        self.name = self.mode

    def _setup(self, extension=False):
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        if extension:
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS gsvector")
            self.vector_oid = self.conn.execute("SELECT 'vector'::regtype::oid").fetchone()[0]
        self.conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.schema_created = True
        self.conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(self.schema)))

    def _connect(self):
        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(self.schema)))
        conn.execute("SET max_parallel_workers_per_gather=0")
        conn.execute("SET enable_seqscan=off")
        if self.vector_oid:
            class VectorDumper(Dumper):
                oid = self.vector_oid
                format = Format.BINARY

                def dump(self, obj):
                    return obj

            conn.adapters.register_dumper(VectorBytes, VectorDumper)
        return conn

    def _record(self, operation, **data):
        with self.log.open("a") as stream:
            stream.write(json.dumps({"operation": operation, **data}) + "\n")

    def load_index(self, dataset):
        return False

    def set_query_arguments(self, args):
        if set(args) - {"threads"}:
            raise ValueError("exact baseline accepts only threads")
        self.threads = int(args.get("threads", 4))
        if self.threads <= 0:
            raise ValueError("threads must be positive")

    def get_memory_usage(self):
        return Gsvector.get_memory_usage(self)

    def get_additional(self):
        return {"execution_mode": self.mode, "native_ann": False, "pg_schema": self.schema,
                "pg_events_file": str(self.log.resolve()),
                "pg_events_sha256": hashlib.sha256(self.log.read_bytes()).hexdigest(),
                "threads": self.threads,
                "memory_accounting": (
                    "Python and local PG process-tree PSS, not peak memory"
                    if os.environ.get("GSVECTOR_PG_PIDFILE") else
                    "Python client only; external PostgreSQL memory is NOT measured")}

    def done(self):
        if self.conn is not None:
            try:
                if not self.keep and self.schema_created:
                    self.conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema)))
            finally:
                self.conn.close()


class PGFilterExact(PGExactBase):
    mode = "postgresql-gin-filtered-exact-l2"

    def __init__(self, metric, index_params):
        if metric != "euclidean":
            raise ValueError("YFCC exact filter baseline requires L2")
        super().__init__(metric, index_params)

    def track(self):
        return "filter"

    def fit(self, dataset):
        started = time.perf_counter()
        ds = DATASETS[dataset]()
        if ds.search_type() != "knn_filtered" or ds.distance() != self.metric:
            raise ValueError("expected an L2 filtered dataset")
        self._setup(extension=True)
        self.nb, self.dim = ds.nb, ds.d
        self.conn.execute(sql.SQL(
            "CREATE TABLE base(id bigint PRIMARY KEY,embedding vector({}) NOT NULL,tags integer[] NOT NULL)"
        ).format(sql.Literal(self.dim)))
        metadata = ds.get_dataset_metadata().tocsr()
        if metadata.shape[0] < self.nb:
            raise ValueError("metadata row count mismatch")
        offset = 0
        for block in ds.get_dataset_iterator(bs=self.batch_rows):
            if not np.isfinite(block).all():
                raise ValueError("non-finite base vector")
            records = bytearray()
            for j, vector in enumerate(block):
                row = offset + j
                tags = metadata.indices[metadata.indptr[row]:metadata.indptr[row + 1]]
                if len(tags) and int(tags.max()) > np.iinfo(np.int32).max:
                    raise ValueError("label ID out of range")
                # PostgreSQL array binary format: ndim, hasnull, elem OID, dimensions.
                array = bytearray(struct.pack("!iii", 1 if len(tags) else 0, 0, 23))
                if len(tags):
                    array.extend(struct.pack("!ii", len(tags), 1))
                    pairs = np.empty(len(tags), dtype=[("len", ">i4"), ("tag", ">i4")])
                    pairs["len"], pairs["tag"] = 4, tags
                    array.extend(pairs.tobytes())
                encoded = encode_vector(vector)
                records.extend(struct.pack("!hiq", 3, 8, row))
                records.extend(struct.pack("!i", len(encoded)))
                records.extend(encoded)
                records.extend(struct.pack("!i", len(array)))
                records.extend(array)
            with self.conn.cursor().copy("COPY base FROM STDIN (FORMAT BINARY)") as copy:
                copy.write(COPY_HEADER)
                copy.write(records)
                copy.write(COPY_END)
            offset += len(block)
        if offset != self.nb:
            raise ValueError("wrong dataset row count")
        self.conn.execute("CREATE INDEX tags_idx ON base USING gin(tags)")
        self.conn.execute("VACUUM (ANALYZE, PARALLEL 0) base")
        self._record("fit", rows=offset, dim=self.dim, elapsed_s=time.perf_counter() - started)

    def filtered_query(self, X, filters, k):
        started = time.perf_counter()
        if X.shape != (filters.shape[0], self.dim) or not np.isfinite(X).all() or k < 1:
            raise ValueError("invalid filtered queries")
        filters = filters.tocsr()
        statement = ("SELECT id FROM base WHERE tags @> %s::integer[] "
                     "ORDER BY embedding <-> %s, id LIMIT %s")

        def batch(indices):
            result = np.full((len(indices), k), -1, dtype=np.int32)
            plan = None
            with self._connect() as conn:
                for row, i in enumerate(indices):
                    tags = filters.indices[filters.indptr[i]:filters.indptr[i + 1]].tolist()
                    params = (tags, encode_vector(X[i]), k)
                    if plan is None and tags:
                        plan = conn.execute("EXPLAIN (FORMAT JSON) " + statement, params).fetchone()[0]
                        if not has_index_plan(plan, "tags_idx"):
                            raise RuntimeError(f"expected GIN tag lookup: {plan}")
                    values = [r[0] for r in conn.execute(statement, params).fetchall()]
                    result[row, :len(values)] = values
            return result, plan

        chunks = [chunk for chunk in np.array_split(np.arange(len(X)), self.threads) if len(chunk)]
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            output = list(pool.map(batch, chunks))
        self.res = np.concatenate([r[0] for r in output]) if output else np.empty((0, k), dtype=np.int32)
        self._record("filtered_query", queries=len(X), k=k,
                     elapsed_s=time.perf_counter() - started, plans=[r[1] for r in output])


class PGSparseExact(PGExactBase):
    mode = "postgresql-postings-exact-nonnegative-ip"

    def __init__(self, metric, index_params):
        if metric != "ip":
            raise ValueError("sparse baseline requires inner product")
        super().__init__(metric, index_params)

    def track(self):
        return "sparse"

    @staticmethod
    def validate_csr(matrix):
        block = matrix.tocsr(copy=True)
        block.sum_duplicates()
        block.eliminate_zeros()
        if not np.isfinite(block.data).all() or np.any(block.data < 0):
            raise ValueError("SPLADE baseline requires finite nonnegative sparse weights")
        return block

    def fit(self, dataset):
        started = time.perf_counter()
        ds = DATASETS[dataset]()
        if ds.data_type() != "sparse" or ds.distance() != "ip":
            raise ValueError("expected sparse IP dataset")
        self._setup()
        self.nb = ds.nb
        self.conn.execute("CREATE TABLE postings(term integer NOT NULL,doc bigint NOT NULL,weight real NOT NULL)")
        dtype = np.dtype([("nf", ">i2"), ("ts", ">i4"), ("term", ">i4"),
                          ("ds", ">i4"), ("doc", ">i8"), ("ws", ">i4"), ("weight", ">f4")])
        offset = nnz = 0
        for data in ds.get_dataset_iterator(bs=self.batch_rows):
            block = self.validate_csr(data)
            rows = np.empty(block.nnz, dtype=dtype)
            rows["nf"], rows["ts"], rows["ds"], rows["ws"] = 3, 4, 8, 4
            rows["term"] = block.indices
            rows["doc"] = np.repeat(np.arange(offset, offset + block.shape[0]), np.diff(block.indptr))
            rows["weight"] = block.data
            with self.conn.cursor().copy("COPY postings FROM STDIN (FORMAT BINARY)") as copy:
                copy.write(COPY_HEADER)
                copy.write(rows.tobytes())
                copy.write(COPY_END)
            offset += block.shape[0]
            nnz += block.nnz
        if offset != self.nb:
            raise ValueError("wrong sparse row count")
        self.conn.execute("CREATE INDEX term_idx ON postings(term) INCLUDE(doc,weight)")
        self.conn.execute("VACUUM (ANALYZE, PARALLEL 0) postings")
        self._record("fit", rows=offset, nonzero_entries=nnz, elapsed_s=time.perf_counter() - started)

    def query(self, X, k):
        started = time.perf_counter()
        if not 0 < k <= self.nb:
            raise ValueError("require 0 < k <= base row count")
        X = self.validate_csr(X)
        statement = """
            SELECT p.doc,sum(p.weight::float8*q.weight::float8) AS score
            FROM unnest(%s::integer[],%s::real[]) AS q(term,weight)
            CROSS JOIN LATERAL (
              SELECT doc,weight FROM postings WHERE term=q.term OFFSET 0
            ) p
            GROUP BY p.doc ORDER BY score DESC,p.doc LIMIT %s
        """

        def batch(indices):
            result = np.empty((len(indices), k), dtype=np.int32)
            plan = None
            with self._connect() as conn:
                for row, i in enumerate(indices):
                    begin, end = X.indptr[i:i + 2]
                    params = (X.indices[begin:end].tolist(), X.data[begin:end].tolist(), k)
                    if plan is None and begin != end:
                        plan = conn.execute("EXPLAIN (FORMAT JSON) " + statement, params).fetchone()[0]
                        if not has_index_plan(plan, "term_idx"):
                            raise RuntimeError(f"expected term-index lookup: {plan}")
                    hits = [r[0] for r in conn.execute(statement, params).fetchall()]
                    # With nonnegative nonzero weights, all matching scores > 0.
                    # Fill missing top-k slots with zero-score docs, tie-broken by ID.
                    seen = set(hits)
                    for candidate in range(min(self.nb, 2 * k)):
                        if len(hits) == k:
                            break
                        if candidate not in seen:
                            hits.append(candidate)
                    if len(hits) != k:
                        raise RuntimeError("sparse top-k fill failed")
                    result[row] = hits
            return result, plan

        chunks = [chunk for chunk in np.array_split(np.arange(X.shape[0]), self.threads) if len(chunk)]
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            output = list(pool.map(batch, chunks))
        self.res = np.concatenate([r[0] for r in output]) if output else np.empty((0, k), dtype=np.int32)
        self._record("sparse_query", queries=X.shape[0], k=k,
                     elapsed_s=time.perf_counter() - started, plans=[r[1] for r in output])
