"""Opt-in dynamic DiskANN PG plugin adapter.

Requires the separately installed experimental gsvector_streaming SQL backend,
an explicitly configured local worker, and one common memory/CPU resource scope.
This is a batch SQL query path; it is not a timing of individual ORDER BY scans.
The existing Gsvector adapter and every other track remain unchanged.
"""
import os
from pathlib import Path
import sys

import numpy as np

from neurips23.streaming.base import BaseStreamingANN


class GsvectorDynamic(BaseStreamingANN):
    def __init__(self, metric, index_params):
        if metric != "euclidean":
            raise ValueError("dynamic Streaming backend currently supports L2 only")
        allowed = {"R", "L", "build_threads", "create_index"}
        if set(index_params) - allowed:
            raise ValueError("unknown dynamic Streaming index parameter")
        self.params = dict(index_params)
        self.backend = None
        self.dimensions = self.capacity = None
        self.search_l = 300
        self.started = False

    def setup(self, dtype, max_pts, ndims):
        if np.dtype(dtype) != np.dtype("float32"):
            raise ValueError("dynamic Streaming backend requires FP32 SQL input")
        runtime = Path(os.environ["GSVECTOR_STREAMING_RUNTIME"]).resolve(strict=True)
        if not (runtime / "streaming_pg_dynamic.py").is_file():
            raise ValueError("missing explicitly configured plugin runtime")
        if str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        from streaming_pg_dynamic import PGDynamicDiskANN
        self.factory = PGDynamicDiskANN
        self.dimensions, self.capacity = ndims, max_pts
        self._new_workload()

    def _new_workload(self):
        if self.backend is not None:
            self.backend.done()
        self.backend = self.factory(
            self.dimensions, self.capacity, self.params.get("build_threads", 32),
            os.environ["GSVECTOR_STREAMING_SOCKET"],
            self.params.get("R", 32), self.params.get("L", 100),
            self.search_l, self.params.get("create_index", True))
        self.started = False

    def set_query_arguments(self, query_args):
        if set(query_args) - {"search_l"}:
            raise ValueError("unknown dynamic Streaming query parameter")
        value = query_args.get("search_l", 300)
        if not isinstance(value, int) or not 1 <= value <= 100000:
            raise ValueError("invalid dynamic search_l")
        self.search_l = value
        # Upstream can replay the runbook with multiple query configurations.
        # Never replay inserts into a previous configuration's populated table.
        if self.started:
            self._new_workload()
        elif self.backend is not None:
            self.backend.search_l = value

    def insert(self, vectors, ids):
        self.started = True
        self.backend.insert(vectors, ids)

    def delete(self, ids):
        self.started = True
        self.backend.delete(ids)

    def replace(self, vectors, ids):
        self.started = True
        with self.backend.conn.transaction():
            self.backend.delete(ids)
            self.backend.insert(vectors, ids)

    def query(self, queries, k):
        self.started = True
        self.backend.query(queries, k)

    def get_results(self):
        return self.backend.get_results()

    def get_additional(self):
        return self.backend.get_additional()

    def done(self):
        if self.backend is not None:
            self.backend.done()
            self.backend = None

    def __str__(self):
        return f"GsvectorDynamic(batch-SQL,R={self.params.get('R',32)},Ls={self.search_l})"
