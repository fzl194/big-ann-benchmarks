"""PostgreSQL dense ANN adapter for Big ANN and array-based ANN/VIBE runners.

The database is external and explicitly selected by GSVECTOR_PG_DSN. HDF5
datasets may be passed directly to fit(): only one bounded slice is materialized.
No upstream metric or result computation is replaced by this adapter.
"""
import time

import numpy as np

from benchmark.datasets import DATASETS
from neurips23.streaming.gsvector.module import Gsvector


def canonical_metric(metric):
    return {"angular": "cosine", "normalized": "cosine"}.get(metric, metric)


class PGDenseANN(Gsvector):
    def track(self):
        return "none"

    def fit(self, dataset):
        started = time.perf_counter()
        if isinstance(dataset, str):
            ds = DATASETS[dataset]()
            if canonical_metric(ds.distance()) != self.metric or ds.data_type() != "dense":
                raise ValueError("dataset metric/type does not match dense adapter")
            dtype, rows, dim = ds.dtype, ds.nb, ds.d
            blocks = ds.get_dataset_iterator(bs=self.batch_rows)
        else:
            if len(dataset.shape) != 2:
                raise ValueError("expected a two-dimensional dense dataset")
            rows, dim = dataset.shape
            dtype = dataset.dtype
            blocks = (dataset[i:i+self.batch_rows] for i in range(0, rows, self.batch_rows))
        self.setup(dtype, rows, dim)
        offset = 0
        for block in blocks:
            ids = np.arange(offset, offset + len(block), dtype=np.int64)
            self._validate_vectors(block, ids)
            with self.conn.transaction():
                self._copy("base", block, ids)
            offset += len(block)
            self.live_count = offset
        if offset != rows:
            raise ValueError("dataset iterator returned the wrong row count")
        load_done = time.perf_counter()
        self._build_index()
        self.build_times = {"load_s": load_done-started,
                            "index_s": time.perf_counter()-load_done}
        self.mutated = True
        self._record("fit", rows=offset, elapsed_s=time.perf_counter()-started, runs=self._runs())

    def set_query_arguments(self, query_args, threads=None):
        if not isinstance(query_args, dict):
            query_args = {"search_l": int(query_args),
                          "threads": self.query_args["threads"] if threads is None else int(threads)}
        self._set_query_arguments(query_args, reset_workload=False)

    def query(self, X, k):
        X = np.asarray(X)
        single = X.ndim == 1
        super().query(X[None, :] if single else X, k)
        return self.res[0] if single else self.res

    def batch_query(self, X, k):
        self.query(X, k)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {**super().get_additional(), "native_ann": True,
                "execution_mode": f"gsvector-native-{self.metric}-ann",
                "index_size_scope": "before/after fit PSS delta, not peak memory or disk size"}

    def get_storage_usage(self):
        """Server-side disk accounting; works with both local and remote PG."""
        index = f"{self.schema}.ann_idx"
        oid = self.conn.execute("SELECT %s::regclass::oid", (index,)).fetchone()[0]
        dboid = self.conn.execute(
            "SELECT oid FROM pg_database WHERE datname=current_database()").fetchone()[0]
        datadir = self.conn.execute("SHOW data_directory").fetchone()[0]
        pending = [f"{datadir}/gsvector/{dboid}/{oid}"]
        external = files = 0
        while pending:
            directory = pending.pop()
            for name, size, isdir in self.conn.execute("""
                SELECT name,(pg_stat_file(%s||'/'||name,true)).size,
                       (pg_stat_file(%s||'/'||name,true)).isdir
                FROM pg_ls_dir(%s,true,false) AS name
            """, (directory,directory,directory)):
                if isdir:
                    pending.append(directory+"/"+name)
                elif size is not None:
                    external += size
                    files += 1
        names = [f"idxstate.{oid}", f"runs.{oid}", f"mt.{oid}.0", f"mt.{oid}.1"]
        internal = self.conn.execute("""
            SELECT coalesce(sum(pg_total_relation_size(c.oid)),0)
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='gsvector' AND c.relname=ANY(%s) AND c.relkind='r'
        """, (names,)).fetchone()[0]
        pg_index = self.conn.execute("SELECT pg_relation_size(%s::regclass)", (index,)).fetchone()[0]
        base = self.conn.execute(
            "SELECT pg_total_relation_size(%s::regclass)",(f"{self.schema}.base",)).fetchone()[0]
        return {"external_index_bytes": external, "external_files": files,
                "internal_relations_bytes": int(internal), "pg_index_bytes": pg_index,
                "index_total_bytes": external+int(internal)+pg_index,
                "base_relation_bytes": base,
                "index_bytes_per_vector": (external+int(internal)+pg_index)/max(1,self.live_count)}


class PGArrayANN(PGDenseANN):
    """Positional constructor convenient for ann-benchmarks/VIBE YAML sweeps."""
    def __init__(self, metric, R=32, L=100, build_threads=4):
        super().__init__(metric, {"R": int(R), "L": int(L), "build_threads": int(build_threads)})
