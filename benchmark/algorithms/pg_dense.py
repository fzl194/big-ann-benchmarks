"""PostgreSQL dense ANN adapter for Big ANN and array-based ANN/VIBE runners.

The database is external and explicitly selected by GSVECTOR_PG_DSN. HDF5
datasets may be passed directly to fit(): only one bounded slice is materialized.
No upstream metric or result computation is replaced by this adapter.
"""
import time
import os

import numpy as np
import psycopg
from psycopg import sql
from psycopg.adapt import Dumper
from psycopg.pq import Format

from benchmark.datasets import DATASETS
from neurips23.streaming.gsvector.module import Gsvector, VectorBytes, encode_vector, has_index_plan


def canonical_metric(metric):
    return {"angular": "cosine", "normalized": "cosine", "manhattan":"l1"}.get(metric, metric)


class PGDenseANN(Gsvector):
    def track(self):
        return "none"

    def load_index(self,dataset):
        schema=os.environ.get("GSVECTOR_REUSE_SCHEMA")
        if not schema:
            return False
        ds=DATASETS[dataset]()
        if canonical_metric(ds.distance())!=self.metric:
            raise ValueError("reused index metric does not match dataset")
        self.attach(schema)
        if self.dim!=ds.d or self.live_count!=ds.nb:
            raise ValueError("reused index dimensions/row count do not match dataset")
        return True

    def attach(self, schema):
        """Explicit read-only reuse of an existing benchmark index for A/B tests.

        Never assumes ownership of or drops the attached schema. Validate its
        opclass/dimension and report the actual row count before querying.
        """
        if self.conn is not None:
            raise RuntimeError("attach requires a fresh adapter")
        self.keep_schema = True
        self.conn = psycopg.connect(self.dsn,autocommit=True)
        self.schema = schema
        self.conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema)))
        metadata = self.conn.execute("""
            SELECT a.atttypmod,c.opcname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attname='embedding'
            JOIN pg_opclass c ON c.oid=i.indclass[0]
            WHERE i.indexrelid=%s::regclass AND i.indrelid=%s::regclass AND i.indisvalid
        """,(f"{schema}.ann_idx",f"{schema}.base")).fetchone()
        if not metadata or metadata[1] != self.opclass:
            raise ValueError("attached index metric/schema mismatch")
        self.dim = metadata[0]
        self.live_count = self.conn.execute("SELECT count(*) FROM base").fetchone()[0]
        self.max_pts = self.live_count
        self.vector_oid = self.conn.execute("SELECT 'vector'::regtype::oid").fetchone()[0]
        self.index_ready = self.mutated = True
        self.reused_index = True
        options=self.conn.execute("SELECT reloptions FROM pg_class WHERE oid=%s::regclass",
                                  (f"{schema}.ann_idx",)).fetchone()[0] or []
        options=dict(x.split("=",1) for x in options)
        self.R=int(options.get("r",self.R))
        self.L=int(options.get("l",self.L))
        self.alpha=float(options.get("alpha",self.alpha))
        self.pq_m=int(options.get("pq_m",0))
        self._record("attach",dimension=self.dim,metric=self.metric,rows=self.live_count,
                     pq_m=self.pq_m)

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
        next_progress = 1_000_000
        for block in blocks:
            ids = np.arange(offset, offset + len(block), dtype=np.int64)
            self._validate_vectors(block, ids)
            with self.conn.transaction():
                self._copy("base", block, ids)
            offset += len(block)
            self.live_count = offset
            if offset >= next_progress:
                self._record("load_progress", rows=offset,
                             elapsed_s=time.perf_counter()-started)
                next_progress = (offset // 1_000_000 + 1) * 1_000_000
        if offset != rows:
            raise ValueError("dataset iterator returned the wrong row count")
        load_done = time.perf_counter()
        self._record("load", rows=offset, elapsed_s=load_done-started)
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
        if single:
            # Array-based CPU runners issue one query at a time. Opening a new
            # backend per call discards the immutable run/codebook cache and
            # measures connection setup rather than a persistent SQL service.
            start = time.perf_counter()
            if X.shape != (self.dim,) or k <= 0 or k > self.query_args["search_l"]:
                raise ValueError("invalid single query dimension/k")
            encoded = encode_vector(X)
            statement = f"SELECT id FROM base ORDER BY embedding {self.operator} %s LIMIT %s"
            state = (self.schema,self.query_args["search_l"],k,self.query_args.get("random_io"),
                     self.query_args.get("fed_run_pool_max"), self.query_args.get("grouped_rerank"),
                     self.query_args.get("federate_k"))
            if getattr(self,"_single_state",None) != state:
                class VectorDumper(Dumper):
                    oid = self.vector_oid
                    format = Format.BINARY
                    def dump(self,obj):
                        return obj
                self.conn.adapters.register_dumper(VectorBytes,VectorDumper)
                self.conn.execute("SET enable_seqscan=off; SET enable_bitmapscan=off; "
                                  "SET max_parallel_workers_per_gather=0; SET gsvector.query_beam_first=on")
                self.conn.execute("SELECT set_config('gsvector.search_l',%s,false)",
                                  (str(self.query_args["search_l"]),))
                self.conn.execute("SELECT set_config('gsvector.scan_target_k',%s,false)", (str(k),))
                self._apply_random_io(self.conn)
                plan=self.conn.execute("EXPLAIN (FORMAT JSON) "+statement,(encoded,k)).fetchone()[0]
                if not has_index_plan(plan):
                    raise RuntimeError(f"ANN index plan required: {plan}")
                self._single_state=state
                self._single_runs=self._runs()
            values=[row[0] for row in self.conn.execute(statement,(encoded,k)).fetchall()]
            if len(values)!=min(k,self.live_count) or len(set(values))!=len(values):
                raise RuntimeError("single-query ANN result cardinality/uniqueness mismatch")
            self.res=np.full((1,k),-1,dtype=np.int32)
            self.res[0,:len(values)]=values
            self.search_number+=1
            self._record("search",checkpoint=self.search_number,queries=1,k=k,
                         elapsed_s=time.perf_counter()-start,query_args=self.query_args,
                         runs=self._single_runs,connection_mode="persistent-single-client")
            return self.res[0]
        super().query(X, k)
        return self.res

    def batch_query(self, X, k):
        self.query(X, k)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {**super().get_additional(), "native_ann": True,
                "reused_index":getattr(self,"reused_index",False),
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
