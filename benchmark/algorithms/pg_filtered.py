"""L2 ANN + GIN exact fallback for conjunction label filters.

The WHERE clause is evaluated inside PostgreSQL for every ANN candidate.
Insufficient candidates trigger bounded expansion, then an explicitly counted
exact fallback. This is not an inline graph-label filter and does not claim one.
The opt-in bounded strategy reuses a small eligible ID set and performs at most
one ANN search per query. It is not resumable graph traversal.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from psycopg import sql

from benchmark.algorithms.pg_exact import PGFilterExact
from benchmark.algorithms.pg_dense import PGDenseANN
from neurips23.streaming.gsvector.module import encode_vector, has_index_plan


class PGFilteredANN(PGFilterExact):
    mode = "gsvector-filtered-ann-with-gin-exact-fallback"

    def __init__(self,metric,index_params):
        params=dict(index_params)
        allowed={"batch_rows","R","L","alpha","build_threads","min_run_rows",
                 "build_beam_first","build_neighbor_codes","build_neighbor_code_bytes",
                 "build_raw_cache_mb","build_gather_threads","pq_m","prepare_refine"}
        if set(params)-allowed:
            raise ValueError("unsupported filtered ANN build parameters")
        super().__init__(metric,{"batch_rows":params.get("batch_rows",4096)})
        self.R=int(params.get("R",32))
        self.L=int(params.get("L",100))
        self.alpha=float(params.get("alpha",1.2))
        self.build_threads=int(params.get("build_threads",4))
        self.min_run_rows=int(params.get("min_run_rows",32768))
        bool_options=("build_beam_first","build_neighbor_codes")
        self.build_options={name:params[name] for name in bool_options
                            if name in params}
        if any(not isinstance(value,bool) for value in self.build_options.values()):
            raise ValueError("build strategy/layout options must be boolean")
        for name,low,high in (("build_neighbor_code_bytes",-1,2147483647),
                              ("build_raw_cache_mb",0,65536),
                              ("build_gather_threads",1,64)):
            if name in params:
                value=params[name]
                if type(value) is not int or not low<=value<=high:
                    raise ValueError(f"invalid {name}")
                self.build_options[name]=value
        self.pq_m=params.get("pq_m",0)
        if type(self.pq_m) is not int or not 0<=self.pq_m<=65528:
            raise ValueError("invalid pq_m")
        self.prepare_refine=params.get("prepare_refine",False)
        if type(self.prepare_refine) is not bool:
            raise ValueError("prepare_refine must be boolean")
        self.refine_prepared=False
        if (not 1<=self.R<=self.L<=4096 or not 1<=self.alpha<=4 or
                self.build_threads<1 or self.min_run_rows<0):
            raise ValueError("invalid filtered ANN build parameters")
        self.query_number=0
        self.stats={}
        self.set_query_arguments({})

    def fit(self,dataset):
        start=time.perf_counter()
        super().fit(dataset)
        loaded=time.perf_counter()
        self.live_count=self.nb
        self.refine_prepared=False
        if self.pq_m and self.dim % self.pq_m:
            raise ValueError("pq_m must divide the indexed dimension")
        self.conn.execute("SELECT set_config('gsvector.build_threads',%s,false)",
                          (str(self.build_threads),))
        self.conn.execute("SELECT set_config('gsvector.min_run_rows',%s,false)",
                          (str(self.min_run_rows),))
        for name,value in self.build_options.items():
            expected="bool" if type(value) is bool else "integer"
            registered=self.conn.execute("SELECT vartype FROM pg_settings WHERE name=%s",
                                         ("gsvector."+name,)).fetchone()
            if not registered or registered[0]!=expected:
                raise RuntimeError(f"matching extension required for {name}")
            setting=("on" if value else "off") if type(value) is bool else str(value)
            self.conn.execute("SELECT set_config(%s,%s,false)",("gsvector."+name,setting))
        pq_option=sql.SQL(",pq_m={}").format(sql.Literal(self.pq_m)) if self.pq_m else sql.SQL("")
        self.conn.execute(sql.SQL(
            "CREATE INDEX ann_idx ON base USING gsvector(embedding vector_l2_ops) "
            "WITH(engine='diskann',r={},l={},alpha={}{})"
        ).format(sql.Literal(self.R),sql.Literal(self.L),sql.Literal(self.alpha),pq_option))
        self.build_times={"load_and_gin_s":loaded-start,
                          "ann_index_s":time.perf_counter()-loaded}
        if self.prepare_refine:
            prepared_at=time.perf_counter()
            self.conn.execute("CREATE FUNCTION prepare_refine(regclass) RETURNS jsonb "
                              "AS '$libdir/gsvector','gsvector_prepare_refine' LANGUAGE C STRICT")
            self.refine_preparation=self.conn.execute(
                "SELECT prepare_refine('ann_idx'::regclass)").fetchone()[0]
            self.refine_prepared=True
            self.build_times["prepare_refine_s"]=time.perf_counter()-prepared_at
        self._record("ann_fit",rows=self.nb,parameters={"R":self.R,"L":self.L,"alpha":self.alpha,
                                                     "pq_m":self.pq_m,"prepare_refine":self.prepare_refine,
                                                     **self.build_options},
                     build_times=self.build_times)

    def set_query_arguments(self,args):
        allowed={"threads","search_l","initial_candidates","max_candidates","exact_cutoff",
                 "filter_strategy","ann_candidates","refine_sq8"}
        if set(args)-allowed:
            raise ValueError("unsupported filtered ANN query parameters")
        self.threads=int(args.get("threads",4))
        self.search_l=int(args.get("search_l",200))
        self.initial_candidates=int(args.get("initial_candidates",256))
        self.max_candidates=int(args.get("max_candidates",8192))
        self.exact_cutoff=int(args.get("exact_cutoff",2048))
        self.filter_strategy=args.get("filter_strategy","legacy")
        if self.filter_strategy not in ("legacy","bounded"):
            raise ValueError("filter_strategy must be legacy or bounded")
        self.ann_candidates=args.get("ann_candidates",self.initial_candidates)
        if type(self.ann_candidates) is not int or not 1<=self.ann_candidates<=self.max_candidates:
            raise ValueError("ann_candidates must be an integer within max_candidates")
        self.refine_sq8=args.get("refine_sq8",False)
        self.refine_explicit="refine_sq8" in args
        if type(self.refine_sq8) is not bool:
            raise ValueError("refine_sq8 must be boolean")
        if (self.threads<1 or not 1<=self.search_l<=32768 or self.exact_cutoff<0 or
                not 1<=self.initial_candidates<=self.max_candidates<=32768):
            raise ValueError("invalid filtered ANN query parameters")

    @staticmethod
    def _new_filter_stats():
        return {"small_exact_queries":0,"ann_queries":0,"fallback_exact_queries":0,
                "ann_attempts":0,"max_candidates_used":0,"ann_plan_verified":False,
                "eligible_probe_queries":0,"eligible_probe_rows":0,"reused_eligible_ids":0}

    def _effective_cutoff(self,k):
        if self.filter_strategy=="bounded" or not self.exact_cutoff:
            return self.exact_cutoff
        return max(self.exact_cutoff,min(self.nb,(2*k*self.nb+self.max_candidates-1)//self.max_candidates))

    def _bounded_query(self,conn,tags,query,k,cutoff,stats):
        """Caller holds one read-only repeatable-read transaction for all steps."""
        conn.execute("SET LOCAL enable_bitmapscan=on; SET LOCAL enable_indexscan=off")
        ids=None
        if tags and cutoff:
            ids=[row[0] for row in conn.execute(
                "SELECT id FROM base WHERE tags @> %s::integer[] LIMIT %s",(tags,cutoff+1))]
            stats["eligible_probe_queries"]+=1
            stats["eligible_probe_rows"]+=len(ids)
        if ids is not None and len(ids)<=cutoff:
            stats["small_exact_queries"]+=1
            stats["reused_eligible_ids"]+=len(ids)
            return self._score_eligible_ids(conn,ids,tags,query,k)
        return self._single_ann_query(conn,tags,query,k,stats)

    @staticmethod
    def _score_eligible_ids(conn,ids,tags,query,k):
        if not ids:
            return []
        # Materialization keeps the label predicate outside the PK lookup,
        # so it cannot be pushed back into a second GIN search. PostgreSQL
        # work_mem controls this bounded intermediate and can spill it.
        # Keep label/MVCC checks; retained IDs are not trusted answers.
        return [row[0] for row in conn.execute(
            "WITH eligible AS MATERIALIZED (SELECT id,embedding,tags FROM base "
            "WHERE id = ANY(%s::bigint[])) SELECT id FROM eligible WHERE tags @> %s::integer[] "
            "ORDER BY embedding <-> %s,id LIMIT %s",(ids,tags,query,k))]

    def _single_ann_query(self,conn,tags,query,k,stats):
        budget=min(self.nb,max(k,self.ann_candidates))
        conn.execute("SET LOCAL enable_indexscan=on; SET LOCAL enable_bitmapscan=off")
        conn.execute("SELECT set_config('gsvector.max_k',%s,true),"
                     "set_config('gsvector.search_l',%s,true)",
                     (str(budget),str(max(self.search_l,budget))))
        ann_sql="SELECT id FROM base WHERE tags @> %s::integer[] ORDER BY embedding <-> %s LIMIT %s"
        params=(tags,query,k)
        if not stats["ann_plan_verified"]:
            plan=conn.execute("EXPLAIN (FORMAT JSON) "+ann_sql,params).fetchone()[0]
            if not has_index_plan(plan,"ann_idx"):
                raise RuntimeError("filtered ANN did not select its native index")
            stats["ann_plan_verified"]=True
        values=[row[0] for row in conn.execute(ann_sql,params)]
        stats["ann_attempts"]+=1
        stats["max_candidates_used"]=max(stats["max_candidates_used"],budget)
        if len(values)==k:
            stats["ann_queries"]+=1
            return values
        conn.execute("SET LOCAL enable_indexscan=off; SET LOCAL enable_bitmapscan=on")
        stats["fallback_exact_queries"]+=1
        return [row[0] for row in conn.execute(
            "SELECT id FROM base WHERE tags @> %s::integer[] ORDER BY embedding <-> %s,id LIMIT %s",
            params)]

    def filtered_query(self,X,filters,k):
        if self.refine_sq8 and not self.refine_prepared:
            raise RuntimeError("refine_sq8 requires prepare_refine=true during fit, before query timing")
        start=time.perf_counter()
        if (X.shape!=(filters.shape[0],self.dim) or not np.isfinite(X).all() or
                not 1<=k<=self.max_candidates):
            raise ValueError("invalid filtered queries")
        filters=filters.tocsr()
        # At large N, predicates rarer than k/max_candidates are unlikely to
        # fill k slots even at the ANN ceiling. Prefer their bounded GIN set.
        # An explicit cutoff of zero disables this rule for diagnostics.
        cutoff=self._effective_cutoff(k)
        ann_sql="SELECT id FROM base WHERE tags @> %s::integer[] ORDER BY embedding <-> %s LIMIT %s"
        exact_sql="SELECT id FROM base WHERE tags @> %s::integer[] ORDER BY embedding <-> %s,id LIMIT %s"

        def batch(indices):
            result=np.full((len(indices),k),-1,dtype=np.int32)
            stats=self._new_filter_stats()
            with self._connect() as conn:
                conn.execute("SET gsvector.federate_k=1; SET gsvector.iterate_max=0; "
                             "SET gsvector.query_beam_first=on")
                if self.refine_explicit:
                    registered=conn.execute("SELECT vartype FROM pg_settings WHERE name=%s",
                                            ("gsvector.refine_sq8",)).fetchone()
                    if not registered or registered[0]!="bool":
                        raise RuntimeError("matching extension required for refine_sq8")
                    conn.execute("SELECT set_config('gsvector.refine_sq8',%s,false)",
                                 ("on" if self.refine_sq8 else "off",))
                for row,qi in enumerate(indices):
                    tags=filters.indices[filters.indptr[qi]:filters.indptr[qi+1]].tolist()
                    params=(tags,encode_vector(X[qi]),k)
                    # All expansion attempts and exact fallback share one MVCC
                    # snapshot even if another client updates labels/vectors.
                    with conn.transaction():
                        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                        if self.filter_strategy=="bounded":
                            values=self._bounded_query(conn,tags,params[1],k,cutoff,stats)
                            if len(set(values))!=len(values) or any(v<0 or v>=self.nb for v in values):
                                raise RuntimeError("invalid filtered neighbor IDs")
                            result[row,:len(values)]=values
                            continue
                        conn.execute("SET LOCAL enable_bitmapscan=on; SET LOCAL enable_indexscan=off")
                        eligible=self.nb if not tags else conn.execute("""
                            SELECT count(*) FROM (
                              SELECT 1 FROM base WHERE tags @> %s::integer[] LIMIT %s
                            ) candidates
                        """,(tags,cutoff+1)).fetchone()[0]
                        if eligible<=cutoff:
                            values=[r[0] for r in conn.execute(exact_sql,params)]
                            stats["small_exact_queries"]+=1
                        else:
                            conn.execute("SET LOCAL enable_indexscan=on; SET LOCAL enable_bitmapscan=off")
                            budget=min(self.nb,max(k,self.initial_candidates))
                            limit=min(self.nb,max(k,self.max_candidates))
                            values=[]
                            while True:
                                conn.execute("SELECT set_config('gsvector.max_k',%s,true),"
                                             "set_config('gsvector.search_l',%s,true)",
                                             (str(budget),str(max(self.search_l,budget))))
                                if not stats["ann_plan_verified"]:
                                    plan=conn.execute("EXPLAIN (FORMAT JSON) "+ann_sql,params).fetchone()[0]
                                    if not has_index_plan(plan,"ann_idx"):
                                        raise RuntimeError("filtered ANN did not select its native index")
                                    stats["ann_plan_verified"]=True
                                values=[r[0] for r in conn.execute(ann_sql,params)]
                                stats["ann_attempts"]+=1
                                stats["max_candidates_used"]=max(stats["max_candidates_used"],budget)
                                if len(values)==k:
                                    stats["ann_queries"]+=1
                                    break
                                if budget>=limit:
                                    conn.execute("SET LOCAL enable_indexscan=off; SET LOCAL enable_bitmapscan=on")
                                    values=[r[0] for r in conn.execute(exact_sql,params)]
                                    stats["fallback_exact_queries"]+=1
                                    break
                                budget=min(limit,budget*2)
                        if len(set(values))!=len(values) or any(v<0 or v>=self.nb for v in values):
                            raise RuntimeError("invalid filtered neighbor IDs")
                        result[row,:len(values)]=values
            return result,stats

        chunks=[x for x in np.array_split(np.arange(len(X)),self.threads) if len(x)]
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            outputs=list(pool.map(batch,chunks))
        self.res=np.concatenate([x[0] for x in outputs]) if outputs else np.empty((0,k),dtype=np.int32)
        keys=("small_exact_queries","ann_queries","fallback_exact_queries","ann_attempts",
              "eligible_probe_queries","eligible_probe_rows","reused_eligible_ids")
        self.stats={key:sum(x[1][key] for x in outputs) for key in keys}
        self.stats["max_candidates_used"]=max((x[1]["max_candidates_used"] for x in outputs),default=0)
        self.stats["ann_plan_verified"]=any(x[1]["ann_plan_verified"] for x in outputs)
        self.stats["effective_exact_cutoff"]=cutoff
        self.stats["filter_strategy"]=self.filter_strategy
        self.stats["ann_candidates"]=self.ann_candidates if self.filter_strategy=="bounded" else None
        self.stats["refine_sq8"]=self.refine_sq8
        self.query_number+=1
        self._record("filtered_query",queries=len(X),k=k,elapsed_s=time.perf_counter()-start,
                     mode=self.mode,stats=self.stats)

    def get_additional(self):
        snapshot=self.log.with_name(self.log.stem+f"-snapshot-{self.query_number}.jsonl")
        snapshot.write_bytes(self.log.read_bytes())
        storage=PGDenseANN.get_storage_usage(self)
        gin=self.conn.execute("SELECT pg_relation_size(%s::regclass)",(f"{self.schema}.tags_idx",)).fetchone()[0]
        storage["gin_bytes"]=gin
        storage["index_total_bytes"]+=gin
        storage["index_bytes_per_vector"]=storage["index_total_bytes"]/max(1,self.nb)
        return {**super().get_additional(),"native_ann":True,
                "pg_events_file":str(snapshot.resolve()),
                "pg_events_sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "pg_filter_stats_json":json.dumps(self.stats),
                "pg_storage_json":json.dumps(storage),
                "filtering_method":("eligible ID reuse + single ANN attempt + explicit exact fallback"
                                    if self.filter_strategy=="bounded" else
                                    "SQL post-filter + bounded expansion + explicit exact fallback")}
