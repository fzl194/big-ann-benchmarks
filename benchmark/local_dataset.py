"""Explicit local dense/CSR subsets; never override official dataset identities."""
import json
from pathlib import Path

import numpy as np

from benchmark.dataset_io import xbin_mmap, sanitize, read_sparse_matrix, knn_result_read


class LocalDenseDataset:
    def __init__(self, name, spec):
        self.name = name
        self.dtype = spec.get("dtype", "float32")
        if self.dtype not in ("float32", "uint8", "int8"):
            raise ValueError("unsupported local dense dtype")
        self.ds_fn = str(Path(spec["base"]).resolve())
        self.qs_fn = str(Path(spec["queries"]).resolve())
        base = xbin_mmap(self.ds_fn, self.dtype)
        queries = xbin_mmap(self.qs_fn, self.dtype)
        if base.shape[1] != queries.shape[1]:
            raise ValueError("base/query dimension mismatch")
        self.d = base.shape[1]
        self.nb = int(spec.get("rows", len(base)))
        self.nq = int(spec.get("query_count", len(queries)))
        if not 0 < self.nb <= len(base) or not 0 < self.nq <= len(queries):
            raise ValueError("local subset exceeds file bounds")
        self.metric = spec.get("metric", "euclidean")
        self.basedir = str(Path(spec["groundtruth_dir"]).resolve())
        self.spec = dict(spec)
        self.query_ids = np.asarray(spec.get("query_ids", list(range(self.nq))), dtype=np.int64)
        if (self.query_ids.shape != (self.nq,) or len(np.unique(self.query_ids)) != self.nq or
                np.any(self.query_ids < 0) or np.any(self.query_ids >= len(queries))):
            raise ValueError("invalid explicit query IDs")

    def prepare(self, skip_data=False):
        # Files were validated above; never download or modify the source files.
        pass

    def get_dataset_fn(self):
        return self.ds_fn

    def get_data_in_range(self, start, end):
        if not 0 <= start <= end <= self.nb:
            raise ValueError("range outside local subset")
        return xbin_mmap(self.ds_fn, self.dtype, maxn=self.nb)[start:end]

    def get_dataset(self):
        return self.get_data_in_range(0, self.nb)

    def get_dataset_iterator(self, bs=512, split=(1, 0)):
        shards, rank = split
        start, end = self.nb * rank // shards, self.nb * (rank + 1) // shards
        for offset in range(start, end, bs):
            yield self.get_data_in_range(offset, min(offset + bs, end))

    def get_queries(self):
        return sanitize(xbin_mmap(self.qs_fn, self.dtype)[self.query_ids])

    def get_dataset_metadata(self):
        return read_sparse_matrix(self.spec["base_metadata"], do_mmap=True)[:self.nb]

    def get_queries_metadata(self):
        return read_sparse_matrix(self.spec["query_metadata"], do_mmap=True)[self.query_ids]

    def get_groundtruth(self, k=None):
        ids, distances = knn_result_read(str(Path(self.basedir) / "groundtruth.gt100"))
        return (ids[:, :k], distances[:, :k]) if k else (ids, distances)

    def get_private_queries(self):
        raise NotImplementedError("local manifest has no private queries")

    def distance(self):
        return self.metric

    def search_type(self):
        return "knn_filtered" if "base_metadata" in self.spec else "knn"

    def data_type(self):
        return "dense"

    def default_count(self):
        return 10

    def short_name(self):
        return self.name

    def __str__(self):
        return f"LocalDenseDataset({self.name}, base={self.nb}, queries={self.nq}, dim={self.d})"


class LocalCSRDataset:
    def __init__(self, name, spec):
        self.name = name
        self.ds_fn = str(Path(spec["base"]).resolve())
        self.qs_fn = str(Path(spec["queries"]).resolve())
        self.basedir = str(Path(spec["groundtruth_dir"]).resolve())
        base = read_sparse_matrix(self.ds_fn, do_mmap=True)
        queries = read_sparse_matrix(self.qs_fn, do_mmap=True)
        self.nb = int(spec.get("rows", base.shape[0]))
        self.nq = int(spec.get("query_count", queries.shape[0]))
        self.d = base.shape[1]
        if not 0 < self.nb <= base.shape[0] or not 0 < self.nq <= queries.shape[0]:
            raise ValueError("sparse subset out of bounds")
        if queries.shape[1] > self.d:
            raise ValueError("query terms exceed base vocabulary")

    def prepare(self, skip_data=False):
        pass

    def get_dataset_iterator(self, bs=512, split=(1, 0)):
        if split != (1, 0):
            raise ValueError("sparse local dataset does not support sharding")
        base = read_sparse_matrix(self.ds_fn, do_mmap=True)
        for start in range(0, self.nb, bs):
            yield base[start:min(start + bs, self.nb)]

    def get_dataset(self):
        return next(self.get_dataset_iterator(bs=self.nb))

    def get_queries(self):
        queries = read_sparse_matrix(self.qs_fn, do_mmap=True)[:self.nq]
        if queries.shape[1] != self.d:
            from scipy.sparse import csr_matrix
            queries = csr_matrix((queries.data, queries.indices, queries.indptr),
                                 shape=(self.nq, self.d))
        return queries

    def get_private_queries(self):
        raise NotImplementedError("no private queries in local CSR manifest")

    def get_groundtruth(self, k=None):
        ids, distances = knn_result_read(str(Path(self.basedir) / "groundtruth.gt100"))
        return (ids[:, :k], distances[:, :k]) if k else (ids, distances)

    def distance(self):
        return "ip"

    def search_type(self):
        return "knn"

    def data_type(self):
        return "sparse"

    def short_name(self):
        return self.name

    def default_count(self):
        return 10


def register_local_datasets(registry, path):
    specs = json.loads(Path(path).read_text())
    for name, spec in specs.items():
        if name in registry:
            raise ValueError(f"cannot override existing dataset: {name}")
        cls = LocalCSRDataset if spec.get("format") == "csr" else LocalDenseDataset
        registry[name] = lambda name=name, spec=spec, cls=cls: cls(name, spec)
