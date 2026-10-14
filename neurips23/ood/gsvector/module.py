import time

import numpy as np

from benchmark.datasets import DATASETS
from neurips23.streaming.gsvector.module import Gsvector


class GsvectorOOD(Gsvector):
    """Static native-IP DiskANN build. No normalization or metric conversion."""

    def track(self):
        return "ood"

    def fit(self, dataset):
        started = time.perf_counter()
        ds = DATASETS[dataset]()
        if ds.distance() != self.metric or ds.data_type() != "dense":
            raise ValueError("dataset metric/type does not match native dense adapter")
        self.setup(ds.dtype, ds.nb, ds.d)
        offset = 0
        for block in ds.get_dataset_iterator(bs=self.batch_rows):
            ids = np.arange(offset, offset + len(block), dtype=np.int64)
            self._validate_vectors(block, ids)
            with self.conn.transaction():
                self._copy("base", block, ids)
            offset += len(block)
            self.live_count = offset
        if offset != ds.nb:
            raise ValueError("dataset iterator returned the wrong row count")
        self._build_index()
        self.mutated = True
        self._record("fit", rows=offset, elapsed_s=time.perf_counter() - started, runs=self._runs())

    def set_query_arguments(self, query_args):
        self._set_query_arguments(query_args, reset_workload=False)

    def get_additional(self):
        return {**super().get_additional(), "native_ann": True,
                "execution_mode": f"gsvector-native-{self.metric}-ann",
                "index_size_scope": "before/after static fit PSS delta, not peak memory"}
