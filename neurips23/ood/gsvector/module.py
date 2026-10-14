from benchmark.algorithms.pg_dense import PGDenseANN


class GsvectorOOD(PGDenseANN):
    """Static native-IP DiskANN build. No normalization or metric conversion."""

    def track(self):
        return "ood"
