"""Query-only locality scheduling; no index, result, or ground-truth access.

The permutation groups nearby input vectors using balanced coordinate splits.
Every query still executes the normal SQL ANN path, and results are scattered
back to the original input order. This changes scheduling, not search budgets.
"""
import numpy as np


def locality_order(queries, leaf_size=16):
    queries = np.asarray(queries)
    if queries.ndim != 2 or not np.isfinite(queries).all():
        raise ValueError("expected a finite query matrix")
    if type(leaf_size) is not int or leaf_size < 1:
        raise ValueError("leaf_size must be a positive integer")
    order = np.arange(len(queries), dtype=np.int64)
    if not len(queries) or not queries.shape[1]:
        return order
    # Scaling avoids overflow in the coordinate range without changing splits.
    coords = queries.astype(np.float64, copy=True)
    scale = np.max(np.abs(coords))
    if scale:
        coords /= scale
    pending = [(0, len(order))]
    while pending:
        lo, hi = pending.pop()
        if hi - lo <= leaf_size:
            continue
        ids = order[lo:hi]
        block = coords[ids]
        axis = int(np.argmax(np.ptp(block, axis=0)))
        order[lo:hi] = ids[np.argsort(block[:, axis], kind="stable")]
        mid = (lo + hi) // 2
        pending.extend(((mid, hi), (lo, mid)))
    return order


def query_partitions(queries, threads, ordering="input", dispatch="contiguous"):
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if ordering not in ("input", "locality"):
        raise ValueError("query_order must be input or locality")
    if dispatch not in ("contiguous", "interleaved"):
        raise ValueError("query_dispatch must be contiguous or interleaved")
    order = (locality_order(queries) if ordering == "locality"
             else np.arange(len(queries), dtype=np.int64))
    parts = (np.array_split(order, threads) if dispatch == "contiguous"
             else [order[i::threads] for i in range(min(threads, len(order)))])
    return [part for part in parts if len(part)]
