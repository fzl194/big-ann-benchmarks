# YFCC filtered L2 experiments

`PGFilteredANN` keeps `filter_strategy=legacy` as the old comparison path:
GIN capped counting, SQL post-filtering, repeated ANN expansion, and exact
fallback. It is not a graph with labels embedded in its navigation.

The opt-in `filter_strategy=bounded` changes two things:

* Probe at most `exact_cutoff + 1` eligible IDs. If the set is complete, reuse
  those IDs for exact L2 ranking through the primary key. A materialized CTE
  prevents the label predicate being pushed into another GIN search. Labels
  and row visibility are still checked. PostgreSQL `work_mem` governs the
  intermediate; it can spill. No Python vector array is created.
* Otherwise execute one ANN query with `ann_candidates`, then use exact
  fallback if fewer than k results survive. This avoids repeated graph starts;
  it does not resume traversal and does not guarantee higher recall or QPS.

All steps for one query use one repeatable-read, read-only snapshot. A truncated
probe is only evidence that the set exceeds the cutoff; it is never treated as
the complete set or used as a selectivity estimate. Empty filters skip probing.
`exact_cutoff=0` disables probing; bounded mode does **not** silently enlarge the
cutoff with dataset size. `ann_candidates` is explicit and bounded by
`max_candidates`; effective L is at least that candidate budget. Increase it
only when measured recall/fallback cost justifies it. Returning k rows is not a
recall guarantee.

Statistics distinguish `eligible_probe_queries`, `eligible_probe_rows`,
`reused_eligible_ids`, `ann_attempts`, `fallback_exact_queries`, and the strategy.
The exact branch still reads SQL vectors: compact-code graph refinement does
not optimize that branch or turn post-filtering into graph-label filtering.

Build options now expose `pq_m`, `build_neighbor_code_bytes`,
`build_raw_cache_mb`, and `build_gather_threads`; they are local controls, not a
whole-process memory ceiling. `prepare_refine=true` prepares the derived code
sidecar during fit and records its time separately. Query `refine_sq8=true`
requires that preparation, so query timing cannot silently build sidecars.
It enables the existing L2 graph refinement and budgeted PostgreSQL shared
cache. Both off/on arms in the example prepare the same sidecars.

The example also enables build option `isolate_refine_cache=true`, exclusively
for the dedicated benchmark PostgreSQL instance. During `fit`, private wrappers
capture the original requested cache cap. During `set_query_arguments` (before
the framework query timer), off requests zero and waits up to ten seconds for
allocated, retired, and pending bytes to reach zero; on restores that original
cap. A failed release aborts the point. This does not change automatic budgeting
or force pages to become resident. Cleanup restores the original requested cap.
No resize, reclamation wait, or code preparation occurs in `filtered_query`.
Repetitions retain the framework's existing warm-through behavior; they are not
cold-cache trials. The feature changes a cluster-wide setting, so never use it
against a shared production database. It defaults to false for legacy callers.

`gsvector-filter-bounded` supplies explicit 1024/4096 candidate experiments and
refinement off/on variants. Rebuild the matching benchmark image first; its
inherited release tag is not evidence an old downloaded image supports these
options. Keep data, resources, R/L/precision, and timing protocol fixed when
comparing; changing the neighbor-code layout requires a separately built index.
Report same-recall performance and all fallback counts, not only successful
ANN queries. No full YFCC performance improvement is asserted by these tests.

Tests: `python -m unittest discover -s tests -p test_filtered_budget.py`.
With a dedicated matching PostgreSQL instance in `GSVECTOR_PG_DSN`, run
`python -m unittest discover -s tests -p test_pg_static.py`; the bounded test
covers AND/missing/empty labels, exact results, single-attempt fallback, explicit
refine preparation, and a concurrent label change between probing and scoring.
