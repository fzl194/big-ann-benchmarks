# PostgreSQL exact Filter baseline

Algorithm name: `pg-filter-exact`. This is **not gsvector ANN filtering**.

Loads dense vectors and their variable-length label sets. A GIN index evaluates
`tags @> required_tags` (AND semantics), then the extension's scalar L2 operator
ranks eligible rows exactly. No global ANN top-k is filtered after truncation.
Empty label sets match all rows; insufficient candidates are padded with -1.

Use `run.py --nodocker --algorithm pg-filter-exact --dataset yfcc-10M
--neurips23track filter`, with an explicitly supplied `GSVECTOR_PG_DSN`.
The parent workspace's `scripts/run-bigann-static.sh filter` provides a
40K-base diagnostic with 100 explicitly selected real queries.
