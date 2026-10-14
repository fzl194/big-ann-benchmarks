# PostgreSQL exact Sparse baseline

Algorithm name: `pg-sparse-exact`. This is **not a gsvector sparse ANN index**.

CSR vectors remain sparse. Nonzero `(term,doc,weight)` postings are indexed by
term. Queries aggregate true inner products over matching postings and rank
documents by score. The baseline requires finite nonnegative SPLADE weights;
negative weights fail explicitly. If fewer than k documents have positive
scores, zero-score documents are filled in ID order.

Use `run.py --nodocker --algorithm pg-sparse-exact --dataset sparse-full
--neurips23track sparse`, with an explicitly supplied `GSVECTOR_PG_DSN`.
The parent workspace's `scripts/run-bigann-static.sh sparse` provides a
20K-base/100-query diagnostic. Full SPLADE ingestion means over a billion
postings; its runtime and storage requirements have not been validated here.
