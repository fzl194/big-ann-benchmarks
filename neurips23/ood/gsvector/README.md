# Native gsvector OOD adapter

Uses a dedicated `vector_ip_ops` index and the `<#>` negative-dot-product
operator. Requires gsvector-pg 0.1.1 and its updated shared library.
Vectors are neither normalized nor transformed into L2.

Connection, schema ownership and binary vector transport are shared with the
streaming adapter. Static fit loads all actual base vectors before building the
index; query-argument sweeps keep that index instead of resetting the workload.

```bash
python run.py --nodocker --algorithm gsvector --dataset text2image-10M \
  --neurips23track ood --count 10 --runs 1 --rebuild --force
```

Set `GSVECTOR_PG_DSN` to a dedicated instance first. The parent workspace's
`scripts/run-bigann-static.sh ood` creates a private instance and runs an explicit
40K-base/100-query diagnostic. Full-scale results and competition resource
limits must be validated separately.
