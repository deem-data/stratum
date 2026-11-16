# UK house prices

Predict house prices in two search stages:

1. **Exploration:** manual preprocessing or TableVectorizer with Ridge,
   ElasticNet, XGBoost, or LightGBM (8 candidates).
2. **Refinement:** TableVectorizer + LightGBM with learning rates
   `{0.01, 0.05, 0.1}` and depths `{5, 7}` (6 candidates).

Both use 3 shuffled folds, seed 42, R² scoring, and no final refit.
The stage definitions are in `two_level_gridsearch_exp.py` and
`two_level_gridsearch_exp2.py`, with shared preprocessing in `pipeline.py`.

From the repository root, using the project environment:

```bash
bash benchmarks/end-to-end/uk-house-price/setup.sh TARGET_DEST
bash benchmarks/end-to-end/uk-house-price/bench_pin_cores.sh 0 8 \
    TARGET_DEST/price_paid_records_200K.csv \
    --modes stratum-greedy stratum-reuse
```

Setup creates 100K, 200K, and 1M samples. Pass the desired CSV to `benchmark.py`.
By default, it runs all four modes (`skrub`, `stratum-default`, `stratum-greedy`,
`stratum-reuse`) with 8 threads. The wrapper accepts
`NUMA THREADS [--offset N] DATA.csv [--modes ...]`, selecting one logical CPU per
physical core and binding memory to that NUMA node.
