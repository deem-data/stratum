# LogisticRegression benchmark

Compares the stratum Rust-backed `LogisticRegression` against scikit-learn's, over
a grid of (rows × columns). One CSV per run; one plotting script for both the
timing and the agreement figures.

## Run

```bash
# speed run: one parameter set, timings on
./.venv/bin/python benchmark/logisticRegression/benchmark.py

# agreement run: three parameter sets, timings off
./.venv/bin/python benchmark/logisticRegression/benchmark.py config_correctness.json
```

A bare config name is resolved next to the script, so either invocation works
from anywhere. Output always lands in `results/`; the config's `output_csv` names
the file, not its directory. Rows are flushed as each cell finishes, so an
interrupted run keeps what it already measured.

## Plot

```bash
# both families from one CSV
./.venv/bin/python benchmark/logisticRegression/plotting.py

# a run made with measure_time: false -- the timing family is skipped
./.venv/bin/python benchmark/logisticRegression/plotting.py --csv results/correctness.csv
```

Timing figures go to `plots/`, correctness figures to `plots/correctness/`.
Useful flags: `--csv`, `--plots-dir`, `--correctness-dir`, `--only time|correctness`.

| figure | what it shows |
| --- | --- |
| `rows_vs_time_cols{c}_{metric}.png` | time vs rows at a fixed column count |
| `cols_vs_time_rows{r}_{metric}.png` | time vs columns at a fixed row count |
| `accuracy_{set}.png` | train/test accuracy, Rust vs scikit-learn |
| `logloss_vs_true_{set}.png` | cross-entropy of each against the true labels |
| `kl_max_{set}.png` | max KL between the two probability vectors; 0 = agreement |

`{metric}` is `fit`, `predict` or `predict_proba`.

## Config knobs

`rows`, `cols` — the grid. `repeats` / `warmup` — timed runs per cell (a fresh
estimator per fit repeat, so each is a cold fit). `measure_time` — false writes
NaN into every timing column. `lr_params` (one dict) is the speed form,
`lr_param_sets` (a list) the agreement form; nothing stops a config from carrying
`lr_param_sets` *and* `measure_time: true`, it just costs the cross product.

`sklearn_solver` is per parameter set because scikit-learn's `lbfgs` only supports
L2 — `l1_ratio > 0` needs `saga`, which is markedly slower. A set may carry a
`sample_weight` spec, drawn once per cell and passed to both implementations.

## Worth knowing

- **`rows` is the total, not the training size.** `rows: 1000` with
  `test_size: 0.2` fits on 800 and tests on 200, so the plots' x-axis is 25%
  larger than what was actually measured.
- **Thread count is not a grid axis.** The Rust LR kernels run on rayon's global
  pool and ignore `SKRUB_RUST_THREADS`; only `fit` is parallel at all.
- **Check `rust_fastpath`.** False means the adapter fell back to scikit-learn
  and the cell compares scikit-learn against itself. The console prints
  `(FELL BACK!)`, and such points are drawn hollow or shaded.
- `results/correctness.csv` is currently missing from the repo, so the committed
  `plots/correctness/*.png` cannot be regenerated without re-running the
  agreement sweep.
