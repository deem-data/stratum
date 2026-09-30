# QuantileTransformer benchmark

Compares the stratum `QuantileTransformer` against scikit-learn's, over a grid of
(rows × columns). One CSV per run; one plotting script for both the timing and the
agreement figures.

Only `fit` is native (a rayon kernel replacing `np.nanpercentile`) — hence the
`stratum_` column prefix rather than `rust_`. `transform` stays in NumPy and wins
from `ndtri`/`ndtr` instead of `scipy.stats.norm.ppf`/`cdf`, a skipped reverse
interpolation pass, an all-finite fast path, and the `n_jobs` fan-out over
features.

## Run

```bash
# speed run: rows × cols × n_jobs × output_distribution, timings on
./.venv/bin/python benchmark/quantileTransformer/benchmark.py

# agreement run: four parameter sets over the awkward data shapes, timings off
./.venv/bin/python benchmark/quantileTransformer/benchmark.py config_correctness.json
```

A bare config name is resolved next to the script. Output always lands in
`results/`; the config's `output_csv` names the file, not its directory. Rows are
flushed as each cell finishes, so an interrupted run keeps what it already
measured.

## Plot

```bash
./.venv/bin/python benchmark/quantileTransformer/plotting.py
./.venv/bin/python benchmark/quantileTransformer/plotting.py --csv results/correctness.csv
```

Timing figures go to `plots/`, correctness figures to `plots/correctness/`.
Useful flags: `--csv`, `--plots-dir`, `--correctness-dir`, `--only time|correctness`,
and `--showtimes` to label every point with its exact time (off by default — the
multi-curve layout carries a few dozen labels).

| figure | what it shows |
| --- | --- |
| `rows_vs_time_cols{c}_{set}_{metric}.png` | time vs rows at a fixed column count |
| `cols_vs_time_rows{r}_{set}_{metric}.png` | time vs columns at a fixed row count |
| ↳ e.g. `cols_vs_time_rows10000_normal_transform.png`, and `..._fit.png` — `fit` carries no `{set}`, see below | |
| `agreement_{set}.png` | max and mean \|Δ\| for `quantiles_`, `transform`, `fit_transform`, `inverse_transform` |
| `roundtrip_{set}.png` | `inverse_transform(transform(X))` vs X, per implementation |
| `checks_{set}.png` | pass/fail matrix: shape, monotonicity, NaN placement, fast path |

`{metric}` is `fit`, `transform` or `fit_transform`.

## The two extra axes

- **`n_jobs`** (top level) is stratum-only, the `ThreadPoolExecutor` width in
  `_transform`. It changes only how fast the transform is, never what it returns,
  so its values are **curves inside one figure**. With several values the timing
  figure grows a second panel: speed-up per thread count as bars anchored at parity.
- **`output_distribution`** may be a string or a list. It changes *what*
  `transform` computes (`uniform` runs scikit-learn's own `_transform_col`,
  `normal` the `ndtri`/`ndtr` rewrite), so each value gets **its own figure**.

`fit` reads neither axis, so its timing is measured once per distinct fit and
reused; its figures carry no distribution in the name, and every `n_jobs` value
holds the same number by construction. A fit speed-up therefore comes from the
rayon kernel, never from the thread fan-out.

## Config knobs

`rows`, `cols`, `repeats`, `warmup`, `measure_time` as usual. `qt_params` (one
dict) is the speed form, `qt_param_sets` (a list) the agreement form.
`subsample: null` makes the fit kernel see every row — which is what makes the
fit measurement a measurement; at scikit-learn's default of 10 000 the fit cost
stops growing with `rows`. A `data` block picks the column distribution
(`normal`, `lognormal`, `ties`, `constant`) plus `nan_frac` and
`n_all_nan_cols`; that belongs in the agreement config, since it changes what the
answer is rather than how fast you get there.

## Worth knowing

- **Memory ceiling.** One matrix is rows × cols × 8 bytes and a cell holds
  several at once: 100 000 × 1000 is 0.8 GB per matrix, 1 000 000 × 1000 is 8 GB.
  `cols: 10000` is only viable against the smallest row count.
- **`rows` is the total.** `rows: 1000` with `test_size: 0.2` fits on 800.
- **`n_transform_rows` does not describe the timings.** It records the held-out
  split, but the *timed* transform runs on the fit split — so throughput derived
  from the CSV is off by `1/test_size`. The agreement columns do use the held-out
  split.
- **`SKRUB_RUST_THREADS`** sizes the rayon pool used by the fit kernel (0 or
  unset = rayon's global pool). It is independent of `n_jobs`, which only affects
  `transform`.
- **Check `stratum_fastpath`.** False means the kernel declined and the adapter
  fell back to scikit-learn's `_dense_fit`, so the cell compares scikit-learn
  against itself. The console prints `(FELL BACK!)`; such cells are drawn hollow
  in the timing figures and shaded in the correctness ones.
- Agreement is reported as raw distances, not pass/fail against a tolerance —
  what is being measured is float64 round-off, and the magnitude is the result.
