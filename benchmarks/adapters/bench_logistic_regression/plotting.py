"""Plot the LogisticRegression benchmark CSV produced by benchmark.py.

Every run records both timings and agreement columns, so one CSV feeds both
families of figures and this script writes whichever the data supports.

Timing, into ``--plots-dir``, one file per metric (fit / predict /
predict_proba). Skipped when the CSV has no timings at all, which is what
``measure_time: false`` writes:

  * rows_vs_time_cols{c}_{metric}.png : for each fixed column count c, time vs rows
  * cols_vs_time_rows{r}_{metric}.png : for each fixed row count r, time vs cols

Each holds a single panel: the Rust and scikit-learn timings (mean with +/-1 SD
error bars), each point labelled with its time and each x labelled with a bold
speedup badge (e.g. "6.2x") sitting in the gap between the curves. Points where
the Rust fast path fell back to scikit-learn are drawn hollow, since their timing
is not a real Rust measurement. Point labels are pushed apart after drawing so
the Rust and scikit-learn annotations never overlap.

Correctness, into ``--correctness-dir``, three figures per parameter set over the
(rows x cols) grid. Always written -- the agreement columns are populated by
every run, including the speed one:

  * accuracy_{param_set}.png        : class-prediction accuracy, Rust vs scikit-learn.
  * logloss_vs_true_{param_set}.png : binary cross-entropy of each implementation's
    probabilities against the true labels.
  * kl_max_{param_set}.png          : max(KL(rust || sklearn), KL(sklearn || rust))
    between the two implementations' probabilities -- 0 when they agree.

Each series is drawn train (dashed, hollow markers) and test (solid). Two-series
figures use C0 with a halo and C1 on top; the single-series KL figure drops the
halo since nothing can hide underneath it.

The two families are typeset at different scales -- the timing figures carry far
fewer labels and can afford bigger type -- so each is drawn inside its own
`rc_context` rather than under one module-level rcParams.

Run:  ./.venv/bin/python benchmark/logisticRegression/plotting.py [--csv results/benchmark.csv]
      ./.venv/bin/python benchmark/logisticRegression/plotting.py --csv results/correctness.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom mathtext fontset so \mathtt in the titles comes out bold monospace (the
# stock fontsets have no bold typewriter face). The other families are spelled
# out explicitly to keep the rest of the mathtext unchanged.
MATHTEXT_RC = {
    "mathtext.fontset": "custom",
    "mathtext.tt": "DejaVu Sans Mono:bold",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "mathtext.cal": "DejaVu Sans:italic",
}

# The timing figures carry a handful of points and can take large type; the
# correctness ones pack a whole (rows x cols) grid onto one axis and cannot.
TIME_RC = {
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 22,
    **MATHTEXT_RC,
}

CORRECTNESS_RC = {
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 19,
    **MATHTEXT_RC,
}

# Titles are set explicitly rather than through rcParams: smaller than the
# default axes title, bold, metric name in typewriter.
TIME_TITLE_FONTSIZE = 16
CORRECTNESS_TITLE_FONTSIZE = 13


def sci(v):
    """Format v as `mantissa x 10^exp` (matplotlib mathtext).

    Examples: 5.23e-5 -> 5.23x10^-5,  5.22e3 -> 5.22x10^3. The mantissa is kept in
    [1, 10) and the exponent is whatever power of ten brings it there, so small and
    large times both read cleanly regardless of scale.
    """
    if v is None or not np.isfinite(v) or v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / (10.0 ** exp)
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def cell_labels(df):
    """Return (sorted dataframe, x positions, "rows x cols" tick labels)."""
    df = df.sort_values(["rows", "cols"]).reset_index(drop=True)
    labels = [f"{r}x{c}" for r, c in zip(df["rows"], df["cols"])]
    return df, np.arange(len(df)), labels


# --------------------------------------------------------------------------
# Timing figures
# --------------------------------------------------------------------------

# Font size for the per-point time annotations.
ANNOT_FONTSIZE = 14
# The speedup badge is meant to be the thing you read first, so it is bigger,
# bold and boxed.
SPEEDUP_FONTSIZE = 18
SPEEDUP_COLOR = "#0b2f6b"
# Below this speedup the two curves are too close for the badge to fit in the
# gap between them, so it is parked underneath the lower of the two points.
BADGE_BELOW_SPEEDUP = 3.5

# (metric label, rust time col, rust std col, sklearn time col, sklearn std col, speedup col)
TIME_METRICS = [
    ("fit", "rust_fit_time", "rust_fit_std",
     "sklearn_fit_time", "sklearn_fit_std", "fit_speedup"),
    ("predict", "rust_predict_time", "rust_predict_std",
     "sklearn_predict_time", "sklearn_predict_std", "predict_speedup"),
    ("predict_proba", "rust_predict_proba_time", "rust_predict_proba_std",
     "sklearn_predict_proba_time", "sklearn_predict_proba_std", "predict_proba_speedup"),
]

# Any of these being all-NaN means the run was made with `measure_time: false`.
TIME_PROBE_COL = "rust_fit_time"


def _clipped_yerr(mean, std):
    """Symmetric SD as an asymmetric [lower, upper] error usable on a log axis.

    Where mean - SD <= 0 the lower whisker would reach zero (minus infinity in log
    space) and matplotlib would draw a spike down to the bottom of the axes. Those
    whiskers are clipped to 90% of the mean, so such a point reads as "noisy" without
    the spike; the CSV keeps the real SD.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    lower = np.where(mean - std <= 0, mean * 0.9, std)
    return np.vstack([lower, std])


def _resolve_overlaps(fig, texts, iterations=80, pad=3.0):
    """Nudge annotations vertically until none of their boxes overlap.

    Runs after the first draw, when every label has a real pixel extent: any
    overlapping pair is pushed apart along y (in offset points, so the anchor
    point the label belongs to does not move) and the process is repeated until
    the layout is clean or the iteration budget runs out.
    """
    if len(texts) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_point = fig.dpi / 72.0

    for _ in range(iterations):
        boxes = [t.get_window_extent(renderer) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if bi.x0 >= bj.x1 + pad or bj.x0 >= bi.x1 + pad:
                    continue
                overlap = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + pad
                if overlap <= 0:
                    continue
                # Push the upper label further up and the lower one further down.
                shift = (overlap / 2.0) / px_per_point
                up, down = (i, j) if bi.y0 + bi.y1 >= bj.y0 + bj.y1 else (j, i)
                for idx, sign in ((up, 1.0), (down, -1.0)):
                    x, y = texts[idx].xyann
                    texts[idx].xyann = (x, y + sign * shift)
                boxes[up] = boxes[up].translated(0, shift * px_per_point)
                boxes[down] = boxes[down].translated(0, -shift * px_per_point)
                moved = True
        if not moved:
            return


def _plot_times(ax, sub, x_col, metric):
    """Draw the timing panel for a slice `sub`, x-axis = x_col."""
    _, r_t, r_s, s_t, s_s, speed_col = metric
    x = sub[x_col].to_numpy()
    rust = sub[r_t].to_numpy()
    skl = sub[s_t].to_numpy()
    rust_sd = sub[r_s].to_numpy()
    skl_sd = sub[s_s].to_numpy()
    speedup = sub[speed_col].to_numpy()

    # Rust: split fast-path (filled) vs fell-back (hollow) markers.
    fast = sub["rust_fastpath"].astype(bool).to_numpy()
    ax.errorbar(x, rust, yerr=_clipped_yerr(rust, rust_sd), marker="o", capsize=4,
                color="C0", label="Rust")
    if (~fast).any():
        ax.scatter(x[~fast], rust[~fast], facecolors="none", edgecolors="C0",
                   s=90, zorder=5, label="Rust (fell back)")

    ax.errorbar(x, skl, yerr=_clipped_yerr(skl, skl_sd), marker="s", capsize=4,
                color="C1", label="scikit-learn")

    # Time labels go on the side of the marker that faces away from the other
    # curve, so the two series' labels open up instead of colliding. Whichever
    # curve is on top at this x gets its label above, the other one below.
    texts = []
    for xi, ry, sy in zip(x, rust, skl):
        rust_above = ry >= sy
        for yi, up, color in ((ry, rust_above, "C0"), (sy, not rust_above, "C1")):
            texts.append(ax.annotate(
                sci(yi), (xi, yi), textcoords="offset points",
                xytext=(0, 11 if up else -11),
                ha="center", va="bottom" if up else "top",
                fontsize=ANNOT_FONTSIZE, color=color, zorder=7,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.75}))

    # Speedup badge: bold, boxed, parked at the geometric midpoint of the two
    # curves, i.e. visually centred in the gap it is measuring. When the gap is
    # small (low speedup) there is no room in it, so the badge drops below the
    # lower of the two points instead.
    for xi, ry, sy, sp in zip(x, rust, skl, speedup):
        if not np.isfinite(sp):
            continue
        tight = sp <= BADGE_BELOW_SPEEDUP
        anchor = min(ry, sy) if tight else float(np.sqrt(ry * sy))
        badge = ax.annotate(
            f"{sp:.1f}×", (xi, anchor), textcoords="offset points",
            xytext=(0, -34) if tight else (0, 0),
            ha="center", va="top" if tight else "center",
            fontsize=SPEEDUP_FONTSIZE,
            fontweight="bold", color=SPEEDUP_COLOR, zorder=8,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
                  "edgecolor": SPEEDUP_COLOR, "linewidth": 1.4, "alpha": 0.92})
        texts.append(badge)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel("time [s]")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=13, loc="upper left")
    # The labels sit above/below the markers, so leave head- and footroom.
    ax.margins(x=0.14, y=0.26)
    return texts


def _make_time_figure(sub, x_col, fixed_label, metric, out_path):
    """Write a single-panel PNG for one metric, cropped tight around the axes."""
    sub = sub.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    texts = _plot_times(ax, sub, x_col, metric)

    # One tick per measured size instead of the log axis' decade ticks, which
    # would silently drop the in-between points (500 columns, 500 000 rows, ...).
    ticks = sub[x_col].to_numpy()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t):,}" for t in ticks])
    ax.set_xticks([], minor=True)

    # e.g. "predict_proba at 100 columns"; the underscore has to be escaped or
    # mathtext reads it as a subscript.
    name = metric[0].replace("_", r"\_")
    ax.set_title(rf"$\mathtt{{{name}}}$  at {fixed_label}",
                 fontsize=TIME_TITLE_FONTSIZE, fontweight="bold", pad=12)

    _resolve_overlaps(fig, texts)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _make_time_figures(sub, x_col, fixed_label, plots_dir, stem):
    for metric in TIME_METRICS:
        _make_time_figure(sub, x_col, fixed_label, metric,
                          plots_dir / f"{stem}_{metric[0]}.png")


def plot_times(df, plots_dir):
    """Write the whole timing family, or report why there is nothing to write."""
    if not np.isfinite(df[TIME_PROBE_COL]).any():
        print("[plot] no timings in this CSV (measure_time was false) "
              "-- skipping the timing figures")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # rows vs time, one figure per fixed column count and metric.
    for c in sorted(df["cols"].unique()):
        sub = df[df["cols"] == c]
        if len(sub) > 1:
            _make_time_figures(sub, "rows", f"{c:,} columns", plots_dir,
                               f"rows_vs_time_cols{c}")

    # cols vs time, one figure per fixed row count and metric.
    for r in sorted(df["rows"].unique()):
        sub = df[df["rows"] == r]
        if len(sub) > 1:
            _make_time_figures(sub, "cols", f"{r:,} rows", plots_dir,
                               f"cols_vs_time_rows{r}")


# --------------------------------------------------------------------------
# Correctness figures
# --------------------------------------------------------------------------

RUST, SKLEARN = "C0", "C1"

# When the two series agree the curves land on top of each other, which would
# hide the first one entirely. It is therefore drawn as a thick translucent halo
# and the second as a thin line on top: overlap shows up as an orange line inside
# a blue band, divergence as two separate curves.
RUST_STYLE = {"color": RUST, "lw": 4.0, "alpha": 0.45, "markersize": 11, "zorder": 2}
SKLEARN_STYLE = {"color": SKLEARN, "lw": 1.6, "markersize": 6, "zorder": 3}
# Used when a figure has a single series, so there is no curve to hide behind a halo.
SOLO_STYLE = {"color": RUST, "lw": 2.0, "markersize": 7, "zorder": 3}

CE_LABEL = "cross-entropy [nats / sample]"
KL_LABEL = "KL divergence [nats / sample]"


class Metric(NamedTuple):
    """One figure: `cols` column stems, each drawn as a train and a test curve."""
    stem: str            # output file stem, "{stem}_{param_set}.png"
    cols: tuple          # CSV column stems; "_train" / "_test" are appended
    labels: tuple        # legend label per column stem
    ylabel: str
    call: str            # estimator method the metric exercises, set in typewriter
    phrase: str          # rest of the title
    symlog: bool = False  # y on a symlog scale: KL spans decades and reaches 0


CORRECTNESS_METRICS = [
    Metric("accuracy", ("rust_acc", "sklearn_acc"), ("Rust", "scikit-learn"),
           "accuracy", "predict", "accuracy"),
    Metric("logloss_vs_true", ("logloss_rust", "logloss_sklearn"),
           ("Rust vs true labels", "scikit-learn vs true labels"),
           CE_LABEL, "predict\\_proba", "cross-entropy against the true labels"),
    # KL, not raw cross-entropy: cross-entropy between two models floors at the
    # target's own entropy (~0.3-0.4 nats here), so identical implementations
    # would still plot well above zero. KL subtracts that floor, and the max over
    # the two directions is the one number that reads as "how far apart are they"
    # -- the max rather than the min because KL is asymmetric and a correctness
    # claim should rest on the worse direction, not the flattering one.
    Metric("kl_max", ("kl_max",), ("max KL, both directions",),
           KL_LABEL, "predict\\_proba",
           "max KL divergence between Rust and scikit-learn", symlog=True),
]


def _set_title(ax, call, phrase, tail):
    """Bold, smallish, metric call in typewriter -- the timing-figure title style."""
    ax.set_title(rf"$\mathtt{{{call}}}$ {phrase} — {tail}",
                 fontsize=CORRECTNESS_TITLE_FONTSIZE, fontweight="bold", pad=12)


def _linthresh(sub, cols):
    """Where the symlog y axis switches from linear to logarithmic.

    One decade below the smallest positive value plotted, so every measured point
    lands in the log region and only the zeros (and any negative round-off) fall
    into the linear window around the origin.
    """
    values = np.concatenate([sub[f"{c}_{s}"].to_numpy(dtype=float)
                             for c in cols for s in ("train", "test")])
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1e-12
    return float(10.0 ** np.floor(np.log10(positive.min())))


def make_grid_figure(sub, name, metric, out_path):
    """One metric over the (rows x cols) grid, each series drawn train and test."""
    sub, x, labels = cell_labels(sub)
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = [SOLO_STYLE] if len(metric.cols) == 1 else [RUST_STYLE, SKLEARN_STYLE]
    for col, label, style, marker in zip(metric.cols, metric.labels, styles, "os"):
        ax.plot(x, sub[f"{col}_train"], ls="--", marker=marker, mfc="none",
                label=f"{label} train", **style)
        ax.plot(x, sub[f"{col}_test"], ls="-", marker=marker,
                label=f"{label} test", **style)

    # Cells where the Rust fast path fell back to scikit-learn are not a Rust
    # measurement, so they are shaded rather than silently compared.
    fell_back = ~sub["rust_fastpath"].astype(bool).to_numpy()
    for i, xi in enumerate(x[fell_back]):
        ax.axvspan(xi - 0.4, xi + 0.4, color="0.6", alpha=0.18, zorder=0,
                   label="fell back to scikit-learn" if i == 0 else None)

    if metric.symlog:
        # KL runs from ~1e-7 (agreement at float32 precision) to ~1e-1 (a real
        # disagreement), and is exactly 0 when the two models match -- which a
        # plain log scale cannot draw. symlog keeps the decades readable and the
        # linear window around 0 wide enough to hold the round-off noise.
        ax.set_yscale("symlog", linthresh=_linthresh(sub, metric.cols))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("rows x cols")
    ax.set_ylabel(metric.ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=10)
    _set_title(ax, metric.call, metric.phrase, f"{name} parameters")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_correctness(df, plots_dir):
    """Write the whole correctness family; every run populates these columns."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in CORRECTNESS_METRICS:
        for name, sub in df.groupby("param_set"):
            if len(sub) > 1:
                make_grid_figure(sub, name, metric,
                                 plots_dir / f"{metric.stem}_{name}.png")


def main():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(script_dir / "results" / "benchmark.csv"))
    ap.add_argument("--plots-dir", default=str(script_dir / "plots"),
                    help="timing figures (default: plots/)")
    ap.add_argument("--correctness-dir", default=None,
                    help="correctness figures (default: <plots-dir>/correctness)")
    ap.add_argument("--only", choices=("time", "correctness"), default=None,
                    help="write just one family (default: both)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plots_dir = Path(args.plots_dir)
    correctness_dir = (Path(args.correctness_dir) if args.correctness_dir
                       else plots_dir / "correctness")

    if args.only != "correctness":
        with plt.rc_context(TIME_RC):
            plot_times(df, plots_dir)
    if args.only != "time":
        with plt.rc_context(CORRECTNESS_RC):
            plot_correctness(df, correctness_dir)


if __name__ == "__main__":
    main()
