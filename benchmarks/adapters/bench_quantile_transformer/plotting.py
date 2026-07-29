"""Plot the QuantileTransformer benchmark CSV produced by benchmark.py.

Every run records both timings and agreement columns, so one CSV feeds both
families of figures and this script writes whichever the data supports.

Timing, into ``--plots-dir``, one file per metric (fit / transform /
fit_transform). Skipped when the CSV has no timings at all, which is what
``measure_time: false`` writes:

  * rows_vs_time_cols{c}_{metric}.png : for each fixed column count c, time vs rows
  * cols_vs_time_rows{r}_{metric}.png : for each fixed row count r, time vs cols

Correctness, into ``--correctness-dir``, three figures over the (rows x cols)
grid. Always written -- the agreement columns are populated by every run,
including the speed one:

  * agreement_{param_set}.png - how far the stratum and scikit-learn results are
    apart, for all four comparisons the benchmark makes: the fitted
    ``quantiles_``, ``transform``, ``fit_transform`` and ``inverse_transform``.
    Each comparison is drawn twice, solid for the worst entry (max |Δ|) and
    dashed for the average one (MAE), so a single bad entry is distinguishable
    from a uniformly shifted matrix. No tolerance line is drawn: agreement here
    is float64 round-off, a few ulps, and the magnitude is the result rather
    than a verdict to threshold.
  * roundtrip_{param_set}.png - ``inverse_transform(transform(X))`` against the
    original X, for each implementation separately. Unlike the figure above this
    is not a comparison between them: quantile transformation genuinely loses
    information at the clipped tails, so neither curve is expected to sit at
    zero. What proves correctness is that the two curves coincide.
  * checks_{param_set}.png - the structural invariants per grid cell as a
    pass/fail matrix: the ``quantiles_`` shape and monotonicity, NaN placement,
    and whether the native fast path ran at all. These are the properties that
    hold exactly or not at all, so unlike the magnitudes above they are the one
    thing in the run that does read as a verdict.

Why max |Δ| and not a relative error: the transformed values are bounded (in
[0, 1] for uniform output, clipped to about ±5.2 for normal), so an absolute
difference is already a scale-free quantity here, and it stays defined where the
reference value is 0 -- which for a quantile transform is a common output, not an
edge case.

How figures are split
---------------------
Both families split on parameter set AND output distribution, and both keep
``n_jobs`` inside a figure. Those two knobs change what ``transform`` computes --
``uniform`` runs scikit-learn's own ``_transform_col``, ``normal`` the
``ndtri``/``ndtr`` rewrite -- so mixing them on one axes would compare different
work; ``n_jobs`` does not, so its values are drawn as CURVES within the same
figure, which is the comparison this benchmark exists to make.

The distribution needs naming separately from ``param_set`` because config.json
sweeps it as a TOP-LEVEL axis: benchmark.py records it in its own column and
leaves ``param_set`` at ``default`` for the whole run. Grouping on ``param_set``
alone would therefore put both distributions on one axes, giving every curve two
points per x. The figure name follows from what the CSV distinguishes: unnamed
sets are named after the distribution alone (``..._normal_transform.png``), named
ones keep their name and take a suffix only when that name spans more than one
distribution.

``fit`` is the exception: it computes the quantiles, which the distribution has
no say in, so it is drawn ONCE for all distributions and its figures carry no
distribution in their name (``..._fit.png``) or subtitle. The rows the CSV holds
for the other distributions are dropped rather than averaged into it -- they are
not repeated measurements but the same measurement copied, since benchmark.py
caches each distinct fit timing and reuses it across the distribution axis.

Timing layout
-------------
The layout adapts to how many n_jobs values the CSV holds:

  * ONE n_jobs value -> the logisticRegression layout exactly. A single panel:
    stratum and scikit-learn timings (mean with +/-1 SD error bars and a bold
    speedup badge in the gap between the curves).

  * SEVERAL n_jobs values -> colour encodes n_jobs, line style the implementation
    (solid + filled marker = stratum, dashed + hollow = scikit-learn), and a
    second panel underneath carries the speed-up per thread count as grouped bars
    anchored at parity, each labelled `5.2x`. The inline speed-up badge of the
    two-curve layout has no equivalent here -- it lives in the gap between two
    curves, and with several curves there is no single gap to put it in -- so the
    bar panel takes over that job, and reads more precisely for it.

``--showtimes`` labels every point with its exact time in `1.03 x 10^-2` form.
It is off by default: the multi-series layout carries two labels per curve per
x, which is a few dozen on one figure, and they cost more legibility than they
add once the shape of the curves is the thing being read. The speed-up badges
and bars are drawn either way -- they report ratios, not times.

Note that ``fit`` is unaffected by ``n_jobs`` (a transform-only parameter), and
benchmark.py measures it once and reuses it across the axis, so every n_jobs
value in a ``fit`` figure holds THE SAME NUMBER, by construction rather than by
measurement. Only one of them is drawn, and the layout collapses back to the
two-curve one. Nothing on the figure marks the collapse: stacking identical
curves would suggest thread counts that happened to agree, and a title caveat
would repeat on every ``fit`` figure something that is true of the metric
itself. That the fit speed-up comes from the rayon kernel rather than from the
thread fan-out is the finding here, and it is this paragraph's job to say so.

Points where the native fast path fell back to scikit-learn are drawn hollow (in
the timing figures) or shaded (in the correctness ones), since such a cell is not
a measurement of the native kernel at all. With --showtimes, the point labels are
pushed apart after drawing so the annotations never overlap.

The two families are typeset at different scales -- the timing figures carry far
fewer points per axis and can afford much bigger type -- so each is drawn inside
its own `rc_context` rather than under one module-level rcParams.

Run:  ./.venv/bin/python benchmark/quantileTransformer/plotting.py [--csv results/benchmark.csv] [--showtimes]
      ./.venv/bin/python benchmark/quantileTransformer/plotting.py --csv results/correctness.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

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

# The timing figures carry a handful of x positions and can take large type; the
# correctness ones pack a whole (rows x cols) grid onto one axis and cannot.
TIME_RC = {
    "font.size": 23,
    "axes.titlesize": 27,
    "axes.labelsize": 25,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 29,
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
TIME_TITLE_FONTSIZE = 23
CORRECTNESS_TITLE_FONTSIZE = 13

# What splits FIGURES in both families: both knobs change WHAT is computed.
# `param_set` is the correctness config's named set; `output_distribution` is
# config.json's own top-level axis, which benchmark.py keeps in this column
# rather than folding into the set name.
FIGURE_COLS = ["param_set", "output_distribution"]

# Curves within a timing figure. n_jobs is the one knob that changes only HOW
# FAST the work is done, so its values belong on the same axes.
SERIES_COL = "n_jobs"

# The name benchmark.py gives a run that defines no named sets -- config.json's
# single `qt_params` dict. It carries no information, so figures from such a run
# are named after the distribution alone.
PLACEHOLDER_SET = "default"

STRATUM_LABEL = "stratum"
SKLEARN_LABEL = "scikit-learn"


def figure_groups(df, per_distribution=True):
    """Yield (name, sub, covered) for each figure the CSV asks for.

    ``covered`` lists the distributions the figure stands for, which is one of
    them normally and all of them for a collapsed `fit` figure. No figure draws
    it: which measurements the curves speak for is a property of the metric, and
    belongs in this module's docstring rather than on every title. It is yielded
    so a caller that does want to report the collapse has it to hand.

    With ``per_distribution`` the split is one figure per set x distribution, and
    the name is the parameter set with the distribution appended -- except in the
    two cases where that would only add noise: an unnamed set (`default`) is
    replaced by the distribution rather than prefixed by a placeholder, and a set
    that was run against a single distribution keeps its bare name, since the
    suffix would be constant across every figure it appears in. So config.json
    yields `..._uniform_transform.png` / `..._normal_transform.png`, and the
    correctness config's sets keep the `baseline` / `ragged` / ... filenames they
    already had. Both families use this rule, so a correctness run and a timing
    run of the same configuration land on matching filenames.

    Without it -- for `fit`, the one metric that never reads the distribution --
    the distributions collapse into a single figure per set and the name loses
    the suffix (so `..._fit.png`, and just `fit` for an unnamed set). The rows of
    one distribution are kept and the rest dropped rather than averaged, because
    they are not repeated measurements: benchmark.py measures each distinct fit
    once and copies the result across the distribution axis, so the duplicates
    are the same numbers and drawing them would put two identical points on every
    x.
    """
    spans = df.groupby("param_set", sort=False)["output_distribution"].nunique()
    for param_set, set_sub in df.groupby("param_set", sort=False):
        if not per_distribution:
            covered = tuple(str(d) for d in pd.unique(set_sub["output_distribution"]))
            sub = set_sub[set_sub["output_distribution"] == covered[0]]
            yield ("" if param_set == PLACEHOLDER_SET else str(param_set)), sub, covered
            continue
        for distribution, sub in set_sub.groupby("output_distribution", sort=False):
            if param_set == PLACEHOLDER_SET:
                name = str(distribution)
            elif spans.get(param_set, 1) > 1:
                name = f"{param_set}_{distribution}"
            else:
                name = str(param_set)
            yield name, sub, (str(distribution),)


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

# Font size for the per-point time annotations. The multi-series layout carries
# one per curve per point -- up to a few dozen on a figure -- so it labels at a
# smaller size than the two-curve layout, which has room for the full size.
ANNOT_FONTSIZE = 21
MULTI_ANNOT_FONTSIZE = 18
# Speed-up bar labels ("5.2×"), rotated upright inside the bar's column.
BAR_LABEL_FONTSIZE = 19
# The speedup badge is meant to be the thing you read first, so it is bigger,
# bold and boxed.
SPEEDUP_FONTSIZE = 25
SPEEDUP_COLOR = "#0b2f6b"
# Below this speedup the two curves are too close for the badge to fit in the
# gap between them, so it is parked underneath the lower of the two points.
BADGE_BELOW_SPEEDUP = 3.5

# Colours for the multi-series layout, one per n_jobs value. C0/C1 are skipped:
# they mean "stratum" and "scikit-learn" in the single-series layout and in the
# correctness figures, and reusing them for a thread count would collide with
# that meaning.
SET_COLORS = ("C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")

# All-NaN in this column means the run was made with `measure_time: false`.
TIME_PROBE_COL = "stratum_fit_time"


class Metric(NamedTuple):
    """One figure: the columns holding one method's timings and its speed-up.

    ``per_distribution`` says whether this method reads ``output_distribution``
    at all, and so whether the distributions deserve separate figures.
    """
    name: str
    stratum_time: str
    stratum_std: str
    sklearn_time: str
    sklearn_std: str
    speedup: str
    per_distribution: bool = True


# The three measured methods. `fit` is the native rayon kernel, `transform` the
# NumPy/ndtri rewrite plus the n_jobs fan-out, `fit_transform` both at once.
# Only `fit` is distribution-independent: the mapping the distribution selects
# happens in `transform`, which `fit_transform` also runs.
TIME_METRICS = [
    Metric("fit", "stratum_fit_time", "stratum_fit_std",
           "sklearn_fit_time", "sklearn_fit_std", "fit_speedup",
           per_distribution=False),
    Metric("transform", "stratum_transform_time", "stratum_transform_std",
           "sklearn_transform_time", "sklearn_transform_std", "transform_speedup"),
    Metric("fit_transform", "stratum_fit_transform_time", "stratum_fit_transform_std",
           "sklearn_fit_transform_time", "sklearn_fit_transform_std",
           "fit_transform_speedup"),
]


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


def series_label(sub):
    """Legend label for one curve: the thread count it was measured at."""
    return f"n_jobs={int(sub.iloc[0][SERIES_COL])}"


def _annotate_times(ax, x, stratum, skl, speedup, show_times):
    """Per-point time labels plus the speedup badge; single-set layout only.

    The speedup badge is drawn either way: it reports a ratio, not a time, and it
    is the thing the figure exists to show. Only the per-point times answer to
    `show_times`.
    """
    texts = []
    # Time labels go on the side of the marker that faces away from the other
    # curve, so the two series' labels open up instead of colliding. Whichever
    # curve is on top at this x gets its label above, the other one below.
    if show_times:
        for xi, ry, sy in zip(x, stratum, skl):
            stratum_above = ry >= sy
            for yi, up, color in ((ry, stratum_above, "C0"),
                                  (sy, not stratum_above, "C1")):
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
    for xi, ry, sy, sp in zip(x, stratum, skl, speedup):
        if not np.isfinite(sp):
            continue
        tight = sp <= BADGE_BELOW_SPEEDUP
        anchor = min(ry, sy) if tight else float(np.sqrt(ry * sy))
        texts.append(ax.annotate(
            f"{sp:.1f}×", (xi, anchor), textcoords="offset points",
            xytext=(0, -34) if tight else (0, 0),
            ha="center", va="top" if tight else "center",
            fontsize=SPEEDUP_FONTSIZE,
            fontweight="bold", color=SPEEDUP_COLOR, zorder=8,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
                  "edgecolor": SPEEDUP_COLOR, "linewidth": 1.4, "alpha": 0.92}))
    return texts


def _plot_times_single(ax, sub, x_col, metric, show_times):
    """Timing panel for one parameter set: the logisticRegression figure."""
    x = sub[x_col].to_numpy()
    stratum = sub[metric.stratum_time].to_numpy()
    skl = sub[metric.sklearn_time].to_numpy()

    # stratum: split fast-path (filled) vs fell-back (hollow) markers.
    fast = sub["stratum_fastpath"].astype(bool).to_numpy()
    ax.errorbar(x, stratum, yerr=_clipped_yerr(stratum, sub[metric.stratum_std]),
                marker="o", capsize=4, color="C0", label=STRATUM_LABEL)
    if (~fast).any():
        ax.scatter(x[~fast], stratum[~fast], facecolors="none", edgecolors="C0",
                   s=90, zorder=5, label=f"{STRATUM_LABEL} (fell back)")

    ax.errorbar(x, skl, yerr=_clipped_yerr(skl, sub[metric.sklearn_std]),
                marker="s", capsize=4, color="C1", label=SKLEARN_LABEL)

    texts = _annotate_times(ax, x, stratum, skl, sub[metric.speedup].to_numpy(),
                            show_times)
    ax.legend(fontsize=20, loc="upper left")
    return texts


def _plot_times_multi(ax, groups, x_col, metric, show_times):
    """Timing panel for several thread counts, colour-coded per n_jobs value.

    Two legends rather than one entry per curve: with four thread counts a
    combined legend would carry eight near-identical strings. Colour (the thread
    count) and line style (the implementation) are independent encodings, so they
    get independent keys.
    """
    texts = []
    for color, (_, sub) in zip(SET_COLORS, groups):
        x = sub[x_col].to_numpy()
        stratum = sub[metric.stratum_time].to_numpy()
        skl = sub[metric.sklearn_time].to_numpy()
        fast = sub["stratum_fastpath"].astype(bool).to_numpy()

        ax.errorbar(x, stratum, yerr=_clipped_yerr(stratum, sub[metric.stratum_std]),
                    marker="o", ls="-", capsize=3, color=color, lw=1.8)
        if (~fast).any():
            ax.scatter(x[~fast], stratum[~fast], facecolors="white",
                       edgecolors=color, s=80, zorder=5)
        ax.errorbar(x, skl, yerr=_clipped_yerr(skl, sub[metric.sklearn_std]),
                    marker="s", ls="--", mfc="none", capsize=3, color=color,
                    lw=1.4, alpha=0.85)

        # Same `sci()` time labels as the logisticRegression figures. The
        # placement rule is fixed here rather than "whichever curve is on top",
        # which is what the two-curve layout uses: with several curves that
        # heuristic flips per series and the labels interleave. stratum always
        # reads above its marker and scikit-learn below, so a colour's pair is
        # always the same way round, and _resolve_overlaps separates what is
        # left after the first draw.
        if show_times:
            for xi, ry, sy in zip(x, stratum, skl):
                for yi, up in ((ry, True), (sy, False)):
                    texts.append(ax.annotate(
                        sci(yi), (xi, yi), textcoords="offset points",
                        xytext=(0, 8 if up else -8),
                        ha="center", va="bottom" if up else "top",
                        fontsize=MULTI_ANNOT_FONTSIZE, color=color, zorder=7,
                        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white",
                              "edgecolor": "none", "alpha": 0.72}))

    color_keys = [Line2D([], [], color=c, lw=2.6, label=series_label(sub))
                  for c, (_, sub) in zip(SET_COLORS, groups)]
    style_keys = [
        Line2D([], [], color="0.25", lw=1.8, ls="-", marker="o", label=STRATUM_LABEL),
        Line2D([], [], color="0.25", lw=1.4, ls="--", marker="s", mfc="none",
               label=SKLEARN_LABEL),
    ]
    first = ax.legend(handles=color_keys, fontsize=19, loc="upper left",
                      title="threads", title_fontsize=19, framealpha=0.9)
    ax.add_artist(first)
    ax.legend(handles=style_keys, fontsize=19, loc="lower right", framealpha=0.9)
    return texts


def _log_bar_edges(xs, n_series, index):
    """Left edges and widths for one series of grouped bars on a LOG x axis.

    The panel shares its x-axis with the timing panel above, which is
    logarithmic, so bars cannot be placed by adding a constant offset to x: a
    fixed offset is a different visual distance at 10 000 than at 1 000 000, and
    the groups would drift apart across the axis. The grouping is therefore laid
    out in LOG space -- offsets and widths are fractions of the smallest gap
    between adjacent measured sizes -- and converted back, which keeps every
    group the same width on screen and centred on its tick.
    """
    logs = np.log10(np.asarray(xs, dtype=float))
    unique = np.unique(np.log10(np.asarray(sorted(set(xs)), dtype=float)))
    # One measured size on the axis has no neighbour to measure a gap against;
    # 0.3 decades is a readable default width in that case.
    spacing = float(np.min(np.diff(unique))) if unique.size > 1 else 0.3
    bar = 0.72 * spacing / n_series
    centers = logs + (index - (n_series - 1) / 2.0) * bar
    left = 10.0 ** (centers - bar * 0.44)
    right = 10.0 ** (centers + bar * 0.44)
    return left, right - left


def _plot_speedup(ax, groups, x_col, metric):
    """Middle panel (multi-series layout only): speed-up per thread count.

    This is where the n_jobs comparison actually reads. Absolute times in the
    panel above are dominated by the problem size, so two thread counts that
    differ by 2x look identical there on a log axis spanning decades; the ratio
    removes the size and leaves only what the extra threads bought.

    Bars rather than a line, anchored at parity: a speed-up is a magnitude
    against a fixed reference, not a trajectory through a continuum, so the ink
    should measure the distance from 1x rather than connect neighbouring
    measurements as if the values in between meant something. Anchoring also
    makes the sign readable at a glance -- bars grow up when stratum wins and
    down when it loses.
    """
    n_series = len(groups)
    for index, (color, (_, sub)) in enumerate(zip(SET_COLORS, groups)):
        x = sub[x_col].to_numpy(dtype=float)
        speedup = sub[metric.speedup].to_numpy(dtype=float)
        left, width = _log_bar_edges(x, n_series, index)

        finite = np.isfinite(speedup) & (speedup > 0)
        if not finite.any():
            continue
        # `bottom=1` with `height = speedup - 1` spans the rectangle from parity
        # to the measured ratio, in either direction, and stays strictly positive
        # so the log axis can draw it.
        ax.bar(left[finite], speedup[finite] - 1.0, width=width[finite], bottom=1.0,
               align="edge", color=color, edgecolor="white", linewidth=0.6,
               zorder=3)

        for xi, wi, sp in zip(left[finite], width[finite], speedup[finite]):
            above = sp >= 1.0
            ax.annotate(f"{sp:.1f}×", (xi + wi / 2.0, sp),
                        textcoords="offset points", xytext=(0, 3 if above else -3),
                        ha="center", va="bottom" if above else "top",
                        fontsize=BAR_LABEL_FONTSIZE, fontweight="bold", color=color,
                        rotation=90, zorder=4)

    # Parity: below this line the stratum implementation is the slower one, which
    # is a legitimate outcome for `transform` on narrow matrices (the Fortran-order
    # copy the n_jobs path takes is not free) and should be visible, not cropped.
    ax.axhline(1.0, color="0.35", ls="--", lw=1.3, zorder=2)
    ax.text(0.995, 1.0, " parity ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=19, fontweight="bold", color="0.35")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("speed-up\n[× scikit-learn]", fontsize=22)
    ax.grid(True, which="both", ls=":", alpha=0.4, zorder=0)
    # Generous vertical margin: the bar labels are rotated upright and sit past
    # the end of their bar, so they need roughly a label's length of clearance
    # beyond the tallest and lowest bar.
    ax.margins(x=0.14, y=0.55)

    # A log axis labels its ticks as `4 x 10^0`, which is unreadable for a ratio
    # that mostly lives between 0.5 and 50. Both tick levels are relabelled as
    # plain multipliers -- the minor ones too, since a range under one decade may
    # contain no major tick at all and would otherwise come out unlabelled.
    for which in ("major", "minor"):
        ax.yaxis.set_tick_params(which=which, labelleft=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(FuncFormatter(
        # Only round-ish minors get a label; the rest would crowd the axis.
        lambda v, _: f"{v:g}×" if v in (0.2, 0.5, 2, 5, 20, 50) else ""))


def _identical_across_series(groups, x_col, metric):
    """True when every thread count produced exactly the same timings.

    This is the expected state of a ``fit`` figure, not an anomaly: ``n_jobs``
    only reaches ``transform``, so benchmark.py measures each distinct fit once
    and reuses it across the axis. Drawing that as N superimposed curves would
    stack N copies of every point label and N identical bars, which reads as
    noise rather than as the finding it is.
    """
    signatures = {
        (tuple(sub[x_col]), tuple(sub[metric.stratum_time]), tuple(sub[metric.sklearn_time]))
        for _, sub in groups
    }
    return len(signatures) == 1


def _make_time_figure(sub, x_col, fixed_label, metric, out_path, show_times):
    """Write one PNG for one metric, cropped tight around the axes."""
    # One curve per thread count, each sorted along its x axis. Sorted by n_jobs
    # (not by config order) so the legend and the colour ramp both run from the
    # fewest threads to the most.
    groups = [(value, g.sort_values(x_col))
              for value, g in sub.groupby(SERIES_COL, sort=True)]

    # When the thread counts are indistinguishable (see above), keep one of them
    # and fall back to the two-curve layout -- which is strictly more readable
    # here, since it restores the inline speed-up badges. Drawing all of them
    # would stack identical curves on one another and read as a measurement that
    # happened to coincide, rather than the one number benchmark.py recorded and
    # reused. The figure says nothing about the collapse: `fit` not responding to
    # n_jobs is a property of the metric, not a caveat about these curves, and it
    # is documented in this module's docstring rather than repeated in every
    # title.
    if len(groups) > 1 and _identical_across_series(groups, x_col, metric):
        groups = groups[:1]
    multi = len(groups) > 1

    if multi:
        fig, (ax_t, ax_s) = plt.subplots(
            2, 1, figsize=(9.5, 9.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08})
    else:
        fig, ax_t = plt.subplots(figsize=(9, 6.5))
        ax_s = None
    # Whichever panel is at the bottom owns the shared x axis furniture.
    ax_bottom = ax_s if multi else ax_t

    if multi:
        texts = _plot_times_multi(ax_t, groups, x_col, metric, show_times)
        _plot_speedup(ax_s, groups, x_col, metric)
    else:
        texts = _plot_times_single(ax_t, groups[0][1], x_col, metric, show_times)

    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_ylabel("time [s]")
    ax_t.grid(True, which="both", ls=":", alpha=0.4)
    # The labels sit above/below the markers, so leave head- and footroom -- more
    # of it with several curves, where every curve contributes its own row of
    # labels and _resolve_overlaps spreads them further apart. Without the time
    # labels there is nothing to make room for, beyond the speed-up badge the
    # single-curve layout parks below its lower point.
    if show_times:
        ax_t.margins(x=0.14, y=0.42 if multi else 0.26)
    else:
        ax_t.margins(x=0.14, y=0.08 if multi else 0.20)

    # One tick per measured size instead of the log axis' decade ticks, which
    # would silently drop the in-between points (500 columns, 500 000 rows, ...).
    # The panels share x, so setting this on the bottom one covers all of them.
    ticks = np.unique(sub[x_col].to_numpy())
    ax_bottom.set_xlabel(x_col)
    ax_bottom.set_xticks(ticks)
    ax_bottom.set_xticklabels([f"{int(t):,}" for t in ticks])
    ax_bottom.set_xticks([], minor=True)

    # e.g. "fit_transform at 100 columns — normal, 1000 quantiles"; the
    # underscore has to be escaped or mathtext reads it as a subscript. A
    # distribution-independent metric names no distribution, because there is
    # none to name: its figure covers all of them, and the filename already says
    # so by carrying no distribution either.
    name = metric.name.replace("_", r"\_")
    row = sub.iloc[0]
    title = rf"$\mathtt{{{name}}}$  at {fixed_label} — "
    if metric.per_distribution:
        title += f"{row['output_distribution']}, "
    title += f"{int(row['n_quantiles']):,} quantiles"
    ax_t.set_title(title, fontsize=TIME_TITLE_FONTSIZE, fontweight="bold", pad=12)

    _resolve_overlaps(fig, texts)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _make_time_figures(sub, x_col, fixed_label, plots_dir, stem, show_times):
    """One figure per (parameter set x output distribution x metric) for this slice."""
    for metric in TIME_METRICS:
        for set_name, set_sub, _covered in figure_groups(sub, metric.per_distribution):
            parts = [stem, set_name, metric.name] if set_name else [stem, metric.name]
            _make_time_figure(set_sub, x_col, fixed_label, metric,
                              plots_dir / f"{'_'.join(parts)}.png", show_times)


def plot_times(df, plots_dir, show_times):
    """Write the whole timing family, or report why there is nothing to write."""
    if not np.isfinite(df[TIME_PROBE_COL]).any():
        print("[plot] no timings in this CSV (measure_time was false) "
              "-- skipping the timing figures")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # rows vs time, one figure per fixed column count and metric.
    for c in sorted(df["cols"].unique()):
        sub = df[df["cols"] == c]
        if sub["rows"].nunique() > 1:
            _make_time_figures(sub, "rows", f"{c:,} columns", plots_dir,
                               f"rows_vs_time_cols{c}", show_times)

    # cols vs time, one figure per fixed row count and metric.
    for r in sorted(df["rows"].unique()):
        sub = df[df["rows"] == r]
        if sub["cols"].nunique() > 1:
            _make_time_figures(sub, "cols", f"{r:,} rows", plots_dir,
                               f"cols_vs_time_rows{r}", show_times)


# --------------------------------------------------------------------------
# Correctness figures
# --------------------------------------------------------------------------

STRATUM, SKLEARN = "C0", "C1"

# When the two series agree the curves land on top of each other, which would
# hide the first one entirely. It is therefore drawn as a thick translucent halo
# and the second as a thin line on top: overlap shows up as an orange line inside
# a blue band, divergence as two separate curves.
STRATUM_STYLE = {"color": STRATUM, "lw": 4.0, "alpha": 0.45, "markersize": 11, "zorder": 2}
SKLEARN_STYLE = {"color": SKLEARN, "lw": 1.6, "markersize": 6, "zorder": 3}

DIFF_LABEL = "difference [absolute]"
ERR_LABEL = "reconstruction error [absolute]"

PASS_COLOR, FAIL_COLOR = "#2e7d32", "#c0392b"


class Comparison(NamedTuple):
    """One stratum-vs-scikit-learn comparison, drawn as a max and a mean curve."""
    label: str
    max_col: str
    mae_col: str


# The four comparisons, in the order the estimator performs them: the fitted
# state first, then the two forward paths, then the inverse.
COMPARISONS = [
    Comparison("quantiles_", "quantiles_max_abs_diff", "quantiles_mae"),
    Comparison("transform", "transform_max_abs_diff", "transform_mae"),
    Comparison("fit_transform", "fit_transform_max_abs_diff", "fit_transform_mae"),
    Comparison("inverse_transform", "inverse_max_abs_diff", "inverse_mae"),
]
COMPARISON_COLORS = ("C0", "C2", "C4", "C6")

# The pass/fail matrix. Each entry maps a row label to a predicate over the CSV.
# Kept as predicates rather than plain column names because not every check is
# stored as a boolean: NaN agreement is a fraction that has to be exactly 1.
# Every cell in a correctness run feeds the kernel float64, so taking the native
# path is itself a check -- a cell that fell back compared scikit-learn against
# itself and its green agreement columns would mean nothing.
CHECKS = [
    ("quantiles_ shape", lambda d: d["quantiles_shape_match"].astype(bool)),
    ("quantiles_ monotonic", lambda d: d["stratum_quantiles_monotonic"].astype(bool)),
    ("NaNs in place", lambda d: d["transform_nan_agree"].astype(float) == 1.0),
    ("native fast path", lambda d: d["stratum_fastpath"].astype(bool)),
]


def _set_title(ax, call, phrase, tail):
    """Bold, smallish, metric call in typewriter -- the timing-figure title style."""
    ax.set_title(rf"$\mathtt{{{call}}}$ {phrase} — {tail}",
                 fontsize=CORRECTNESS_TITLE_FONTSIZE, fontweight="bold", pad=12)


def _linthresh(sub, columns):
    """Where the symlog y axis switches from linear to logarithmic.

    One decade below the smallest positive value plotted, so every measured point
    lands in the log region and only the zeros (and any negative round-off) fall
    into the linear window around the origin. Exact zeros are the normal case
    here -- two implementations that agree bit-for-bit on a cell produce one --
    which is precisely why these axes are symlog and not log.
    """
    values = np.concatenate([sub[c].to_numpy(dtype=float) for c in columns])
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1e-18
    return float(10.0 ** np.floor(np.log10(positive.min())))


def _shade_fallbacks(ax, sub, x):
    """Shade the cells that did not run the native kernel.

    A fallback cell compares scikit-learn against scikit-learn, so its agreement
    is trivially perfect and says nothing about the kernel. Shading keeps that
    visible instead of letting a flat zero line read as a strong result.
    """
    fell_back = ~sub["stratum_fastpath"].astype(bool).to_numpy()
    for i, xi in enumerate(x[fell_back]):
        ax.axvspan(xi - 0.4, xi + 0.4, color="0.6", alpha=0.18, zorder=0,
                   label="fell back to scikit-learn" if i == 0 else None)


def make_agreement_figure(sub, name, out_path):
    """Max and mean |Δ| between the two implementations, over the grid."""
    sub, x, labels = cell_labels(sub)
    fig, ax = plt.subplots(figsize=(10, 6))

    for comp, color in zip(COMPARISONS, COMPARISON_COLORS):
        ax.plot(x, sub[comp.max_col], ls="-", marker="o", color=color, lw=2.0,
                markersize=7, label=f"{comp.label} max")
        ax.plot(x, sub[comp.mae_col], ls="--", marker="o", mfc="none", color=color,
                lw=1.4, markersize=6, alpha=0.85, label=f"{comp.label} mean")

    _shade_fallbacks(ax, sub, x)

    ax.set_yscale("symlog",
                  linthresh=_linthresh(sub, [c.max_col for c in COMPARISONS]
                                       + [c.mae_col for c in COMPARISONS]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("rows x cols")
    ax.set_ylabel(DIFF_LABEL)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, ncol=2)
    _set_title(ax, "fit/transform", "agreement with scikit-learn",
               f"{name} parameters")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def make_roundtrip_figure(sub, name, out_path):
    """inverse_transform(transform(X)) vs X, each implementation on its own."""
    sub, x, labels = cell_labels(sub)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, sub["roundtrip_max_abs_diff_stratum"], ls="-", marker="o",
            label="stratum", **STRATUM_STYLE)
    ax.plot(x, sub["roundtrip_max_abs_diff_sklearn"], ls="-", marker="s",
            label="scikit-learn", **SKLEARN_STYLE)

    _shade_fallbacks(ax, sub, x)

    ax.set_yscale("symlog", linthresh=_linthresh(
        sub, ["roundtrip_max_abs_diff_stratum", "roundtrip_max_abs_diff_sklearn"]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("rows x cols")
    ax.set_ylabel(ERR_LABEL)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=10)
    # Deliberately not "agreement": the y value is each implementation's own
    # information loss, and the claim being made is that the two lose the same.
    _set_title(ax, "inverse\\_transform", "round trip against the original data",
               f"{name} parameters")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def make_checks_figure(sub, name, out_path):
    """Pass/fail matrix: one row per check, one column per grid cell."""
    sub, x, labels = cell_labels(sub)
    matrix = np.vstack([predicate(sub).to_numpy(dtype=bool) for _, predicate in CHECKS])

    # Height follows the check count so the cells stay square-ish whatever the
    # grid size, rather than stretching into bands on a small sweep.
    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 + 0.62 * len(x)),
                                    1.4 + 0.42 * len(CHECKS)))
    ax.imshow(matrix, cmap=ListedColormap([FAIL_COLOR, PASS_COLOR]), vmin=0, vmax=1,
              aspect="auto")

    # A tick per cell, and a symbol in every cell: the colours alone would not
    # survive a greyscale print or a colour-vision deficiency, and this figure is
    # the one that carries the verdict.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "✓" if matrix[i, j] else "✗", ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(CHECKS)))
    ax.set_yticklabels([label for label, _ in CHECKS])
    ax.set_xlabel("rows x cols")
    # Thin white gridlines between the cells, drawn on the minor ticks so they
    # sit on the boundaries rather than through the middle of each square.
    ax.set_xticks(np.arange(-0.5, len(x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CHECKS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)

    n_fail = int((~matrix).sum())
    verdict = "all checks pass" if n_fail == 0 else f"{n_fail} check(s) FAILED"
    _set_title(ax, "fit/transform", f"correctness checks — {verdict}",
               f"{name} parameters")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


CORRECTNESS_FIGURES = [
    ("agreement", make_agreement_figure),
    ("roundtrip", make_roundtrip_figure),
    ("checks", make_checks_figure),
]


def plot_correctness(df, plots_dir):
    """Write the whole correctness family; every run populates these columns."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stem, make in CORRECTNESS_FIGURES:
        for name, sub, _covered in figure_groups(df):
            if len(sub) > 1:
                make(sub, name, plots_dir / f"{stem}_{name}.png")


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
    ap.add_argument(
        "--showtimes", action="store_true",
        help="label every point with its exact time (default: off). The speed-up "
             "badges and bars are drawn either way.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plots_dir = Path(args.plots_dir)
    correctness_dir = (Path(args.correctness_dir) if args.correctness_dir
                       else plots_dir / "correctness")

    if args.only != "correctness":
        with plt.rc_context(TIME_RC):
            plot_times(df, plots_dir, args.showtimes)
    if args.only != "time":
        with plt.rc_context(CORRECTNESS_RC):
            plot_correctness(df, correctness_dir)


if __name__ == "__main__":
    main()
