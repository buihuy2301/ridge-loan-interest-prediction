"""Plotting, with the conventions of section 3 of CLAUDE.md applied in one place.

Every comparison leaves the same pair of figures, one against iterations and one
against wall-clock seconds, because the whole point of the exercise is that the
ranking can differ between the two. The colour a method gets is fixed here and
nowhere else, so a curve labelled Newton is the same red in chapter 7 as in the
summary chapter.

All text on the axes is English, per section 1 of CLAUDE.md; the discussion of
what the figures show is Vietnamese and lives in the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np

from .record import RunRecord

FIGURE_DIR = Path("results/figures")
SLIDE_DIR = FIGURE_DIR / "slides"

# One colour per method family, used throughout the report and the slides.
COLORS = {
    "GD": "#1f77b4",
    "SGD": "#ff7f0e",
    "AGD": "#2ca02c",
    "Newton": "#d62728",
    "library": "#7f7f7f",
    "other": "#9467bd",
}

# A quadratic objective lets Newton land exactly on w*, so the gap really is
# 0.0 and has no place on a log axis. Curves are clipped here instead, which
# draws them running into the bottom of the frame.
GAP_FLOOR = 1e-32

X_AXES = {
    "iters": ("Iteration $k$", "iters"),
    "time": ("Wall-clock time (s)", "times"),
    "epochs": ("Epochs", None),
    "rows": ("Rows of $X$ accessed", "rows_seen"),
    "fevals": ("Objective evaluations", "fevals"),
}


def use_report_style() -> None:
    """Sizes chosen so the text stays readable when the PDF is dropped into Beamer."""
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
        }
    )


def use_slide_style() -> None:
    """Fewer, thicker curves and larger text for the projector."""
    use_report_style()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.4,
        }
    )


use_report_style()


# ------------------------------------------------------------------- colours


def family_of(label: str) -> str:
    """Map a record's direction label onto one of the fixed colour families."""
    head = label.split()[0].split(",")[0] if label else ""
    if head in COLORS:
        return head

    lowered = label.lower()
    # Library estimators are tested first on purpose: "SGDRegressor" starts with
    # "sgd", and colouring a scikit-learn baseline like our own mini-batch SGD
    # would defeat the point of a fixed palette.
    if any(key in lowered for key in ("ridge", "sklearn", "regressor", "regression")):
        return "library"
    for family in ("newton", "sgd", "agd", "gd"):
        if lowered.startswith(family):
            return "Newton" if family == "newton" else family.upper()
    return "other"


DASHES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def style_for(record: RunRecord, index: int) -> dict:
    """Colour by family, then dashes to separate configurations within a family."""
    return {
        "color": COLORS[family_of(record.direction_label or record.label)],
        "linestyle": DASHES[index % len(DASHES)],
    }


# --------------------------------------------------------------------- saving


def save_figure(fig: plt.Figure, name: str, out_dir: Path | str = FIGURE_DIR) -> list[Path]:
    """Write PDF for LaTeX and PNG for quick viewing, both tightly cropped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 150})):
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


# ------------------------------------------------------------------- curves


def _series(record: RunRecord, axis: str) -> np.ndarray:
    if axis == "epochs":
        return np.asarray(record.epochs)
    attribute = X_AXES[axis][1]
    return np.asarray(getattr(record, attribute), dtype=float)


def _gaps(record: RunRecord) -> np.ndarray:
    return np.maximum(np.asarray(record.gaps, dtype=float), GAP_FLOOR)


def convergence_figure(
    records: Sequence[RunRecord],
    axis: str = "iters",
    title: str | None = None,
    ylabel: str = r"$f(w_k) - f^*$",
    ylim: tuple[float, float] | None = None,
    max_decades: float = 20.0,
    annotate_status: bool = True,
) -> plt.Figure:
    """One panel of suboptimality on a log scale.

    Diverging runs are marked in the legend rather than dropped: the run with
    t > 2/L is deliberate, and a reader who cannot see it will wonder why the
    grid contains a value nobody plotted.

    Two things this has to handle that a plain semilogy does not. A library
    result is a single point rather than a trajectory, and a one-point line
    draws nothing, so those are marked instead. And when a diverging run at 1e8
    shares the frame with Newton at machine precision, autoscaling spans nearly
    forty decades and flattens everything; `max_decades` caps the window below
    the largest value plotted.
    """
    if axis not in X_AXES:
        raise ValueError(f"axis must be one of {sorted(X_AXES)}, got {axis!r}")

    fig, ax = plt.subplots()
    highest = 0.0
    singles: list[tuple[np.ndarray, np.ndarray, str]] = []

    for index, record in enumerate(records):
        label = record.label
        if annotate_status and record.status in ("diverged", "line_search_failed"):
            label = f"{label} [{record.status}]"

        x, y = _series(record, axis), _gaps(record)
        highest = max(highest, float(np.max(y)))
        style = style_for(record, index)
        if len(x) <= 1:
            # A closed-form solver reports one point, not a curve. Markers vary
            # within the family the way dashes do for trajectories.
            marker = MARKERS[len(singles) % len(MARKERS)]
            ax.semilogy(x, y, label=label, marker=marker, markersize=7,
                        linestyle="none", color=style["color"])
            singles.append((x, y, marker))
        else:
            ax.semilogy(x, y, label=label, **style)

    ax.set_xlabel(X_AXES[axis][0])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif highest > 0.0 and max_decades > 0:
        top = highest * 5
        ax.set_ylim(bottom=max(top * 10.0**-max_decades, GAP_FLOOR), top=top)

    _flag_points_below_axis(ax, singles)
    ax.legend(loc="best", framealpha=0.9)
    return fig


def _flag_points_below_axis(ax: plt.Axes, singles: Sequence[tuple]) -> None:
    """Draw a hollow down-triangle at the frame for any point off the bottom.

    Silently clipping is the dangerous option here: the exact solvers land many
    decades below anything else on the axis, and a reader who cannot see them
    would conclude they were never run.
    """
    bottom, _ = ax.get_ylim()
    for x, y, marker in singles:
        below = np.asarray(y) < bottom
        if below.any():
            ax.plot(
                np.asarray(x)[below],
                np.full(int(below.sum()), bottom),
                marker="v", markersize=9, linestyle="none",
                markerfacecolor="none", markeredgecolor=COLORS["library"],
                clip_on=False, zorder=5,
            )


def convergence_pair(
    records: Sequence[RunRecord],
    name: str,
    title: str | None = None,
    axes: Iterable[str] = ("iters", "time"),
    out_dir: Path | str = FIGURE_DIR,
    **kwargs,
) -> dict[str, list[Path]]:
    """The mandatory pair: the same runs against iterations and against seconds.

    Stochastic groups pass an extra `epochs` axis, because one SGD iteration and
    one full-batch iteration are not comparable units.
    """
    written = {}
    for axis in axes:
        fig = convergence_figure(records, axis=axis, title=title, **kwargs)
        written[axis] = save_figure(fig, f"{name}_{axis}", out_dir)
    return written


def band_figure(
    bands: dict[str, dict],
    axis: str = "iters",
    title: str | None = None,
    ylabel: str = r"$f(w_k) - f^*$",
) -> plt.Figure:
    """Median across seeds with a min-max ribbon, for the stochastic comparisons.

    A single seed of SGD invites the reader to over-read one wiggle; the ribbon
    shows how much of the shape is the schedule and how much is the draw.
    """
    key = {"iters": "x_iters", "time": "x_time", "epochs": "x_epochs"}[axis]
    # Bands are keyed by schedule name, not by method, so they cycle through a
    # palette of their own rather than through the per-method colours.
    palette = [COLORS["SGD"], COLORS["GD"], COLORS["AGD"], COLORS["Newton"], COLORS["other"]]

    fig, ax = plt.subplots()
    for index, (name, band) in enumerate(bands.items()):
        x = np.asarray(band[key], dtype=float)
        style = {"color": palette[index % len(palette)]}
        ax.semilogy(x, np.maximum(band["median"], GAP_FLOOR),
                    label=f"{name} (median of {band['n_seeds']} seeds)", **style)
        ax.fill_between(x, np.maximum(band["low"], GAP_FLOOR),
                        np.maximum(band["high"], GAP_FLOOR), alpha=0.18, **style)

    ax.set_xlabel(X_AXES[axis][0])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    return fig


def cost_figure(
    records: Sequence[RunRecord],
    title: str | None = None,
) -> plt.Figure:
    """Objective evaluations per iteration, the price a line search charges.

    Backtracking can win on iteration count and lose on time, and this is the
    figure that says why. The tail usually rises sharply: once the true decrease
    drops below the resolution of f, every trial step fails the Armijo test.
    """
    fig, ax = plt.subplots()
    for index, record in enumerate(records):
        iters = np.asarray(record.iters, dtype=float)
        fevals = np.asarray(record.fevals, dtype=float)
        per_iteration = np.diff(fevals) / np.maximum(np.diff(iters), 1.0)
        ax.plot(iters[1:], per_iteration, label=record.label, **style_for(record, index))

    ax.set_xlabel(X_AXES["iters"][0])
    ax.set_ylabel("Objective evaluations per iteration")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    return fig


def spectrum_figure(eigenvalues: np.ndarray, lam: float, title: str | None = None) -> plt.Figure:
    """Gram eigenvalues against the regularisation floor.

    The figure behind section 5.6: lambda sets a floor under the spectrum, so it
    sets mu, and therefore it sets kappa. Where that floor sits relative to the
    smallest data eigenvalue decides whether regularisation or the data controls
    the convergence rate.
    """
    fig, ax = plt.subplots()
    values = np.sort(np.asarray(eigenvalues))[::-1]
    ax.semilogy(np.arange(1, len(values) + 1), np.maximum(values, GAP_FLOOR),
                color=COLORS["GD"], label=r"eigenvalues of $\frac{1}{n}X^\top X$")
    ax.axhline(lam, color=COLORS["Newton"], linestyle="--",
               label=rf"$\lambda = {lam:.4g}$")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    return fig


def lambda_figure(rows: Sequence[dict], title: str | None = None) -> plt.Figure:
    """Two axes on one frame: how lambda moves kappa, and how it moves test error.

    The trade of section 5.6. The lambda that converges fastest is rarely the
    lambda that predicts best, and a single figure makes the gap between the two
    hard to miss.
    """
    fig, ax = plt.subplots()
    lams = [row["lam"] for row in rows]
    ax.loglog(lams, [row["kappa"] for row in rows], "o-", color=COLORS["GD"],
              label=r"$\kappa$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Condition number $\kappa$", color=COLORS["GD"])
    ax.tick_params(axis="y", labelcolor=COLORS["GD"])

    twin = ax.twinx()
    twin.semilogx(lams, [row["rmse_test"] for row in rows], "s--", color=COLORS["Newton"],
                  label="test RMSE")
    twin.set_ylabel("Test RMSE", color=COLORS["Newton"])
    twin.tick_params(axis="y", labelcolor=COLORS["Newton"])
    twin.grid(False)

    if title:
        ax.set_title(title)
    return fig


# --------------------------------------------------------------- slide copies


def slide_copy(
    records: Sequence[RunRecord],
    name: str,
    keep: Sequence[str] | None = None,
    title: str | None = None,
    axes: Iterable[str] = ("iters", "time"),
) -> dict[str, list[Path]]:
    """Redraw a figure for the slides, keeping the same file name.

    `\\resultgraphic` in preamble.tex looks in results/figures/slides first, so a
    slide version overrides the report version without any change to the source
    of the slide. Fewer curves is usually the only difference that matters.
    """
    selected = [r for r in records if keep is None or r.label in keep]
    use_slide_style()
    try:
        return convergence_pair(selected, name, title=title, axes=axes, out_dir=SLIDE_DIR)
    finally:
        use_report_style()
