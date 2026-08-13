"""Checks for the plotting conventions of section 3 of CLAUDE.md.

Figures cannot be tested for looking right, but the rules they must follow are
mechanical: both file formats, the fixed colour per method, a log axis that a
single outlier cannot stretch, and no silent disappearance of a run.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import figures
from src.figures import (
    COLORS,
    GAP_FLOOR,
    band_figure,
    convergence_figure,
    convergence_pair,
    cost_figure,
    family_of,
    save_figure,
    slide_copy,
    spectrum_figure,
    style_for,
)
from src.record import RunRecord


def make_record(label="GD, t = 1/L", direction="GD", gaps=None, n=40, status="converged"):
    gaps = gaps if gaps is not None else list(np.logspace(0, -12, n))
    return RunRecord(
        label=label,
        status=status,
        direction_label=direction,
        step_label="t = 1/L",
        iters=list(range(len(gaps))),
        gaps=list(gaps),
        gnorms=[0.0] * len(gaps),
        times=[0.01 * i for i in range(len(gaps))],
        rows_seen=[100 * i for i in range(len(gaps))],
        fevals=[i for i in range(len(gaps))],
        rows_per_epoch=100,
        is_stochastic=False,
    )


# ------------------------------------------------------------------ colours


@pytest.mark.parametrize(
    "label, family",
    [
        ("GD", "GD"),
        ("SGD (B = 256)", "SGD"),
        ("AGD (beta from t, mu)", "AGD"),
        ("Newton", "Newton"),
        ("Newton (Hessian reused)", "Newton"),
        ("Ridge (solver='auto')", "library"),
        ("SGDRegressor (defaults)", "library"),
        ("something else", "other"),
    ],
)
def test_family_of_maps_labels_onto_the_fixed_palette(label, family):
    assert family_of(label) == family
    assert family in COLORS


def test_one_method_keeps_one_colour_across_configurations():
    """Section 3 of CLAUDE.md: the colour identifies the method, not the run."""
    styles = [style_for(make_record(direction="Newton"), i) for i in range(4)]
    assert len({s["color"] for s in styles}) == 1
    assert len({s["linestyle"] for s in styles}) == 4


def test_different_methods_get_different_colours():
    gd = style_for(make_record(direction="GD"), 0)
    newton = style_for(make_record(direction="Newton"), 0)
    assert gd["color"] != newton["color"]


# -------------------------------------------------------------------- saving


def test_save_figure_writes_both_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    written = save_figure(fig, "smoke", tmp_path)

    assert {p.suffix for p in written} == {".pdf", ".png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_convergence_pair_emits_one_figure_per_axis(tmp_path):
    written = convergence_pair([make_record()], "group", out_dir=tmp_path)
    assert set(written) == {"iters", "time"}
    assert (tmp_path / "group_iters.pdf").exists()
    assert (tmp_path / "group_time.pdf").exists()


def test_slide_copy_keeps_the_file_name_and_changes_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "SLIDE_DIR", tmp_path / "slides")
    records = [make_record(label="a"), make_record(label="b")]
    written = slide_copy(records, "group", keep=["a"])

    assert (tmp_path / "slides" / "group_iters.pdf").exists()
    assert all("slides" in str(p) for paths in written.values() for p in paths)
    # The style must be restored, or every later report figure comes out large.
    assert plt.rcParams["font.size"] == 10


# --------------------------------------------------------------------- axes


def test_axis_name_is_validated():
    with pytest.raises(ValueError, match="axis"):
        convergence_figure([make_record()], axis="nonsense")


@pytest.mark.parametrize("axis", ["iters", "time", "rows", "fevals"])
def test_every_axis_draws(axis):
    fig = convergence_figure([make_record()], axis=axis)
    assert fig.axes[0].get_xlabel()
    plt.close(fig)


def test_exact_zero_gap_survives_the_log_axis():
    """Newton lands on w* exactly, so the gap really is 0 and must be clipped."""
    fig = convergence_figure([make_record(gaps=[1.0, 1e-8, 0.0])])
    line = fig.axes[0].lines[0]
    assert line.get_ydata()[-1] == GAP_FLOOR
    plt.close(fig)


def test_one_outlier_cannot_stretch_the_axis_over_forty_decades():
    """A diverged run at 1e8 next to Newton at 1e-30 would flatten everything."""
    diverged = make_record(label="GD (t = 2.1/L)", gaps=[1.0, 1e4, 1e8], status="diverged")
    exact = make_record(label="Newton", direction="Newton", gaps=[1.0, 1e-30])
    fig = convergence_figure([diverged, exact], max_decades=20)

    bottom, top = fig.axes[0].get_ylim()
    assert np.log10(top / bottom) == pytest.approx(20, abs=0.5)
    plt.close(fig)


def test_explicit_limits_win_over_the_automatic_window():
    fig = convergence_figure([make_record()], ylim=(1e-9, 1e1))
    assert fig.axes[0].get_ylim() == (1e-9, 1e1)
    plt.close(fig)


def test_diverged_runs_are_labelled_not_dropped():
    record = make_record(label="GD (t = 2.1/L)", gaps=[1.0, 1e6], status="diverged")
    fig = convergence_figure([record])
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert any("diverged" in str(label) for label in labels)
    plt.close(fig)


# ------------------------------------------------------------ single points


def test_a_library_result_is_drawn_as_a_marker():
    """One point drawn as a line is invisible, which is how a solver goes missing."""
    single = RunRecord(label="Ridge", status="converged", direction_label="library",
                       iters=[1], gaps=[1e-28], gnorms=[0.0], times=[0.05],
                       rows_seen=[0], fevals=[0])
    fig = convergence_figure([make_record(), single])
    markers = [line.get_marker() for line in fig.axes[0].lines]
    assert "o" in markers
    plt.close(fig)


def test_library_results_get_distinct_markers():
    singles = [
        RunRecord(label=f"Ridge {i}", status="converged", direction_label="library",
                  iters=[1], gaps=[10.0 ** -i], gnorms=[0.0], times=[0.05],
                  rows_seen=[0], fevals=[0])
        for i in range(1, 5)
    ]
    fig = convergence_figure(singles)
    markers = [line.get_marker() for line in fig.axes[0].lines if line.get_linestyle() == "None"]
    assert len(set(markers)) == 4
    plt.close(fig)


def test_a_point_below_the_window_is_flagged_at_the_frame():
    below = RunRecord(label="Ridge", status="converged", direction_label="library",
                      iters=[1], gaps=[1e-30], gnorms=[0.0], times=[0.05],
                      rows_seen=[0], fevals=[0])
    loud = make_record(label="GD", gaps=[1e8, 1e6])
    fig = convergence_figure([loud, below], max_decades=10)

    bottom, _ = fig.axes[0].get_ylim()
    flags = [line for line in fig.axes[0].lines if line.get_marker() == "v"]
    assert flags, "a clipped point must leave a mark at the axis"
    assert flags[0].get_ydata()[0] == pytest.approx(bottom)
    plt.close(fig)


# ------------------------------------------------------------- other panels


def test_band_figure_draws_a_median_and_a_ribbon():
    bands = {
        "constant": {
            "x_iters": [0, 1, 2],
            "x_time": [0.0, 0.1, 0.2],
            "x_epochs": [0.0, 1.0, 2.0],
            "median": [1.0, 1e-2, 1e-3],
            "low": [0.5, 5e-3, 5e-4],
            "high": [2.0, 2e-2, 2e-3],
            "n_seeds": 5,
        }
    }
    fig = band_figure(bands, axis="epochs")
    assert fig.axes[0].collections, "the min-max ribbon is missing"
    assert "5 seeds" in fig.axes[0].lines[0].get_label()
    plt.close(fig)


def test_cost_figure_reports_probes_per_iteration():
    record = make_record()
    record.fevals = [3 * i for i in range(len(record.iters))]
    fig = cost_figure([record])
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), 3.0)
    assert "evaluation" in fig.axes[0].get_ylabel().lower()
    plt.close(fig)


def test_spectrum_figure_marks_the_regularisation_floor():
    fig = spectrum_figure(np.logspace(1, -6, 40), lam=1e-3)
    horizontals = [line for line in fig.axes[0].lines if len(set(line.get_ydata())) == 1]
    assert horizontals
    plt.close(fig)
