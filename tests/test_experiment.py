"""Checks for the declarative grid runner.

The grids themselves are only smoke-tested: what matters is that every builder
produces specs that run, that the product really is a product, and that a group
already on disk is not silently recomputed.
"""

import numpy as np
import pytest

from src.direction import NewtonStep, SteepestDescent
from src.experiment import (
    BUILDERS,
    GROUPS,
    ExperimentGroup,
    RunSpec,
    best_of,
    headline_group,
    product,
    run_group,
    run_spec,
    seed_band,
    summary_table,
)
from src.objective import RidgeObjective
from src.stepsize import Armijo, Fixed


@pytest.fixture(scope="module")
def objective() -> RidgeObjective:
    rng = np.random.default_rng(0)
    n, d = 5000, 15
    raw = rng.standard_normal((n, d)) @ np.diag(np.logspace(0, -1, d))
    X = (raw - raw.mean(0)) / raw.std(0)
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n)
    return RidgeObjective(X, y - y.mean(), lam=1e-2)


# --------------------------------------------------------------- the product


def test_product_covers_every_pairing():
    directions = {"GD": SteepestDescent, "Newton": NewtonStep}
    steps = {"fixed": lambda _: Fixed(0.1), "armijo": lambda _: Armijo()}
    specs = product(directions, steps)

    assert len(specs) == 4
    assert {s.label for s in specs} == {
        "GD, fixed", "GD, armijo", "Newton, fixed", "Newton, armijo",
    }
    assert all(s.params["direction"] in directions for s in specs)


def test_product_passes_common_settings_through():
    specs = product({"GD": SteepestDescent}, {"fixed": lambda _: Fixed(0.1)}, max_iter=7, repeats=3)
    assert specs[0].max_iter == 7 and specs[0].repeats == 3


# ------------------------------------------------------------------ running


def test_run_spec_records_the_parameters_it_was_given(objective):
    spec = RunSpec(
        make_direction=SteepestDescent,
        make_step=lambda o: Fixed(1 / o.L),
        max_iter=20,
        params={"t": "1/L", "note": "smoke"},
    )
    record = run_spec(objective, spec)
    assert record.meta["t"] == "1/L"
    assert record.meta["note"] == "smoke"
    assert record.meta["repeats"] == 1


def test_repetitions_keep_one_consistent_record(objective):
    """Repeats exist for the clock; the trajectory must not become an average."""
    spec = RunSpec(
        make_direction=SteepestDescent,
        make_step=lambda o: Fixed(1 / o.L),
        max_iter=30,
        repeats=3,
    )
    record = run_spec(objective, spec)
    assert len(record.meta["times_all"]) == 3
    assert record.total_time in record.meta["times_all"]
    # The median repetition, so the reported time is one of the measured ones.
    assert sorted(record.meta["times_all"])[1] == record.total_time


def test_a_fresh_direction_is_built_per_repetition(objective):
    """Reusing a stateful Nesterov across repeats would corrupt the second run."""
    from src.direction import Nesterov

    spec = RunSpec(
        make_direction=lambda o: Nesterov(o, "strongly_convex"),
        make_step=lambda o: Fixed(1 / o.L),
        max_iter=40,
        repeats=3,
    )
    record = run_spec(objective, spec)
    single = run_spec(
        objective,
        RunSpec(make_direction=lambda o: Nesterov(o, "strongly_convex"),
                make_step=lambda o: Fixed(1 / o.L), max_iter=40, repeats=1),
    )
    assert record.gaps == single.gaps


# ------------------------------------------------------------------ caching


def test_group_is_written_once_and_loaded_afterwards(objective, tmp_path):
    calls = []

    def build(obj):
        calls.append(obj)
        return [RunSpec(SteepestDescent, lambda o: Fixed(1 / o.L), max_iter=10)]

    group = ExperimentGroup("cached", "smoke", build)
    first = run_group(group, objective, tmp_path, verbose=False)
    second = run_group(group, objective, tmp_path, verbose=False)

    assert len(calls) == 1, "the second call must load, not rebuild"
    assert (tmp_path / "cached.json").exists()
    assert [r.gaps for r in first] == [r.gaps for r in second]


def test_force_reruns_a_cached_group(objective, tmp_path):
    calls = []

    def build(obj):
        calls.append(obj)
        return [RunSpec(SteepestDescent, lambda o: Fixed(1 / o.L), max_iter=10)]

    group = ExperimentGroup("cached", "smoke", build)
    run_group(group, objective, tmp_path, verbose=False)
    run_group(group, objective, tmp_path, force=True, verbose=False)
    assert len(calls) == 2


# ------------------------------------------------------------- the real grids


@pytest.mark.parametrize("name", list(BUILDERS))
def test_every_builder_produces_runnable_specs(objective, name):
    _, build = BUILDERS[name]
    specs = build(objective)
    assert specs, name

    for spec in specs:
        assert spec.make_direction(objective) is not None
        assert spec.make_step(objective) is not None

    # Run the first one briefly: a factory that closes over a loop variable by
    # reference rather than by default argument only shows up when called.
    spec = specs[0]
    spec.max_iter, spec.max_epochs, spec.repeats = 5, None, 1
    record = run_spec(objective, spec)
    assert record.gaps and np.isfinite(record.gaps[0])


def test_step_fixed_labels_are_distinct(objective):
    """Late binding in a lambda would give every run the last label."""
    _, build = BUILDERS["step-fixed"]
    labels = [spec.make_step(objective).label for spec in build(objective)]
    assert len(set(labels)) == len(labels)
    assert "t = 2.1/L" in labels  # the divergence probe must survive


def test_group_names_match_the_figure_names():
    assert [g.name for g in GROUPS] == list(BUILDERS)
    assert all(g.name == g.name.lower().replace(" ", "-") for g in GROUPS)


# ------------------------------------------------------- picking and reporting


def _finished(gap_curve, label="x", time_scale=1.0):
    from src.record import RunRecord

    return RunRecord(
        label=label,
        status="converged",
        iters=list(range(len(gap_curve))),
        gaps=list(gap_curve),
        gnorms=[0.0] * len(gap_curve),
        times=[i * time_scale for i in range(len(gap_curve))],
        rows_seen=[0] * len(gap_curve),
        fevals=[0] * len(gap_curve),
    )


def test_best_of_prefers_whichever_reaches_the_threshold_first():
    slow = _finished([1e0, 1e-3, 1e-7], "slow", time_scale=10.0)
    fast = _finished([1e0, 1e-3, 1e-7], "fast", time_scale=1.0)
    assert best_of([slow, fast]).label == "fast"
    # By iterations the two are identical, so the first one wins the tie.
    assert best_of([slow, fast], key="iters").label == "slow"


def test_best_of_ranks_runs_that_never_arrive_last():
    stuck = _finished([1e0, 1e-1, 1e-2], "stuck", time_scale=0.1)
    arrives = _finished([1e0, 1e-3, 1e-7], "arrives", time_scale=100.0)
    assert best_of([stuck, arrives]).label == "arrives"


def test_best_of_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="key"):
        best_of([_finished([1.0])], key="epochs")


def test_summary_table_reports_both_axes():
    rows = summary_table([_finished([1e0, 1e-3, 1e-7], "a", 2.0)])
    assert rows[0]["iters_to_1e-06"] == 2
    assert rows[0]["seconds_to_1e-06"] == 4.0
    assert rows[0]["final_gap"] == 1e-7


def test_seed_band_collapses_seeds_into_a_median_and_a_range():
    records = []
    for seed, offset in enumerate([1.0, 2.0, 3.0]):
        record = _finished([offset, offset / 10, offset / 100], f"seed {seed}")
        record.meta["seed_group"] = "constant"
        records.append(record)

    bands = seed_band(records, "seed_group")
    assert set(bands) == {"constant"}
    band = bands["constant"]
    assert band["n_seeds"] == 3
    assert band["median"][0] == pytest.approx(2.0)
    assert band["low"][0] == pytest.approx(1.0)
    assert band["high"][0] == pytest.approx(3.0)


def test_headline_group_forces_repetitions_because_it_reports_time(objective):
    chosen = {"GD": RunSpec(SteepestDescent, lambda o: Fixed(1 / o.L), max_iter=5, repeats=1)}
    specs = headline_group(chosen).build(objective)
    assert specs[0].repeats >= 3
    assert headline_group(chosen).name == "headline"
