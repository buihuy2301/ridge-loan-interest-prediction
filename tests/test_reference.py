"""Checks for the scikit-learn comparison, above all the objective conversion.

Getting alpha wrong raises no error: both sides solve cleanly and the comparison
is simply between two different problems. So the conversion is asserted here in
both directions, including the failure that a plausible-looking mistake gives.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.objective import RidgeObjective
from src.record import load_records
from src.reference import (
    baseline_table,
    build_baselines,
    ridge_alpha,
    run_baselines,
    save_baselines,
    sgd_alpha,
    verify_alpha_conversion,
)


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    n, d = 3000, 12
    X = rng.standard_normal((n, d))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n)
    objective = RidgeObjective(X, y - y.mean(), lam=1e-2)

    X_test = rng.standard_normal((500, d))
    X_test = (X_test - X_test.mean(0)) / X_test.std(0)
    y_test = X_test @ objective.w_star + 0.3 * rng.standard_normal(500)
    return objective, X_test, y_test - y_test.mean()


# ------------------------------------------------------------- the constants


def test_the_two_estimators_need_different_constants(problem):
    """Ridge drops the 1/n and the 1/2; SGDRegressor keeps both."""
    objective, *_ = problem
    assert ridge_alpha(objective) == objective.lam * objective.n
    assert sgd_alpha(objective) == objective.lam
    assert ridge_alpha(objective) != sgd_alpha(objective)


def test_conversion_reproduces_the_closed_form(problem):
    objective, *_ = problem
    report = verify_alpha_conversion(objective)

    assert report["verified"]
    assert report["relative_w_distance"] < 1e-10
    assert report["gap"] < 1e-20


def test_the_other_constant_would_solve_a_different_problem(problem):
    """The mistake section 6 warns about, made deliberately."""
    objective, *_ = problem
    wrong = Ridge(alpha=sgd_alpha(objective), fit_intercept=False, solver="cholesky")
    wrong.fit(objective.X, objective.y)
    w = np.asarray(wrong.coef_, dtype=np.float64)

    distance = np.linalg.norm(w - objective.w_star) / np.linalg.norm(objective.w_star)
    assert distance > 1e-6, "the two constants must be distinguishable by their solutions"
    assert objective.suboptimality(w) > 1e-12


def test_ridge_alpha_scales_with_the_sample_size(problem):
    """The 1/n in our objective is exactly why alpha has to follow n."""
    objective, *_ = problem
    half = RidgeObjective(objective.X[:1500], objective.y[:1500], objective.lam)
    assert ridge_alpha(half) == pytest.approx(ridge_alpha(objective) / 2)
    assert verify_alpha_conversion(half)["verified"]


# ---------------------------------------------------------------- the table


def test_every_baseline_is_constructed_with_a_converted_alpha(problem):
    objective, *_ = problem
    models = build_baselines(objective)
    assert len(models) >= 5
    for name, model in models.items():
        if name.startswith("Ridge"):
            assert model.alpha == ridge_alpha(objective)
        if name.startswith("SGDRegressor"):
            assert model.alpha == sgd_alpha(objective)


def test_run_baselines_reports_gap_and_time(problem):
    objective, X_test, y_test = problem
    results = run_baselines(objective, X_test, y_test, repeats=2, verbose=False)

    assert results
    by_name = {r.name: r for r in results}
    direct = by_name["Ridge (solver='cholesky')"]
    assert direct.gap < 1e-20, "a direct solver must land on f* to machine precision"
    assert direct.fit_seconds > 0
    assert len(direct.times_all) == 2
    assert direct.rmse_test > 0


def test_a_direct_solver_beats_the_iterative_ones_on_accuracy(problem):
    objective, X_test, y_test = problem
    by_name = {r.name: r for r in run_baselines(objective, X_test, y_test,
                                                repeats=1, verbose=False)}
    assert by_name["Ridge (solver='cholesky')"].gap <= by_name["Ridge (solver='lsqr')"].gap


def test_baseline_table_leads_with_our_closed_form(problem):
    objective, X_test, y_test = problem
    results = run_baselines(objective, X_test, y_test, repeats=1, verbose=False)
    rows = baseline_table(results, objective)

    assert rows[0]["method"] == "closed form (ours)"
    assert rows[0]["gap"] == 0.0
    assert len(rows) == len(results) + 1


def test_results_become_records_the_figures_can_plot(problem, tmp_path):
    objective, X_test, y_test = problem
    results = run_baselines(objective, X_test, y_test, repeats=1, verbose=False)

    record = results[0].to_record()
    assert record.direction_label == "library"
    assert len(record.gaps) == 1, "a library fit is a point, not a trajectory"
    assert record.times == [results[0].fit_seconds]

    path = save_baselines(results, objective, tmp_path / "library.json")
    restored, meta = load_records(path)
    assert len(restored) == len(results)
    assert meta["conversion"]["verified"]
    assert meta["table"][0]["method"] == "closed form (ours)"
