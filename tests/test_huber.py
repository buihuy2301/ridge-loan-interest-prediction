"""Checks on the Huber objective, section 8 of KE_HOACH_TRIEN_KHAI.md carried over
to the second problem.

The strongest check is first: with a threshold far above every residual, the
Huber loss is the squared loss, so the whole class has to reproduce
`RidgeObjective` to machine precision. That one test catches almost every
constant and sign error the derivatives could carry.
"""

import numpy as np
import pytest

from src.huber import HuberObjective
from src.objective import RidgeObjective, SmoothObjective

EPS = 1e-6
LAM = 1e-2


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, d = 400, 12
    X = rng.standard_normal((n, d))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n)
    return X, y - y.mean()


@pytest.fixture(scope="module")
def objective(data) -> HuberObjective:
    # A threshold below the residual scale, so both branches of the loss are
    # exercised. A fixture where every row sits in the quadratic zone would pass
    # every test below while testing nothing about Huber.
    X, y = data
    return HuberObjective(X, y, lam=LAM, delta=0.5)


@pytest.fixture(scope="module")
def probe(objective) -> np.ndarray:
    return np.random.default_rng(7).standard_normal(objective.d)


def test_both_branches_of_the_loss_are_exercised(objective, probe):
    residual = objective.X @ probe - objective.y
    outside = float(np.mean(np.abs(residual) > objective.delta))
    assert 0.05 < outside < 0.95


def test_a_large_threshold_reproduces_the_ridge_objective(data, probe):
    X, y = data
    ridge = RidgeObjective(X, y, lam=LAM)
    huber = HuberObjective(X, y, lam=LAM, delta=1e6)
    assert huber.value(probe) == pytest.approx(ridge.value(probe), rel=1e-12)
    assert np.allclose(huber.gradient(probe), ridge.gradient(probe), rtol=1e-12)
    assert np.allclose(huber.compute_hessian(probe), ridge.compute_hessian(), rtol=1e-12)


def test_gradient_matches_central_differences(objective, probe):
    analytic = objective.gradient(probe)
    for i in range(objective.d):
        shift = np.zeros(objective.d)
        shift[i] = EPS
        numeric = (objective.value(probe + shift) - objective.value(probe - shift)) / (2 * EPS)
        assert abs(numeric - analytic[i]) <= 1e-5 * max(abs(analytic[i]), 1.0)


def test_hessian_matches_differences_of_the_gradient(objective, probe):
    hessian = objective.compute_hessian(probe)
    for i in range(objective.d):
        shift = np.zeros(objective.d)
        shift[i] = EPS
        numeric = (objective.gradient(probe + shift) - objective.gradient(probe - shift)) / (2 * EPS)
        assert np.allclose(numeric, hessian[i], atol=1e-5)


def test_hessian_stays_above_the_regularisation_floor(objective, probe):
    eigenvalues = np.linalg.eigvalsh(objective.compute_hessian(probe))
    assert eigenvalues[0] >= objective.lam - 1e-12


def test_value_and_gradient_agree_with_the_separate_calls(objective, probe):
    value, gradient = objective.value_and_gradient(probe)
    assert value == pytest.approx(objective.value(probe), rel=1e-14)
    assert np.allclose(gradient, objective.gradient(probe), rtol=1e-14)


def test_the_reference_solution_has_a_vanishing_gradient(objective):
    assert np.linalg.norm(objective.gradient(objective.w_star)) <= 1e-13


def test_the_constants_are_the_bounds_the_algorithm_may_assume(objective, data):
    X, y = data
    ridge = RidgeObjective(X, y, lam=LAM)
    # The Huber Hessian never exceeds the Ridge one, so they share an upper bound.
    assert objective.L == pytest.approx(ridge.L, rel=1e-12)
    # Rows outside the quadratic zone contribute nothing, so only lam is certain.
    assert objective.mu == pytest.approx(LAM)
    assert objective.kappa == pytest.approx(objective.L / LAM)
    # Measured at the optimum the constant is tighter than the bound.
    assert objective.mu_at_optimum >= objective.mu


def test_suboptimality_is_never_negative(objective):
    rng = np.random.default_rng(11)
    for _ in range(20):
        w = objective.w_star + 0.1 * rng.standard_normal(objective.d)
        assert objective.suboptimality(w) >= 0.0


def test_suboptimality_agrees_with_the_plain_difference_far_from_the_optimum(objective, probe):
    plain = objective.value(probe) - objective.f_star
    assert objective.suboptimality(probe) == pytest.approx(plain, rel=1e-9)


def test_suboptimality_resolves_below_the_plain_difference(objective):
    """Close to w*, the plain difference is rounding noise and the paired one is not.

    A piecewise-quadratic function's gap near its minimiser is the quadratic form
    of the Hessian, so that form is the truth both estimates are judged against.
    The distance is chosen where subtracting two floats has lost every digit:
    `value(w) - f_star` collapses to exactly zero, while the paired formula still
    lands on the right order of magnitude. The bounds are ratios rather than
    tolerances because at 1e-18 the paired formula carries its own rounding noise,
    measured at about 14 percent here.
    """
    step = 1e-9 * np.ones(objective.d)
    w = objective.w_star + step
    quadratic = 0.5 * float(step @ (objective.compute_hessian(objective.w_star) @ step))

    plain = objective.value(w) - objective.f_star
    assert abs(plain - quadratic) > 0.9 * quadratic

    assert 0.5 * quadratic < objective.suboptimality(w) < 2.0 * quadratic


def test_solve_hessian_freezes_the_factor_at_the_origin(objective):
    """`reuse_factorization=True` is the chord method here, not Newton."""
    rhs = np.ones(objective.d)
    expected = np.linalg.solve(objective.compute_hessian(np.zeros(objective.d)), rhs)
    assert np.allclose(objective.solve_hessian(rhs), expected, rtol=1e-10)
    # A second call must reuse the same factor rather than refactor at a new point.
    assert np.allclose(objective.solve_hessian(rhs), expected, rtol=1e-10)


def test_summary_records_what_the_report_needs(objective):
    summary = objective.summary()
    for key in ("n", "d", "lam", "delta", "L", "mu", "kappa",
                "mu_at_optimum", "f_star", "outside_fraction"):
        assert key in summary
    assert 0.0 <= summary["outside_fraction"] <= 1.0


def test_a_batch_over_every_row_is_the_full_objective(objective, probe):
    batch = objective.batch(np.arange(objective.n))
    value, gradient = batch.value_and_gradient(probe)
    full_value, full_gradient = objective.value_and_gradient(probe)
    # rel alone would not govern here: pytest.approx and np.allclose both fall
    # back to a nonzero absolute floor (1e-12 and 1e-8 respectively) that
    # swallows a wrong-by-1e-9 gradient on quantities of this size, so the
    # floor is pinned to zero to make the relative tolerance the real check.
    assert value == pytest.approx(full_value, rel=1e-12, abs=0.0)
    assert np.allclose(gradient, full_gradient, rtol=1e-12, atol=0.0)
    assert batch.n_rows == objective.n


def test_a_batch_keeps_the_penalty_whole(objective):
    """Scaling the penalty by |B|/n would bias the gradient estimate.

    Checked against the full objective restricted to the same rows: a batch of
    fifty rows must be the fifty-row problem carrying the whole penalty, not one
    fiftieth of it.
    """
    rows = np.arange(50)
    batch = objective.batch(rows)
    same_rows = HuberObjective(
        objective.X[rows].copy(), objective.y[rows].copy(), objective.lam, objective.delta
    )
    w = np.ones(objective.d)
    assert batch.value(w) == pytest.approx(same_rows.value(w), rel=1e-12, abs=0.0)

    penalty = objective.lam / 2 * float(w @ w)
    assert batch.value(w) > penalty


def test_choose_delta_leaves_a_real_fraction_outside_the_quadratic_zone(data):
    from src.huber import choose_delta

    X, y = data
    ridge = RidgeObjective(X, y, lam=LAM)
    delta = choose_delta(ridge)
    outside = float(np.mean(np.abs(X @ ridge.w_star - y) > delta))
    assert 0.05 < outside < 0.4


def test_build_huber_reuses_a_stored_reference(data, tmp_path):
    from src.huber import build_huber

    X, y = data
    path = tmp_path / "huber_reference.json"
    first = build_huber(X, y, lam=LAM, path=path, verbose=False)
    assert path.exists()
    assert first.reference_iters >= 1

    second = build_huber(X, y, lam=LAM, path=path, verbose=False)
    assert second.delta == first.delta
    assert np.allclose(second.w_star, first.w_star)
    assert second.reference_iters == first.reference_iters


def test_build_huber_solves_again_when_the_stored_problem_differs(data, tmp_path):
    import json as json_module

    from src.huber import build_huber

    X, y = data
    path = tmp_path / "huber_reference.json"
    build_huber(X, y, lam=LAM, path=path, verbose=False)
    other = build_huber(X, y, lam=10 * LAM, path=path, verbose=False)

    assert other.lam == pytest.approx(10 * LAM)
    assert json_module.loads(path.read_text())["lam"] == pytest.approx(10 * LAM)
    assert np.linalg.norm(other.gradient(other.w_star)) <= 1e-13


def test_build_huber_solves_again_when_the_stored_data_differs_at_fixed_shape(data, tmp_path):
    """Same n, d, lam is not enough to trust a stored reference.

    `build_scaling_variants` in src/dataset.py produces exactly this shape
    collision from real data: raw, centred, and standardised design matrices
    share n, d, and are run at the same lam, but their rows are different
    numbers. A stored reference built from one and reused for another is a
    wrong-w* bug that the n/d/lam comparison alone cannot see, because delta
    and w_star are both derived from the data, not passed in alongside it.
    """
    import json as json_module

    from src.huber import build_huber

    X, y = data
    path = tmp_path / "huber_reference.json"
    first = build_huber(X, y, lam=LAM, path=path, verbose=False)

    # Same n, d, lam as `data`, but a different draw with a much larger noise
    # scale, so its residual spread -- and therefore its delta -- differs
    # measurably from the first dataset's.
    rng = np.random.default_rng(99)
    n, d = X.shape
    X_other = rng.standard_normal((n, d))
    X_other = (X_other - X_other.mean(0)) / X_other.std(0)
    y_other = X_other @ rng.standard_normal(d) + 3.0 * rng.standard_normal(n)
    y_other = y_other - y_other.mean()

    other = build_huber(X_other, y_other, lam=LAM, path=path, verbose=False)

    # The returned objective must genuinely solve the second problem, not
    # silently carry over the first dataset's w_star.
    assert np.linalg.norm(other.gradient(other.w_star)) <= 1e-13
    assert other.delta != pytest.approx(first.delta, rel=1e-2)

    stored = json_module.loads(path.read_text())
    assert stored["delta"] == pytest.approx(other.delta)


def test_both_objectives_satisfy_the_shared_protocol(
    data: tuple[np.ndarray, np.ndarray], objective: HuberObjective
):
    """The protocol exists so one loop can drive both problems; this is what checks it.

    The call does nothing at runtime. Its value is static: pyright rejects the
    argument if either class drifts out of conformance, which is the failure this
    protocol was introduced to prevent and which no other test would notice.
    `objective` must carry its `HuberObjective` annotation here rather than being
    left implicit, since an unannotated parameter is typed `Unknown` and pyright
    would silently pass any argument through the `accepts` call below without
    ever consulting `SmoothObjective` at all.
    """
    def accepts(problem: SmoothObjective) -> int:
        return problem.d

    X, y = data
    assert accepts(RidgeObjective(X, y, lam=LAM)) == objective.d
    assert accepts(objective) == objective.d
