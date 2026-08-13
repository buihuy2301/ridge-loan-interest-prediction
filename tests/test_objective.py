"""Checks 1 to 4 of section 8: derivatives, closed form, and the gap identity."""

import numpy as np
import pytest

from src.objective import RidgeObjective, batch_smoothness, max_row_smoothness

EPS = 1e-6


@pytest.fixture(scope="module")
def objective() -> RidgeObjective:
    rng = np.random.default_rng(0)
    n, d = 400, 12
    X = rng.standard_normal((n, d))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n)
    return RidgeObjective(X, y - y.mean(), lam=1e-2)


@pytest.fixture(scope="module")
def probe(objective) -> np.ndarray:
    return np.random.default_rng(7).standard_normal(objective.d)


def test_gradient_matches_central_differences(objective, probe):
    analytic = objective.gradient(probe)
    for i in range(objective.d):
        shift = np.zeros(objective.d)
        shift[i] = EPS
        numeric = (objective.value(probe + shift) - objective.value(probe - shift)) / (2 * EPS)
        assert abs(numeric - analytic[i]) <= 1e-6 * max(abs(analytic[i]), 1.0)


def test_hessian_matches_differences_of_the_gradient(objective, probe):
    hessian = objective.hessian
    for i in range(objective.d):
        shift = np.zeros(objective.d)
        shift[i] = EPS
        numeric = (objective.gradient(probe + shift) - objective.gradient(probe - shift)) / (2 * EPS)
        assert np.allclose(numeric, hessian[:, i], atol=1e-7)


def test_value_and_gradient_agrees_with_the_separate_calls(objective, probe):
    value, gradient = objective.value_and_gradient(probe)
    assert value == pytest.approx(objective.value(probe))
    assert np.allclose(gradient, objective.gradient(probe))


def test_closed_form_is_stationary(objective):
    assert np.linalg.norm(objective.gradient(objective.w_star)) < 1e-12


def test_closed_form_beats_every_perturbation(objective):
    rng = np.random.default_rng(3)
    for _ in range(20):
        perturbed = objective.w_star + 1e-3 * rng.standard_normal(objective.d)
        assert objective.value(perturbed) > objective.f_star


def test_constants_are_ordered(objective):
    assert 0 < objective.mu <= objective.L
    assert objective.kappa == pytest.approx(objective.L / objective.mu)
    # lam shifts the whole spectrum, so it is a floor on the strong convexity.
    assert objective.mu >= objective.lam


def test_suboptimality_is_exact_and_survives_cancellation(objective):
    """The quadratic form must stay accurate where the naive difference cannot.

    Section 13.5 of the plan: subtracting two nearly equal floats bottoms out
    around 1e-16 * f*, which would put a false floor on every convergence plot.
    """
    rng = np.random.default_rng(11)
    # Far from the optimum the two agree, which is what makes the identity exact
    # rather than merely better conditioned.
    for scale in (1e-1, 1e-2, 1e-3):
        w = objective.w_star + scale * rng.standard_normal(objective.d)
        naive = objective.value(w) - objective.f_star
        assert objective.suboptimality(w) == pytest.approx(naive, rel=1e-6)

    # Close in, the naive difference is pure rounding noise and can even go
    # negative, while the quadratic form stays positive and meaningful.
    w = objective.w_star + 1e-12 * rng.standard_normal(objective.d)
    assert objective.suboptimality(w) > 0.0
    assert objective.suboptimality(w) < 1e-20
    assert objective.suboptimality(objective.w_star) == 0.0


def test_batch_objective_is_an_unbiased_view(objective):
    everything = np.arange(objective.n)
    batch = objective.batch(everything)
    probe = np.random.default_rng(5).standard_normal(objective.d)
    value, gradient = batch.value_and_gradient(probe)
    assert value == pytest.approx(objective.value(probe))
    assert np.allclose(gradient, objective.gradient(probe))
    assert batch.n_rows == objective.n


def test_batch_smoothness_interpolates_between_L_and_L_max(objective):
    L_max = max_row_smoothness(objective)
    assert L_max > objective.L

    # A batch of one sees the worst single sample; larger batches average the
    # extreme away and approach L from above, never reaching it for finite n.
    assert batch_smoothness(objective, 1) == pytest.approx(L_max)
    sizes = [1, 8, 64, 512]
    values = [batch_smoothness(objective, b) for b in sizes]
    assert values == sorted(values, reverse=True)
    assert all(v > objective.L for v in values)
    assert batch_smoothness(objective, objective.n) - objective.L == pytest.approx(
        (L_max - objective.L) / objective.n
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"lam": 0.0}, "lam"),
        ({"lam": -1.0}, "lam"),
    ],
)
def test_rejects_non_positive_regularisation(objective, kwargs, message):
    with pytest.raises(ValueError, match=message):
        RidgeObjective(objective.X, objective.y, **kwargs)


def test_rejects_float32_because_it_would_fake_a_convergence_floor(objective):
    with pytest.raises(ValueError, match="float64"):
        RidgeObjective(objective.X.astype(np.float32), objective.y, lam=1e-2)


def test_compute_hessian_ignores_the_point_it_is_given(objective, probe):
    """A quadratic has a constant Hessian, and the Ridge class says so by ignoring w.

    The argument exists because the Huber objective needs it. Ridge accepting and
    discarding it is what lets one loop drive both.
    """
    at_probe = objective.compute_hessian(probe)
    at_zero = objective.compute_hessian(np.zeros(objective.d))
    assert np.array_equal(at_probe, at_zero)
    assert np.array_equal(at_probe, objective.compute_hessian())
