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
from src.objective import RidgeObjective

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
