"""Check 5 of section 8: run the whole Direction x StepRule product.

The assignment asks for four algorithms with two step rules each. Because the
two axes are separate objects, "did we implement all eight?" is a loop rather
than a reading exercise, and the list exercised here is the list submitted.
"""

import numpy as np
import pytest

from src.direction import MiniBatch, Nesterov, NewtonStep, SteepestDescent
from src.iterate import iterate
from src.objective import RidgeObjective, batch_smoothness
from src.record import load_records, save_records
from src.stepsize import Armijo, Decay, Fixed


@pytest.fixture(scope="module")
def objective() -> RidgeObjective:
    rng = np.random.default_rng(0)
    n, d = 2000, 20
    # Decaying column scales give a spectrum with some spread, so the methods
    # are not all trivially one step from the optimum.
    raw = rng.standard_normal((n, d)) @ np.diag(np.logspace(0, -1.5, d))
    X = (raw - raw.mean(0)) / raw.std(0)
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n)
    return RidgeObjective(X, y - y.mean(), lam=1e-3)


DETERMINISTIC = ("SteepestDescent", "Nesterov", "Nesterov+restart",
                 "NewtonStep", "NewtonStep-reuse")


def make_direction(objective, name):
    """Fresh instance per call: every Direction except SteepestDescent has state."""
    return {
        "SteepestDescent": lambda: (SteepestDescent(objective), 1.0 / objective.L, 5000),
        "Nesterov": lambda: (Nesterov(objective, "strongly_convex"), 1.0 / objective.L, 2000),
        "Nesterov+restart": lambda: (Nesterov(objective, "nesterov", restart=True),
                                     1.0 / objective.L, 2000),
        "NewtonStep": lambda: (NewtonStep(objective), 1.0, 20),
        "NewtonStep-reuse": lambda: (NewtonStep(objective, reuse_factorization=True), 1.0, 20),
    }[name]()


@pytest.mark.parametrize("name", DETERMINISTIC)
@pytest.mark.parametrize("rule", ["fixed", "armijo"])
def test_every_pairing_reaches_the_same_optimum(objective, name, rule):
    direction, step, max_iter = make_direction(objective, name)
    step_rule = Fixed(step) if rule == "fixed" else Armijo(t0=step)
    record = iterate(objective, direction, step_rule, max_iter=max_iter, warmup=False)

    assert record.status in ("converged", "stalled"), record.status
    assert np.linalg.norm(np.asarray(record.w_final) - objective.w_star) < 1e-8
    assert record.final_gap < 1e-16


def test_newton_with_a_full_step_finishes_in_one_iteration(objective):
    """On a quadratic the Newton step is the closed-form solution itself."""
    record = iterate(objective, NewtonStep(objective), Fixed(1.0), max_iter=20, warmup=False)
    assert record.status == "converged"
    assert record.iters[-1] <= 2
    assert np.allclose(record.w_final, objective.w_star, atol=1e-12)


def test_armijo_accepts_the_full_newton_step_immediately(objective):
    """Backtracking has nothing to do here, which is why section 4.4 adds Huber."""
    record = iterate(objective, NewtonStep(objective), Armijo(t0=1.0), max_iter=20, warmup=False)
    assert record.fevals[1] == 1


@pytest.mark.parametrize(
    "multiple, expected",
    [(1.9, "converged"), (2.1, "diverged")],
)
def test_divergence_above_two_over_L_is_reported_not_hidden(objective, multiple, expected):
    """Stall detection must not swallow the deliberate divergence experiment."""
    record = iterate(
        objective,
        SteepestDescent(objective),
        Fixed(multiple / objective.L),
        max_iter=2000,
        warmup=False,
    )
    assert record.status == expected


def test_armijo_probe_count_grows_when_t0_is_too_large(objective):
    """The cost that makes backtracking lose on time while winning on iterations."""
    cheap = iterate(objective, SteepestDescent(objective), Armijo(t0=1.0 / objective.L),
                    max_iter=400, warmup=False)
    dear = iterate(objective, SteepestDescent(objective), Armijo(t0=50.0 / objective.L),
                   max_iter=400, warmup=False)

    # Starting at 1/L the descent lemma guarantees acceptance on the first probe,
    # so the average sits just above one; the excess comes from the last few
    # iterations, where the true decrease falls below the resolution of f.
    assert cheap.fevals_per_iter < 1.5
    # Starting fifty times too high costs one probe per halving, every iteration.
    assert dear.fevals_per_iter > cheap.fevals_per_iter + 2.0
    assert dear.fevals[-1] > 2 * cheap.fevals[-1]


def test_stochastic_runs_are_measured_in_epochs_and_never_stall(objective):
    """A constant step plateaus by design; cutting it off would delete the result."""
    direction = MiniBatch(objective, 128, seed=1)
    step = Decay("constant", 1.0 / batch_smoothness(objective, 128))
    record = iterate(objective, direction, step, max_epochs=30, warmup=False)

    assert record.is_stochastic
    assert record.status == "max_iter"
    assert record.epochs[-1] == pytest.approx(30, rel=0.05)
    assert record.final_gap > 1e-12  # the plateau, not a failure


def test_decaying_steps_beat_a_constant_step_for_sgd(objective):
    eta0 = 1.0 / batch_smoothness(objective, 128)
    gaps = {}
    for kind, rule in [
        ("constant", Decay("constant", eta0)),
        ("sqrt", Decay("sqrt", eta0)),
        ("inverse", Decay("inverse", eta0, gamma=0.01)),
    ]:
        record = iterate(objective, MiniBatch(objective, 128, seed=2), rule,
                         max_epochs=40, warmup=False)
        gaps[kind] = record.final_gap
    assert gaps["sqrt"] < gaps["constant"]
    assert gaps["inverse"] < gaps["constant"]


def test_rows_seen_charges_newton_for_rebuilding_the_hessian(objective):
    fresh = iterate(objective, NewtonStep(objective), Fixed(1.0), max_iter=3, warmup=False)
    reused = iterate(objective, NewtonStep(objective, reuse_factorization=True),
                     Fixed(1.0), max_iter=3, warmup=False)
    assert fresh.rows_seen[-1] == 2 * reused.rows_seen[-1]


def test_logging_density_does_not_change_the_measured_time(objective):
    """The clock stops around logging, so checkpoint spacing must not show up."""
    # A deliberately small step and tol = 0 keep both runs at the full 300
    # iterations, so the only difference between them is how often they log.
    slow = dict(max_iter=300, tol=0.0, warmup=True)
    dense = iterate(objective, SteepestDescent(objective), Fixed(0.02 / objective.L),
                    record_every=1, **slow)
    sparse = iterate(objective, SteepestDescent(objective), Fixed(0.02 / objective.L),
                     record_every=100, **slow)
    assert dense.iters[-1] == sparse.iters[-1] == 300
    assert len(dense.iters) > 10 * len(sparse.iters)
    assert dense.total_time == pytest.approx(sparse.total_time, rel=0.6)


def test_records_survive_a_json_round_trip(objective, tmp_path):
    original = [
        iterate(objective, SteepestDescent(objective), Fixed(1 / objective.L),
                max_iter=50, warmup=False),
        iterate(objective, NewtonStep(objective), Fixed(1.0), max_iter=5, warmup=False),
    ]
    path = save_records(original, tmp_path / "group.json", meta={"kappa": objective.kappa})
    restored, meta = load_records(path)

    assert meta["kappa"] == pytest.approx(objective.kappa)
    for before, after in zip(original, restored):
        assert after.label == before.label
        assert after.direction_label == before.direction_label
        assert after.step_label == before.step_label
        assert after.gaps == before.gaps
        assert after.status == before.status
