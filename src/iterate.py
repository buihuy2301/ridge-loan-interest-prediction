"""The one loop that pairs a `Direction` with a `StepRule`.

Every algorithm in this project runs through this function. Nothing here knows
which algorithm it is driving: anything specific to one method belongs in that
method's `Direction` class. If a change to this file needs `isinstance`, the
change belongs somewhere else.

Keeping the loop in one place is what makes the timing trustworthy. The rule
that the clock stops before logging has to hold for every curve on a shared pair
of axes, and here it is written once instead of once per algorithm.
"""

from __future__ import annotations

import time

import numpy as np

from .direction import Direction
from .objective import RidgeObjective
from .record import RunRecord
from .stepsize import StepRule


class _Clock:
    """Accumulates only the segments that are actual algorithm work."""

    __slots__ = ("total", "_started")

    def __init__(self) -> None:
        self.total = 0.0
        self._started: float | None = None

    def start(self) -> None:
        self._started = time.perf_counter()

    def stop(self) -> None:
        assert self._started is not None, "clock stopped without being started"
        self.total += time.perf_counter() - self._started
        self._started = None


def warm_up(objective: RidgeObjective, repeats: int = 3) -> None:
    """Touch the expensive paths once so the first timed iteration is not special.

    The first numpy call pays initialisation and cache-warming costs that would
    otherwise land entirely on iteration zero. Both primitives are exercised,
    the memory-bound pass over X and the BLAS-3 product behind the Hessian, so
    the loop does not need to know which one the direction will use.
    """
    probe = np.zeros(objective.d)
    for _ in range(repeats):
        objective.value_and_gradient(probe)
    objective.compute_hessian()
    _ = objective.w_star  # forces the Gram matrix and its factorisation


def iterate(
    objective: RidgeObjective,
    direction: Direction,
    step_rule: StepRule,
    *,
    w0: np.ndarray | None = None,
    max_iter: int = 1000,
    max_epochs: int | None = None,
    tol: float = 1e-10,
    record_every: int | None = None,
    warmup: bool = True,
    label: str | None = None,
    stall_window: int = 20,
    diverge_factor: float = 1e6,
) -> RunRecord:
    """Run one configuration and return its history.

    Args:
        max_epochs: for stochastic directions, an alternative to `max_iter`
            expressed in passes over the data. Ignored otherwise.
        tol: stop when ||grad|| <= tol * ||grad at w0||. Not applied to
            stochastic directions, where the gradient is a noisy batch estimate.
        record_every: checkpoint spacing; None picks 1 for deterministic methods
            and a tenth of an epoch for stochastic ones, since logging every one
            of 150000 SGD iterations costs more than the run itself.
        stall_window: stop after this many checkpoints without improving on the
            best gap so far. Runs that are diverging are exempt, otherwise the
            deliberate t > 2/L experiment would be cut short before it shows
            anything.
    """
    steps_per_epoch = getattr(direction, "steps_per_epoch", None)
    is_stochastic = steps_per_epoch is not None

    if is_stochastic and max_epochs is not None:
        max_iter = max_epochs * steps_per_epoch
    if record_every is None:
        record_every = max(1, steps_per_epoch // 10) if is_stochastic else 1

    if warmup:
        warm_up(objective)

    w = np.zeros(objective.d) if w0 is None else np.array(w0, dtype=np.float64)

    record = RunRecord(
        label=label or f"{direction.label}, {step_rule.label}",
        status="max_iter",
        direction_label=direction.label,
        step_label=step_rule.label,
        is_stochastic=is_stochastic,
        rows_per_epoch=objective.n,
        meta={
            "max_iter": max_iter,
            "tol": tol,
            "record_every": record_every,
            "L": objective.L,
            "mu": objective.mu,
            "kappa": objective.kappa,
            "f_star": objective.f_star,
        },
    )

    clock = _Clock()
    rows_seen = 0
    fevals = 0
    grad0_norm = -1.0  # negative until the first checkpoint sets it
    gap0 = -1.0
    best_gap = np.inf
    prev_gap = np.inf
    since_improvement = 0

    def log(k: int, gnorm: float) -> None:
        """Untimed. The gap costs O(d^2) via the quadratic form, the norm is free."""
        record.iters.append(k)
        record.gaps.append(objective.suboptimality(w))
        record.gnorms.append(gnorm)
        record.times.append(clock.total)
        record.rows_seen.append(rows_seen)
        record.fevals.append(fevals)

    for k in range(max_iter):
        clock.start()
        proposal = direction.propose(w, k)
        clock.stop()

        rows_seen += proposal.n_rows
        gnorm = float(np.linalg.norm(proposal.gradient))

        if k % record_every == 0:
            log(k, gnorm)
            gap = record.gaps[-1]

            if gap0 < 0.0:
                gap0, grad0_norm = max(gap, np.finfo(float).tiny), gnorm

            if not np.isfinite(gap) or gap > diverge_factor * gap0:
                record.status = "diverged"
                break

            # Stall detection is for deterministic methods only. SGD with a
            # constant step settles into a neighbourhood of w* and stops
            # improving by design; cutting it off there would delete the very
            # plateau the report is about.
            #
            # A run whose gap is still growing is not stalling, it is diverging,
            # and it must be left alone until it trips the threshold above.
            # Counting growth as "no improvement" would stop the deliberate
            # t > 2/L experiment after `stall_window` checkpoints, long before
            # the divergence is visible on a log axis.
            if not is_stochastic:
                if gap < best_gap * (1.0 - 1e-9):
                    best_gap, since_improvement = gap, 0
                elif gap > prev_gap:
                    since_improvement = 0
                else:
                    since_improvement += 1
                    if since_improvement >= stall_window:
                        record.status = "stalled"
                        break
            prev_gap = gap

        # The gradient here is taken at `proposal.point`, which is w itself for
        # every method except Nesterov, where it sits at the extrapolated point.
        # Using it as the stopping quantity costs nothing extra and differs from
        # the gradient at w only while momentum is still moving the iterate.
        if not is_stochastic and grad0_norm > 0.0 and gnorm <= tol * grad0_norm:
            record.status = "converged"
            break

        clock.start()
        step = step_rule(proposal, k)
        if not step.accepted:
            clock.stop()
            fevals += step.n_evals
            record.status = "line_search_failed"
            break
        w = proposal.point + step.size * proposal.direction
        direction.accept(w, step.size, k)
        clock.stop()

        fevals += step.n_evals
    else:
        k = max_iter

    # Closing checkpoint. The gradient computed here exists only to fill the
    # record, so it stays outside the clock like every other logging cost.
    if not record.iters or record.iters[-1] != k:
        log(k, float(np.linalg.norm(objective.gradient(w))))

    record.w_final = [float(v) for v in w]
    return record


def iterations_for_epochs(direction: Direction, epochs: int) -> int:
    """Convert a budget in passes over the data into a budget in iterations."""
    steps_per_epoch = getattr(direction, "steps_per_epoch", None)
    if steps_per_epoch is None:
        raise TypeError(f"{type(direction).__name__} is not a stochastic direction")
    return epochs * steps_per_epoch
