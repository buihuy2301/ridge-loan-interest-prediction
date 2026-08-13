"""Axis 1 of the experiment matrix: where a step starts and which way it points.

Each algorithm is a stateful object answering one question, "from which point and
in which direction", and knows nothing about how far to travel. The step length
is axis 2 (`stepsize.py`), and `iterate.py` is the single loop that pairs them.
Splitting the two axes this way is what makes the assignment's four-by-two matrix
a literal Cartesian product in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .objective import LocalObjective, RidgeObjective


@dataclass(frozen=True, slots=True)
class Proposal:
    """One step offer, everything a step rule needs and nothing more."""

    point: np.ndarray
    """Where the step starts. Equal to w for GD, SGD and Newton, but equal to the
    extrapolated y_k for Nesterov. Conflating this with the current iterate is
    the classic way to get Nesterov subtly wrong, so it is a separate field."""

    direction: np.ndarray
    """The descent direction p at `point`."""

    gradient: np.ndarray
    """Gradient at `point`, reused by the line search."""

    value: float
    """Objective at `point`, computed alongside the gradient to avoid a second
    pass over X."""

    local: LocalObjective
    """The function a line search may probe: the full objective for
    deterministic methods, the current mini-batch for SGD. Armijo on a batch only
    guarantees decrease on that batch, and naming the field makes that visible."""

    n_rows: int
    """Rows of X touched to build this proposal, for the fair-comparison axis."""

    def directional_derivative(self) -> float:
        """g^T p, the slope along the direction. Negative for a descent direction."""
        return float(self.gradient @ self.direction)


class Direction(Protocol):
    label: str

    def propose(self, w: np.ndarray, k: int) -> Proposal: ...

    def accept(self, w_new: np.ndarray, t: float, k: int) -> None:
        """Called once the loop has committed to step size t."""
        ...


class SteepestDescent:
    """Full-batch gradient descent: p = -grad f(w)."""

    def __init__(self, objective: RidgeObjective) -> None:
        self.objective = objective
        self.label = "GD"

    def propose(self, w: np.ndarray, k: int) -> Proposal:
        value, gradient = self.objective.value_and_gradient(w)
        return Proposal(
            point=w,
            direction=-gradient,
            gradient=gradient,
            value=value,
            local=self.objective,
            n_rows=self.objective.n,
        )

    def accept(self, w_new: np.ndarray, t: float, k: int) -> None:
        pass


class MiniBatch:
    """Mini-batch SGD: the same direction as GD, estimated on a random subset.

    Batches are drawn by shuffling once per epoch and consuming the permutation
    in chunks, which is what mini-batch SGD means in practice. Sampling with
    replacement would match the textbook analysis more closely but visits some
    rows twice and others not at all within an epoch, making the epoch axis of
    the plots harder to read.
    """

    def __init__(
        self,
        objective: RidgeObjective,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if not 1 <= batch_size <= objective.n:
            raise ValueError(f"batch_size must be in [1, {objective.n}], got {batch_size}")
        self.objective = objective
        self.batch_size = batch_size
        self.seed = seed
        self.label = f"SGD (B = {batch_size})"

        self._rng = np.random.default_rng(seed)
        self._permutation = self._rng.permutation(objective.n)
        self._cursor = 0

    @property
    def steps_per_epoch(self) -> int:
        return self.objective.n // self.batch_size

    def _next_indices(self) -> np.ndarray:
        if self._cursor + self.batch_size > self.objective.n:
            self._permutation = self._rng.permutation(self.objective.n)
            self._cursor = 0
        indices = self._permutation[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return indices

    def propose(self, w: np.ndarray, k: int) -> Proposal:
        batch = self.objective.batch(self._next_indices())
        value, gradient = batch.value_and_gradient(w)
        return Proposal(
            point=w,
            direction=-gradient,
            gradient=gradient,
            value=value,
            local=batch,
            n_rows=self.batch_size,
        )

    def accept(self, w_new: np.ndarray, t: float, k: int) -> None:
        pass


class Nesterov:
    """Accelerated gradient descent: the gradient is taken at an extrapolated point.

        y_k     = w_k + beta_k (w_k - w_{k-1})
        w_{k+1} = y_k - t grad f(y_k)

    Three momentum rules are supported. The `strongly_convex` one recomputes beta
    from the step length the line search actually accepted,

        beta = (1 - sqrt(t mu)) / (1 + sqrt(t mu))

    rather than from the textbook constant (sqrt(kappa) - 1) / (sqrt(kappa) + 1),
    which is only the same thing when t = 1/L. Pairing that constant with a
    larger backtracked step makes the objective blow up instead of decreasing.
    Since t is only known after the line search, this uses the previously
    accepted step, falling back to 1/L on the first iteration.
    """

    RULES = ("strongly_convex", "nesterov", "constant")

    def __init__(
        self,
        objective: RidgeObjective,
        rule: str = "strongly_convex",
        beta: float = 0.9,
        restart: bool = False,
    ) -> None:
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}, got {rule!r}")
        self.objective = objective
        self.rule = rule
        self.constant_beta = beta
        self.restart = restart

        name = {
            "strongly_convex": "AGD (beta from t, mu)",
            "nesterov": "AGD (beta = (k-1)/(k+2))",
            "constant": f"AGD (beta = {beta})",
        }[rule]
        self.label = name + (" + restart" if restart else "")

        self._w_prev: np.ndarray | None = None
        self._w_current: np.ndarray | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_step = 1.0 / objective.L
        self._momentum_k = 0

    def _beta(self) -> float:
        if self.rule == "constant":
            return self.constant_beta
        if self.rule == "nesterov":
            # Increasing towards 1; the k here is reset by adaptive restart, which
            # is precisely what damps the oscillation this rule produces on a
            # strongly convex objective.
            return (self._momentum_k - 1) / (self._momentum_k + 2)
        root = np.sqrt(self._last_step * self.objective.mu)
        return float((1.0 - root) / (1.0 + root))

    def propose(self, w: np.ndarray, k: int) -> Proposal:
        if self._w_prev is None:
            point = w
        else:
            point = w + self._beta() * (w - self._w_prev)

        value, gradient = self.objective.value_and_gradient(point)
        self._w_current = w
        self._last_gradient = gradient
        return Proposal(
            point=point,
            direction=-gradient,
            gradient=gradient,
            value=value,
            local=self.objective,
            n_rows=self.objective.n,
        )

    def accept(self, w_new: np.ndarray, t: float, k: int) -> None:
        assert self._w_current is not None and self._last_gradient is not None

        # Gradient-based adaptive restart: a positive inner product means the
        # momentum is pushing against the gradient, so the accumulated velocity
        # has overshot and the counter goes back to zero.
        if self.restart and self._last_gradient @ (w_new - self._w_current) > 0.0:
            self._momentum_k = 0
        else:
            self._momentum_k += 1

        self._w_prev = self._w_current
        self._last_step = t


class NewtonStep:
    """Newton direction: p solves H p = -grad f(w), with H = (1/n) X^T X + lam I.

    The system is solved by Cholesky rather than by forming an inverse, which is
    both cheaper and better conditioned.

    On this objective H is constant, so `reuse_factorization=True` factors once
    and reuses it. That is correct here but would not be for a general function,
    so the default refactors every iteration and the report compares the cost of
    both. Refactoring touches X a second time, which `n_rows` reflects.
    """

    def __init__(
        self,
        objective: RidgeObjective,
        reuse_factorization: bool = False,
    ) -> None:
        self.objective = objective
        self.reuse_factorization = reuse_factorization
        self.label = "Newton" + (" (Hessian reused)" if reuse_factorization else "")

    def propose(self, w: np.ndarray, k: int) -> Proposal:
        value, gradient = self.objective.value_and_gradient(w)

        if self.reuse_factorization:
            direction = self.objective.solve_hessian(-gradient)
            n_rows = self.objective.n
        else:
            factor = cho_factor(self.objective.compute_hessian(), lower=True)
            direction = cho_solve(factor, -gradient)
            n_rows = 2 * self.objective.n

        return Proposal(
            point=w,
            direction=direction,
            gradient=gradient,
            value=value,
            local=self.objective,
            n_rows=n_rows,
        )

    def accept(self, w_new: np.ndarray, t: float, k: int) -> None:
        pass
