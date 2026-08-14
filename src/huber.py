"""The Huber objective, the second function this project minimises.

Ridge is a quadratic, which hands Newton the solution in one step and lets
backtracking accept the full step at every iteration. Section 4.4 of
KE_HOACH_TRIEN_KHAI.md asks for a function on the same data that takes those two
privileges away without leaving the class of smooth convex problems:

    f(w) = 1/n sum_i H_delta(x_i^T w - y_i) + lam/2 ||w||^2

    H_delta(r) = r^2 / 2                    if |r| <= delta
                 delta (|r| - delta / 2)    otherwise

The function is convex and continuously differentiable. Its Hessian exists
wherever no residual sits exactly on the kink; rows on the kink are counted as
quadratic by convention, which is what `<=` in the mask below encodes.

Nothing here is a quadratic, so three things Ridge gets for free have to be
worked for: the Hessian changes with w, the minimiser is found numerically, and
the suboptimality gap cannot be written as a quadratic form.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .objective import RidgeObjective

REFERENCE_PATH = Path("data/processed/huber_reference.json")


class HuberObjective:
    """f, its derivatives, and the constants that bound its convergence rate."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        delta: float,
        w_star: np.ndarray | None = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        if lam <= 0.0:
            raise ValueError(f"lam must be positive, got {lam}")
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        if X.dtype != np.float64 or y.dtype != np.float64:
            raise ValueError("X and y must be float64")

        self.X = X
        self.y = y
        self.lam = float(lam)
        self.delta = float(delta)
        self.n, self.d = X.shape

        # Set on first access to `w_star`, so the Newton solve in
        # `solve_reference` runs at most once per instance rather than once per
        # call.
        self._w_star: np.ndarray | None = None
        if w_star is not None:
            self._w_star = np.asarray(w_star, dtype=np.float64)
        self.reference_grad_norm = float("nan")
        self.reference_iters = -1
        self._frozen_factor = None

    # ---------------------------------------------------------------- values

    @property
    def n_rows(self) -> int:
        return self.n

    def _residual(self, w: np.ndarray) -> np.ndarray:
        return self.X @ w - self.y

    def _loss(self, residual: np.ndarray) -> np.ndarray:
        """H_delta applied row by row."""
        size = np.abs(residual)
        return np.where(
            size <= self.delta,
            0.5 * residual * residual,
            self.delta * (size - 0.5 * self.delta),
        )

    def value(self, w: np.ndarray) -> float:
        return float(self._loss(self._residual(w)).sum() / self.n + self.lam / 2 * (w @ w))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        residual = self._residual(w)
        return self.X.T @ np.clip(residual, -self.delta, self.delta) / self.n + self.lam * w

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        """Both at once, reusing the single expensive product X @ w."""
        residual = self._residual(w)
        value = float(self._loss(residual).sum() / self.n + self.lam / 2 * (w @ w))
        gradient = self.X.T @ np.clip(residual, -self.delta, self.delta) / self.n + self.lam * w
        return value, gradient

    # --------------------------------------------------------------- Hessian

    def compute_hessian(self, w: np.ndarray | None = None) -> np.ndarray:
        """(1/n) X_S^T X_S + lam I, with S the rows inside the quadratic zone.

        Rows outside it have a linear loss and contribute no curvature, so the
        Hessian is the Ridge one built from a subset of the rows. Unlike the Ridge
        case this genuinely changes with w, which is why the argument is required
        here and ignored there.
        """
        if w is None:
            raise ValueError("the Huber Hessian depends on w, so w is required")
        inside = np.flatnonzero(np.abs(self._residual(w)) <= self.delta)
        active = self.X[inside]
        hessian = active.T @ active / self.n
        hessian.flat[:: self.d + 1] += self.lam
        return hessian

    def solve_hessian(self, rhs: np.ndarray) -> np.ndarray:
        """Solve H(0) p = rhs, with the factorisation frozen at the starting point.

        `NewtonStep(reuse_factorization=True)` means "reuse the Hessian", which on
        a quadratic is exact and here is the chord method: the direction stays a
        descent direction but the quadratic convergence is gone. Freezing at w = 0
        rather than at whatever point happens to call first keeps the run
        reproducible.
        """
        if self._frozen_factor is None:
            self._frozen_factor = cho_factor(self.compute_hessian(np.zeros(self.d)), lower=True)
        return cho_solve(self._frozen_factor, rhs)

    # ------------------------------------------------- optimum and constants

    @property
    def w_star(self) -> np.ndarray:
        """The minimiser, solved for rather than read off a formula."""
        if self._w_star is None:
            self._w_star, self.reference_grad_norm, self.reference_iters = solve_reference(self)
        return self._w_star

    @cached_property
    def _optimal_value(self) -> float:
        return self.value(self.w_star)

    @property
    def f_star(self) -> float:
        # SmoothObjective declares this as a `@property`, and a protocol property
        # is not satisfied by `cached_property` (pyright rejects it even though
        # both are read-only from the caller's side), so the expensive part lives
        # in `_optimal_value` and this is a thin wrapper over it, matching
        # `RidgeObjective`. Do not collapse it back into `cached_property`.
        return self._optimal_value

    @cached_property
    def _residual_star(self) -> np.ndarray:
        return self._residual(self.w_star)

    @cached_property
    def _gram_eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.X.T @ self.X / self.n)

    @property
    def L(self) -> float:
        """Upper bound on the Lipschitz constant of the gradient.

        Every row of the Hessian carries a weight in [0, 1], so the Huber Hessian
        never exceeds the Ridge one in the positive semidefinite order and the
        Ridge constant bounds both. It is also the only bound an algorithm can
        compute without knowing where its iterates will go.
        """
        return float(self._gram_eigenvalues[-1]) + self.lam

    @property
    def mu(self) -> float:
        """Lower bound on the strong convexity constant.

        Rows outside the quadratic zone contribute no curvature, so in the worst
        case the regularisation term is all that is left. Returning lam rather
        than the tighter value measured at w* keeps the constant honest: Nesterov
        reads `mu` to build its momentum, and it cannot be handed the answer.
        """
        return self.lam

    @property
    def kappa(self) -> float:
        return self.L / self.mu

    @cached_property
    def mu_at_optimum(self) -> float:
        """The smallest Hessian eigenvalue at w*, for the report rather than the loop.

        The gap between this and `mu` is how loose the guaranteed bound is on this
        particular problem.
        """
        return float(np.linalg.eigvalsh(self.compute_hessian(self.w_star))[0])

    # ---------------------------------------------------------- convergence

    def suboptimality(self, w: np.ndarray) -> float:
        """f(w) - f*, with the cancellation pushed down to individual rows.

        Subtracting f* from f(w) at the end loses everything below about
        1e-16 * f*, which on the sweep problem is 1e-15 and cuts off the last
        stage of Newton's quadratic tail. Differencing the per-row losses first,
        and writing the penalty difference as a product rather than as a
        difference of two squares, keeps several more orders of magnitude.

        The Ridge class does better still, because a quadratic has an exact
        quadratic form for the gap. No such identity exists here, so this is the
        floor the Huber figures are drawn against.
        """
        rows = self._loss(self._residual(w)) - self._loss(self._residual_star)
        penalty = self.lam / 2 * float((w - self.w_star) @ (w + self.w_star))
        return float(rows.sum() / self.n + penalty)

    def summary(self) -> dict:
        """Constants worth recording alongside a group of runs."""
        return {
            "n": self.n,
            "d": self.d,
            "lam": self.lam,
            "delta": self.delta,
            "L": self.L,
            "mu": self.mu,
            "kappa": self.kappa,
            "mu_at_optimum": self.mu_at_optimum,
            "f_star": self.f_star,
            "w_star_norm": float(np.linalg.norm(self.w_star)),
            "outside_fraction": float(np.mean(np.abs(self._residual_star) > self.delta)),
            "reference_grad_norm": self.reference_grad_norm,
            "reference_iters": self.reference_iters,
        }


def solve_reference(
    objective: HuberObjective,
    tol: float = 1e-14,
    max_iter: int = 100,
) -> tuple[np.ndarray, float, int]:
    """Newton with backtracking, run to machine precision, for w* and f*.

    Every convergence curve is measured against this point, so it has to be far
    more accurate than the runs it judges. Newton earns its place here rather than
    in a library call: it reaches 1e-16 in single digits of iterations on this
    problem, and the code is the same three lines the report is about.

    Returns the minimiser, the gradient norm reached, and the iterations spent.
    """
    w = np.zeros(objective.d)
    for k in range(max_iter):
        value, gradient = objective.value_and_gradient(w)
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm <= tol:
            return w, grad_norm, k

        direction = np.linalg.solve(objective.compute_hessian(w), -gradient)
        slope = float(gradient @ direction)
        step = 1.0
        while objective.value(w + step * direction) > value + 1e-4 * step * slope:
            step *= 0.5
            if step < 1e-12:
                raise RuntimeError(f"line search collapsed at iteration {k}")
        w = w + step * direction

    raise RuntimeError(
        f"reference solve did not reach {tol:g} in {max_iter} iterations; "
        f"last gradient norm was {grad_norm:.3e}"
    )
