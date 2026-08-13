"""Ridge regression objective and the constants that govern convergence speed.

The optimisation problem is

    f(w) = 1/(2n) * ||X w - y||^2 + lam/2 * ||w||^2

which is strongly convex, infinitely differentiable, and has a closed-form
minimiser. Every experiment in this project minimises this single function, so
the object built here is created once and then shared, never rebuilt.
"""

from __future__ import annotations

from functools import cached_property
from typing import Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class LocalObjective(Protocol):
    """What a line search is allowed to probe.

    Deterministic methods hand the full objective to the line search, while
    mini-batch SGD hands a view restricted to the current batch. Keeping the two
    behind one protocol makes the difference explicit at the call site instead of
    hiding it inside a long `sgd` function.
    """

    def value(self, w: np.ndarray) -> float: ...

    @property
    def n_rows(self) -> int:
        """Rows of X touched by one call to `value`, for the fairness axis."""
        ...


class SmoothObjective(Protocol):
    """Everything `iterate` and the `Direction` classes ask of an objective.

    Two implementations exist. `RidgeObjective` gets every constant below in
    closed form because it is a quadratic; `HuberObjective` finds the same
    quantities numerically. Nothing in the loop distinguishes them, which is what
    makes the two problems comparable: the same code path produces both sets of
    curves.
    """

    X: np.ndarray
    n: int
    d: int
    lam: float

    @property
    def n_rows(self) -> int: ...

    @property
    def L(self) -> float: ...

    @property
    def mu(self) -> float: ...

    @property
    def kappa(self) -> float: ...

    @property
    def w_star(self) -> np.ndarray: ...

    @property
    def f_star(self) -> float: ...

    def value(self, w: np.ndarray) -> float: ...

    def gradient(self, w: np.ndarray) -> np.ndarray: ...

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]: ...

    def compute_hessian(self, w: np.ndarray | None = None) -> np.ndarray: ...

    def solve_hessian(self, rhs: np.ndarray) -> np.ndarray: ...

    def suboptimality(self, w: np.ndarray) -> float: ...

    def batch(self, indices: np.ndarray) -> LocalObjective: ...

    def summary(self) -> dict: ...


class RidgeObjective:
    """f, its derivatives, and the constants L, mu, kappa for a fixed (X, y, lam)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lam: float) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        if lam <= 0.0:
            raise ValueError(f"lam must be positive, got {lam}")
        # float32 would cap the convergence plots several orders of magnitude
        # above the floor we want to show, so refuse it rather than silently
        # producing curves that flatten out for the wrong reason.
        if X.dtype != np.float64 or y.dtype != np.float64:
            raise ValueError("X and y must be float64")

        self.X = X
        self.y = y
        self.lam = float(lam)
        self.n, self.d = X.shape

    # ---------------------------------------------------------------- values

    @property
    def n_rows(self) -> int:
        return self.n

    def value(self, w: np.ndarray) -> float:
        residual = self.X @ w - self.y
        return float(residual @ residual / (2 * self.n) + self.lam / 2 * (w @ w))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        residual = self.X @ w - self.y
        return self.X.T @ residual / self.n + self.lam * w

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        """Both at once, reusing the single expensive product X @ w.

        A line search needs f at the current point and the gradient there, so
        computing them separately would scan X one extra time per iteration.
        """
        residual = self.X @ w - self.y
        value = float(residual @ residual / (2 * self.n) + self.lam / 2 * (w @ w))
        gradient = self.X.T @ residual / self.n + self.lam * w
        return value, gradient

    # --------------------------------------------------------------- Hessian

    def compute_hessian(self, w: np.ndarray | None = None) -> np.ndarray:
        """Form (1/n) X^T X + lam I from scratch.

        `w` is accepted and ignored: the Hessian of a quadratic is the same
        everywhere. The Huber objective needs the argument, and taking it here as
        well is what lets `NewtonStep` drive both without an `isinstance` check.

        Newton with `reuse_factorization=False` calls this every iteration on
        purpose: reusing the factorisation is correct on this objective but would
        not be in the general case, and the report compares the cost of both.
        """
        gram = self.X.T @ self.X / self.n
        gram.flat[:: self.d + 1] += self.lam
        return gram

    @cached_property
    def hessian(self) -> np.ndarray:
        return self.compute_hessian()

    @cached_property
    def _hessian_factor(self):
        # Cholesky rather than an explicit inverse: cheaper and numerically
        # better behaved. The Hessian is positive definite because lam > 0.
        return cho_factor(self.hessian, lower=True)

    def solve_hessian(self, rhs: np.ndarray) -> np.ndarray:
        """Return H^{-1} rhs using the cached Cholesky factor."""
        return cho_solve(self._hessian_factor, rhs)

    # ------------------------------------------------- optimum and constants

    @cached_property
    def w_star(self) -> np.ndarray:
        return self.solve_hessian(self.X.T @ self.y / self.n)

    @cached_property
    def f_star(self) -> float:
        return self.value(self.w_star)

    @cached_property
    def _gram_eigenvalues(self) -> np.ndarray:
        gram = self.hessian.copy()
        gram.flat[:: self.d + 1] -= self.lam
        return np.linalg.eigvalsh(gram)

    @cached_property
    def L(self) -> float:
        """Lipschitz constant of the gradient."""
        return float(self._gram_eigenvalues[-1]) + self.lam

    @cached_property
    def mu(self) -> float:
        """Strong convexity constant."""
        return float(self._gram_eigenvalues[0]) + self.lam

    @cached_property
    def kappa(self) -> float:
        return self.L / self.mu

    # ---------------------------------------------------------- convergence

    def suboptimality(self, w: np.ndarray) -> float:
        """f(w) - f* evaluated as a quadratic form rather than a difference.

        Because f is quadratic and the gradient vanishes at w*, the Taylor
        expansion has no remainder:

            f(w) - f* = 1/2 * (w - w*)^T H (w - w*)

        Computing it this way matters for the plots. Subtracting two nearly
        equal floats loses everything below about 1e-16 * f*, which puts a floor
        on the curve well above the accuracy the methods actually reach; the
        quadratic form keeps full relative precision all the way down.
        """
        delta = w - self.w_star
        return float(delta @ (self.hessian @ delta) / 2)

    def batch(self, indices: np.ndarray) -> "BatchObjective":
        return BatchObjective(self, indices)

    def summary(self) -> dict:
        """Constants worth recording in problem_config.json."""
        return {
            "n": self.n,
            "d": self.d,
            "lam": self.lam,
            "L": self.L,
            "mu": self.mu,
            "kappa": self.kappa,
            "f_star": self.f_star,
            "w_star_norm": float(np.linalg.norm(self.w_star)),
        }


class BatchObjective:
    """The objective restricted to one mini-batch of rows.

    The regularisation term is kept whole rather than scaled by |B|/n, so this
    is an unbiased estimate of the full gradient when the batch is drawn
    uniformly. A line search handed this object only guarantees decrease on the
    batch, never on f itself; see section 4.1 of KE_HOACH_TRIEN_KHAI.md.
    """

    __slots__ = ("parent", "X", "y", "lam", "size")

    def __init__(self, parent: RidgeObjective, indices: np.ndarray) -> None:
        self.parent = parent
        self.X = parent.X[indices]
        self.y = parent.y[indices]
        self.lam = parent.lam
        self.size = len(indices)

    @property
    def n_rows(self) -> int:
        return self.size

    def value(self, w: np.ndarray) -> float:
        residual = self.X @ w - self.y
        return float(residual @ residual / (2 * self.size) + self.lam / 2 * (w @ w))

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        residual = self.X @ w - self.y
        value = float(residual @ residual / (2 * self.size) + self.lam / 2 * (w @ w))
        gradient = self.X.T @ residual / self.size + self.lam * w
        return value, gradient


def max_row_smoothness(objective: SmoothObjective) -> float:
    """L_max, the largest per-sample Lipschitz constant max_i ||x_i||^2 + lam.

    A step size that is safe for the full gradient can still make SGD with a
    small batch diverge, because L_max is an extreme while L is an average. The
    mini-batch bound in `batch_smoothness` interpolates between the two.

    The bound holds for the Huber objective as well: its per-row Hessian is
    s_i x_i x_i^T with s_i in [0, 1], so no row is steeper than in the Ridge case.
    """
    return float(np.max(np.einsum("ij,ij->i", objective.X, objective.X))) + objective.lam


def batch_smoothness(objective: SmoothObjective, batch_size: int) -> float:
    """L_B = L + (L_max - L) / B, the smoothness seen by a batch of size B."""
    L_max = max_row_smoothness(objective)
    return objective.L + (L_max - objective.L) / batch_size
