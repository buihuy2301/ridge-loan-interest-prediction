"""scikit-learn baselines, and the objective conversion that makes them comparable.

Section 6 of the plan asks for a comparison of the final objective value and the
training time. The trap it warns about is that each estimator normalises the
objective differently, so the first job here is to convert lambda into whatever
constant the estimator expects and then to *check the conversion numerically*
rather than trusting the algebra.

    ours          f(w)  = 1/(2n) ||Xw - y||^2 + lam/2 ||w||^2

    Ridge               ||Xw - y||^2 + alpha ||w||^2
                        multiply ours by 2n  ->  alpha = lam * n

    SGDRegressor        1/n sum 1/2 (x_i^T w - y_i)^2 + alpha/2 ||w||^2
                        already ours          ->  alpha = lam

Two estimators, two different constants. Getting this wrong does not raise an
error, it silently compares two different problems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

from .objective import RidgeObjective
from .record import RunRecord, save_records


def ridge_alpha(objective: RidgeObjective) -> float:
    """alpha for `Ridge`, which drops both the 1/n and the 1/2 of our objective."""
    return objective.lam * objective.n


def sgd_alpha(objective: RidgeObjective) -> float:
    """alpha for `SGDRegressor`, whose objective already matches ours."""
    return objective.lam


@dataclass
class BaselineResult:
    name: str
    fit_seconds: float
    gap: float
    """f(w_hat) - f*, from the same exact quadratic form used for our own runs."""
    rmse_test: float
    grad_norm: float
    n_iter: int | None = None
    times_all: list[float] = field(default_factory=list)

    def to_record(self) -> RunRecord:
        """A one-point record, so the library results can share the plotting code."""
        return RunRecord(
            label=self.name,
            status="converged",
            direction_label="library",
            step_label=self.name,
            iters=[self.n_iter or 1],
            gaps=[self.gap],
            gnorms=[self.grad_norm],
            times=[self.fit_seconds],
            rows_seen=[0],
            fevals=[0],
            meta={"library": True, "rmse_test": self.rmse_test},
        )


def verify_alpha_conversion(objective: RidgeObjective, tolerance: float = 1e-6) -> dict:
    """Fit `Ridge` at alpha = lam * n and check it lands on our closed-form solution.

    Section 6 insists this be checked with numbers. If the constant were wrong the
    two objectives would differ, both would still solve cleanly, and every later
    comparison would be between two unrelated problems.
    """
    model = Ridge(alpha=ridge_alpha(objective), fit_intercept=False, solver="cholesky")
    model.fit(objective.X, objective.y)
    w = np.asarray(model.coef_, dtype=np.float64)

    report = {
        "alpha_ridge": ridge_alpha(objective),
        "alpha_sgd": sgd_alpha(objective),
        "gap": objective.suboptimality(w),
        "w_distance": float(np.linalg.norm(w - objective.w_star)),
        "relative_w_distance": float(
            np.linalg.norm(w - objective.w_star) / np.linalg.norm(objective.w_star)
        ),
    }
    report["verified"] = report["relative_w_distance"] < tolerance
    return report


def _timed_fit(model, X: np.ndarray, y: np.ndarray, repeats: int = 3) -> tuple[np.ndarray, list[float]]:
    """Fit `repeats` times and keep every duration, so the caller can take a median.

    The same discipline as `iterate`: only the fit is inside the clock, and one
    throwaway fit warms the BLAS paths first.
    """
    model.fit(X[: min(1000, len(X))], y[: min(1000, len(y))])

    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.fit(X, y)
        durations.append(time.perf_counter() - started)
    return np.asarray(model.coef_, dtype=np.float64).ravel(), durations


def _evaluate(
    name: str,
    model,
    objective: RidgeObjective,
    X_test: np.ndarray,
    y_test: np.ndarray,
    repeats: int,
) -> BaselineResult:
    w, durations = _timed_fit(model, objective.X, objective.y, repeats)
    residual = X_test @ w - y_test
    return BaselineResult(
        name=name,
        fit_seconds=float(np.median(durations)),
        gap=objective.suboptimality(w),
        rmse_test=float(np.sqrt(residual @ residual / len(y_test))),
        grad_norm=float(np.linalg.norm(objective.gradient(w))),
        n_iter=int(np.ravel(model.n_iter_)[0]) if hasattr(model, "n_iter_") and model.n_iter_ is not None else None,
        times_all=[float(d) for d in durations],
    )


def build_baselines(objective: RidgeObjective, max_iter: int = 1000, seed: int = 0) -> dict:
    """The estimators of the table in section 6, each with its converted constant."""
    alpha = ridge_alpha(objective)
    return {
        "Ridge (solver='auto')": Ridge(alpha=alpha, fit_intercept=False, solver="auto"),
        "Ridge (solver='cholesky')": Ridge(alpha=alpha, fit_intercept=False, solver="cholesky"),
        "Ridge (solver='lsqr')": Ridge(alpha=alpha, fit_intercept=False, solver="lsqr"),
        "Ridge (solver='sparse_cg')": Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg"),
        "Ridge (solver='sag')": Ridge(alpha=alpha, fit_intercept=False, solver="sag",
                                      max_iter=max_iter, random_state=seed),
        "SGDRegressor (defaults)": SGDRegressor(
            loss="squared_error", penalty="l2", alpha=sgd_alpha(objective),
            fit_intercept=False, random_state=seed,
        ),
        # lambda = 0, included only as the unregularised reference point.
        "LinearRegression": LinearRegression(fit_intercept=False),
    }


def run_baselines(
    objective: RidgeObjective,
    X_test: np.ndarray,
    y_test: np.ndarray,
    repeats: int = 3,
    max_iter: int = 1000,
    seed: int = 0,
    verbose: bool = True,
) -> list[BaselineResult]:
    log = print if verbose else (lambda *a, **k: None)

    check = verify_alpha_conversion(objective)
    if not check["verified"]:
        raise RuntimeError(
            "alpha = lam * n did not reproduce the closed-form solution "
            f"(relative distance {check['relative_w_distance']:.2e}); "
            "every comparison below would be between different objectives"
        )
    log(
        f"conversion checked: alpha = lam*n = {check['alpha_ridge']:.4g}, "
        f"|w_sklearn - w*|/|w*| = {check['relative_w_distance']:.2e}, "
        f"gap = {check['gap']:.3e}"
    )

    results = []
    for name, model in build_baselines(objective, max_iter, seed).items():
        result = _evaluate(name, model, objective, X_test, y_test, repeats)
        results.append(result)
        log(
            f"   {name:<30} t={result.fit_seconds:7.3f}s  gap={result.gap:11.3e}  "
            f"rmse={result.rmse_test:.4f}  n_iter={result.n_iter}"
        )
    return results


def baseline_table(results: list[BaselineResult], objective: RidgeObjective) -> list[dict]:
    """Rows for the comparison table, with our own closed form as the reference."""
    rows = [
        {
            "method": "closed form (ours)",
            "seconds": None,
            "gap": 0.0,
            "grad_norm": float(np.linalg.norm(objective.gradient(objective.w_star))),
            "rmse_test": None,
            "n_iter": 1,
        }
    ]
    rows.extend(
        {
            "method": r.name,
            "seconds": r.fit_seconds,
            "gap": r.gap,
            "grad_norm": r.grad_norm,
            "rmse_test": r.rmse_test,
            "n_iter": r.n_iter,
        }
        for r in results
    )
    return rows


def save_baselines(
    results: list[BaselineResult],
    objective: RidgeObjective,
    path: str | Path = "results/raw/library.json",
) -> Path:
    """Store them as records so the `library` figures reuse the shared plotting code."""
    return save_records(
        [r.to_record() for r in results],
        path,
        meta={
            "group": "library",
            "title": "scikit-learn baselines",
            "conversion": verify_alpha_conversion(objective),
            "table": baseline_table(results, objective),
            **objective.summary(),
        },
    )
