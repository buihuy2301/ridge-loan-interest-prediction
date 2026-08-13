# Hàm mục tiêu Huber: kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm hàm mục tiêu Huber kèm hiệu chỉnh Ridge làm bài toán thứ hai của dự án, chạy cả bốn thuật toán đã có trên nó, rồi viết kết quả thành một chương báo cáo và hai frame slide.

**Architecture:** `iterate` đã gọi hàm mục tiêu qua một bề mặt hẹp không có gì riêng của hàm bậc hai, nên thiết kế tách bề mặt ấy thành `Protocol` tên `SmoothObjective`, giữ nguyên `RidgeObjective` với toàn bộ đường đi nghiệm đóng, và thêm `HuberObjective` trong file riêng. Hessian của Huber phụ thuộc $w$ nên `compute_hessian` đổi chữ ký thành `compute_hessian(w)`; đó là thay đổi giao diện duy nhất chạm vào mã đang chạy.

**Tech Stack:** Python 3.14, NumPy, SciPy (`cho_factor`, `cho_solve`), matplotlib, pytest, LaTeX với `latexmk -xelatex`.

**Spec:** `docs/superpowers/specs/2026-08-13-ham-muc-tieu-huber-design.md`

## Global Constraints

- Mã nguồn, comment, docstring, chuỗi log, nhãn biểu đồ: **tiếng Anh**, không ngoại lệ. Markdown, báo cáo, slide, ô markdown trong notebook: **tiếng Việt**. Quy tắc ở mục 1 của `CLAUDE.md`.
- Trước khi viết bất kỳ đoạn văn tiếng Việt nào cho báo cáo hoặc slide, đọc `docs/van-phong-tieng-viet.md` và làm đủ ba lượt A, B, C như mục 1 của file đó quy định.
- Trước khi sửa `report/*.tex`, đọc `docs/quy-uoc-bao-cao.md`.
- Quy mô chạy: chỉ `SWEEP`, tức $n = 200{\,}000$, $d = 116$. Không chạy quy mô `FULL`.
- Hằng số cố định: $\lambda = 0{,}03162277660168379$ giữ nguyên như bài toán Ridge; $\delta = 1{,}345 \hat\sigma$ với $\hat\sigma$ ước lượng qua MAD của phần dư nghiệm Ridge, giá trị đo được là 4,0666.
- Mỗi hình lưu ra `results/figures/` ở hai định dạng, PDF và PNG với `dpi=150`, luôn dùng `bbox_inches='tight'`. Hàm `save_figure` trong `src/figures.py` đã làm đúng việc này.
- Đo thời gian: dừng đồng hồ trước khi tính và ghi log. `iterate` đã cài đúng, không sửa.
- Mọi hình và bảng trong báo cáo phải có `\caption`, `\label`, và được `\ref` ít nhất một lần. Mục nào thêm vào `refs.bib` phải được `\cite`.
- Chạy `pytest` từ thư mục gốc dự án. `conftest.py` đã lo phần `sys.path`.

---

## Cấu trúc file

| File | Trạng thái | Trách nhiệm |
| --- | --- | --- |
| `src/objective.py` | Sửa | Thêm `Protocol` `SmoothObjective`; đổi chữ ký `compute_hessian(w=None)`; nới chú thích kiểu của `max_row_smoothness` và `batch_smoothness` |
| `src/huber.py` | Tạo | `HuberObjective`, `HuberBatchObjective`, `choose_delta`, `solve_reference`, `build_huber` |
| `src/direction.py` | Sửa | Một dòng gọi `compute_hessian(w)`; nới chú thích kiểu sang `SmoothObjective` |
| `src/iterate.py` | Sửa | `warm_up` truyền điểm thăm dò vào `compute_hessian`; nới chú thích kiểu |
| `src/experiment.py` | Sửa | Hai builder mới, `HUBER_BUILDERS`, `HUBER_GROUPS`, cờ `--problem` cho `main()`; nới chú thích kiểu |
| `tests/test_huber.py` | Tạo | Toàn bộ phép kiểm cho hàm mục tiêu mới |
| `tests/test_experiment.py` | Sửa | Phép kiểm cho hai builder mới |
| `notebooks/09_huber.ipynb` | Tạo | Chạy hai nhóm, vẽ năm hình, đo chi phí dựng Hessian |
| `report/report.tex` | Sửa | Chương mới chèn trước dòng 752 |
| `report/slides.tex` | Sửa | Hai frame mới |
| `report/refs.bib` | Sửa | Mục Huber 1964 |
| `KE_HOACH_TRIEN_KHAI.md` | Sửa | Mục 4.4, 4.5, 11, 13, 14 |

---

## Task 1: Bề mặt chung cho hai hàm mục tiêu

**Files:**
- Modify: `src/objective.py` (thêm Protocol sau `LocalObjective` ở dòng 36; sửa `compute_hessian` ở dòng 86; sửa chú thích kiểu ở dòng 210 và 220)
- Modify: `src/direction.py:259`
- Modify: `src/iterate.py:43-55`
- Test: `tests/test_objective.py`

**Interfaces:**
- Consumes: không có, đây là task đầu.
- Produces: `SmoothObjective` Protocol trong `src/objective.py`, và chữ ký `compute_hessian(self, w: np.ndarray | None = None) -> np.ndarray` trên mọi lớp hàm mục tiêu.

- [ ] **Step 1: Viết test cho chữ ký mới**

Thêm vào cuối `tests/test_objective.py`:

```python
def test_compute_hessian_ignores_the_point_it_is_given(objective, probe):
    """A quadratic has a constant Hessian, and the Ridge class says so by ignoring w.

    The argument exists because the Huber objective needs it. Ridge accepting and
    discarding it is what lets one loop drive both.
    """
    at_probe = objective.compute_hessian(probe)
    at_zero = objective.compute_hessian(np.zeros(objective.d))
    assert np.array_equal(at_probe, at_zero)
    assert np.array_equal(at_probe, objective.compute_hessian())
```

- [ ] **Step 2: Chạy test, xác nhận nó hỏng**

Run: `.venv/bin/python -m pytest tests/test_objective.py::test_compute_hessian_ignores_the_point_it_is_given -v`
Expected: FAIL, `TypeError: RidgeObjective.compute_hessian() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Đổi chữ ký trong `src/objective.py`**

Thay phần khai báo `compute_hessian` ở dòng 86:

```python
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
```

- [ ] **Step 4: Thêm `Protocol` vào `src/objective.py`, ngay sau `LocalObjective`**

```python
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
```

- [ ] **Step 5: Nới chú thích kiểu của hai hàm cuối `src/objective.py`**

Đổi `objective: RidgeObjective` thành `objective: SmoothObjective` trong `max_row_smoothness` (dòng 210) và `batch_smoothness` (dòng 220). Thêm vào docstring của `max_row_smoothness` một câu giải thích vì sao cận này dùng chung được:

```python
def max_row_smoothness(objective: SmoothObjective) -> float:
    """L_max, the largest per-sample Lipschitz constant max_i ||x_i||^2 + lam.

    A step size that is safe for the full gradient can still make SGD with a
    small batch diverge, because L_max is an extreme while L is an average. The
    mini-batch bound in `batch_smoothness` interpolates between the two.

    The bound holds for the Huber objective as well: its per-row Hessian is
    s_i x_i x_i^T with s_i in [0, 1], so no row is steeper than in the Ridge case.
    """
```

- [ ] **Step 6: Sửa chỗ gọi trong `src/direction.py:259`**

```python
            factor = cho_factor(self.objective.compute_hessian(w), lower=True)
```

Đồng thời đổi mọi chú thích kiểu `objective: RidgeObjective` trong file thành `objective: SmoothObjective`, và sửa dòng import ở đầu file thành:

```python
from .objective import LocalObjective, SmoothObjective
```

Sửa docstring của `NewtonStep` để nói đúng cả hai bài toán:

```python
class NewtonStep:
    """Newton direction: p solves H(w) p = -grad f(w).

    The system is solved by Cholesky rather than by forming an inverse, which is
    both cheaper and better conditioned.

    `reuse_factorization=True` factors once and reuses it. On the Ridge objective
    that is exact, because a quadratic has a constant Hessian. On the Huber
    objective the same flag turns Newton into the chord method: still a descent
    direction, but the quadratic convergence is gone. The report compares both.
    Refactoring touches X a second time, which `n_rows` reflects.
    """
```

- [ ] **Step 7: Sửa `warm_up` trong `src/iterate.py`**

```python
def warm_up(objective: SmoothObjective, repeats: int = 3) -> None:
    """Touch the expensive paths once so the first timed iteration is not special.

    The first numpy call pays initialisation and cache-warming costs that would
    otherwise land entirely on iteration zero. Both primitives are exercised, the
    memory-bound pass over X and the BLAS-3 product behind the Hessian, so the
    loop does not need to know which one the direction will use.
    """
    probe = np.zeros(objective.d)
    for _ in range(repeats):
        objective.value_and_gradient(probe)
    objective.compute_hessian(probe)
    _ = objective.w_star  # forces the reference solution and its factorisation
```

Đổi chú thích kiểu của tham số `objective` trong `iterate` thành `SmoothObjective`, và sửa dòng import thành `from .objective import SmoothObjective`.

- [ ] **Step 8: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 124 test.

- [ ] **Step 9: Commit**

```bash
git add src/objective.py src/direction.py src/iterate.py tests/test_objective.py
git commit -m "refactor: widen the objective surface to a SmoothObjective protocol"
```

---

## Task 2: `HuberObjective`, phần giá trị và đạo hàm

**Files:**
- Create: `src/huber.py`
- Test: `tests/test_huber.py`

**Interfaces:**
- Consumes: `SmoothObjective` từ Task 1; `RidgeObjective` để so sánh trong test.
- Produces: `HuberObjective(X, y, lam, delta)` với `value`, `gradient`, `value_and_gradient`, `compute_hessian(w)`, thuộc tính `X`, `y`, `lam`, `delta`, `n`, `d`, `n_rows`.

- [ ] **Step 1: Viết test cho phần giá trị và đạo hàm**

Tạo `tests/test_huber.py`:

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận nó hỏng**

Run: `.venv/bin/python -m pytest tests/test_huber.py -v`
Expected: FAIL ở bước thu thập, `ModuleNotFoundError: No module named 'src.huber'`

- [ ] **Step 3: Viết `src/huber.py`**

```python
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

        # Set when the reference solution is read from disk, so a rerun does not
        # pay for the Newton solve again.
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
```

- [ ] **Step 4: Chạy test, xác nhận nó pass**

Run: `.venv/bin/python -m pytest tests/test_huber.py -v`
Expected: PASS, 6 test.

- [ ] **Step 5: Commit**

```bash
git add src/huber.py tests/test_huber.py
git commit -m "feat: add the Huber objective with its value, gradient and Hessian"
```

---

## Task 3: Hằng số, nghiệm tham chiếu, và cách đo sai số

**Files:**
- Modify: `src/huber.py`
- Test: `tests/test_huber.py`

**Interfaces:**
- Consumes: `HuberObjective` từ Task 2.
- Produces: `solve_reference(objective, tol=1e-14, max_iter=100) -> tuple[np.ndarray, float, int]`; các thuộc tính `L`, `mu`, `kappa`, `mu_at_optimum`, `w_star`, `f_star`; các phương thức `solve_hessian(rhs)`, `suboptimality(w)`, `summary()`.

- [ ] **Step 1: Viết test**

Thêm vào cuối `tests/test_huber.py`:

```python
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
    """Close to w* the plain difference is rounding noise and the paired one is not.

    The gap of a smooth function near its minimiser is the quadratic form of the
    Hessian, and that is what the paired formula has to reproduce at a distance
    where subtracting two floats cannot.
    """
    step = 1e-9 * np.ones(objective.d)
    w = objective.w_star + step
    quadratic = 0.5 * float(step @ (objective.compute_hessian(objective.w_star) @ step))

    assert objective.suboptimality(w) == pytest.approx(quadratic, rel=1e-3)
    plain = objective.value(w) - objective.f_star
    assert abs(plain - quadratic) > 0.5 * quadratic


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
```

- [ ] **Step 2: Chạy test, xác nhận nó hỏng**

Run: `.venv/bin/python -m pytest tests/test_huber.py -v`
Expected: FAIL, `AttributeError: 'HuberObjective' object has no attribute 'w_star'`

- [ ] **Step 3: Thêm phần hằng số và nghiệm tham chiếu vào `src/huber.py`**

Thêm vào cuối lớp `HuberObjective`:

```python
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
    def f_star(self) -> float:
        return self.value(self.w_star)

    @cached_property
    def _residual_star(self) -> np.ndarray:
        return self._residual(self.w_star)

    @cached_property
    def _gram_eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.X.T @ self.X / self.n)

    @cached_property
    def L(self) -> float:
        """Upper bound on the Lipschitz constant of the gradient.

        Every row of the Hessian carries a weight in [0, 1], so the Huber Hessian
        never exceeds the Ridge one in the positive semidefinite order and the
        Ridge constant bounds both. It is also the only bound an algorithm can
        compute without knowing where its iterates will go.
        """
        return float(self._gram_eigenvalues[-1]) + self.lam

    @cached_property
    def mu(self) -> float:
        """Lower bound on the strong convexity constant.

        Rows outside the quadratic zone contribute no curvature, so in the worst
        case the regularisation term is all that is left. Returning lam rather
        than the tighter value measured at w* keeps the constant honest: Nesterov
        reads `mu` to build its momentum, and it cannot be handed the answer.
        """
        return self.lam

    @cached_property
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

    def batch(self, indices: np.ndarray) -> "HuberBatchObjective":
        return HuberBatchObjective(self, indices)

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
```

Thêm hàm module sau lớp:

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận nó pass**

Run: `.venv/bin/python -m pytest tests/test_huber.py -v`
Expected: PASS, 13 test.

- [ ] **Step 5: Commit**

```bash
git add src/huber.py tests/test_huber.py
git commit -m "feat: add the Huber constants, reference solution and gap measure"
```

---

## Task 4: Lô con và bản đông cứng của bài toán

**Files:**
- Modify: `src/huber.py`
- Test: `tests/test_huber.py`

**Interfaces:**
- Consumes: `HuberObjective`, `solve_reference` từ Task 3.
- Produces: `HuberBatchObjective`; `choose_delta(ridge, factor=1.345) -> float`; `build_huber(X, y, lam, path=REFERENCE_PATH, verbose=True) -> HuberObjective`.

- [ ] **Step 1: Viết test**

Thêm vào cuối `tests/test_huber.py`:

```python
def test_a_batch_over_every_row_is_the_full_objective(objective, probe):
    batch = objective.batch(np.arange(objective.n))
    value, gradient = batch.value_and_gradient(probe)
    full_value, full_gradient = objective.value_and_gradient(probe)
    assert value == pytest.approx(full_value, rel=1e-12)
    assert np.allclose(gradient, full_gradient, rtol=1e-12)
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
    assert batch.value(w) == pytest.approx(same_rows.value(w), rel=1e-12)

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
```

- [ ] **Step 2: Chạy test, xác nhận nó hỏng**

Run: `.venv/bin/python -m pytest tests/test_huber.py -v`
Expected: FAIL, `AttributeError: 'HuberObjective' object has no attribute 'batch'` hoặc `ImportError: cannot import name 'choose_delta'`

- [ ] **Step 3: Thêm lớp lô con và ba hàm module vào `src/huber.py`**

```python
class HuberBatchObjective:
    """The Huber objective restricted to one mini-batch of rows.

    The regularisation term is kept whole rather than scaled by |B|/n, so this is
    an unbiased estimate of the full gradient when the batch is drawn uniformly.
    A line search handed this object only guarantees decrease on the batch, never
    on f itself; see section 4.1 of KE_HOACH_TRIEN_KHAI.md.
    """

    __slots__ = ("parent", "X", "y", "lam", "delta", "size")

    def __init__(self, parent: HuberObjective, indices: np.ndarray) -> None:
        self.parent = parent
        self.X = parent.X[indices]
        self.y = parent.y[indices]
        self.lam = parent.lam
        self.delta = parent.delta
        self.size = len(indices)

    @property
    def n_rows(self) -> int:
        return self.size

    def _loss(self, residual: np.ndarray) -> np.ndarray:
        size = np.abs(residual)
        return np.where(
            size <= self.delta,
            0.5 * residual * residual,
            self.delta * (size - 0.5 * self.delta),
        )

    def value(self, w: np.ndarray) -> float:
        residual = self.X @ w - self.y
        return float(self._loss(residual).sum() / self.size + self.lam / 2 * (w @ w))

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        residual = self.X @ w - self.y
        value = float(self._loss(residual).sum() / self.size + self.lam / 2 * (w @ w))
        gradient = self.X.T @ np.clip(residual, -self.delta, self.delta) / self.size + self.lam * w
        return value, gradient


def choose_delta(ridge: RidgeObjective, factor: float = 1.345) -> float:
    """delta = 1.345 * a robust estimate of the residual scale.

    The constant is the classical one: it gives 95 percent asymptotic efficiency
    against the squared loss when the noise really is Gaussian, while still
    clipping the tail. Scale is estimated from the median absolute deviation
    rather than the standard deviation, because the heavy tail this loss exists
    to handle would otherwise set the threshold meant to clip it.
    """
    residual = ridge.X @ ridge.w_star - ridge.y
    spread = float(np.median(np.abs(residual - np.median(residual))))
    return factor * spread / 0.6745


def build_huber(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    path: str | Path = REFERENCE_PATH,
    verbose: bool = True,
) -> HuberObjective:
    """The frozen Huber problem: delta and w* read from disk, or solved and written.

    Section 1 of KE_HOACH_TRIEN_KHAI.md requires the objective to be fixed before
    any timing run starts. Recomputing delta on every call would satisfy that in
    practice, since the inputs are frozen too, but writing it down makes the
    requirement checkable and saves the Newton solve on every rerun.

    A stored file describing a different problem is ignored rather than trusted:
    silently reusing the wrong w* would put every curve in the chapter on the
    wrong axis.
    """
    path = Path(path)
    log = print if verbose else (lambda *a, **k: None)
    n, d = X.shape

    if path.exists():
        stored = json.loads(path.read_text())
        matches = (
            stored["n"] == n
            and stored["d"] == d
            and np.isclose(stored["lam"], lam, rtol=1e-12)
        )
        if matches:
            log(f"huber: reference read from {path}")
            objective = HuberObjective(X, y, lam, stored["delta"], w_star=stored["w_star"])
            objective.reference_grad_norm = stored["grad_norm"]
            objective.reference_iters = stored["iters"]
            return objective
        log(f"huber: {path} describes a different problem, solving again")

    objective = HuberObjective(X, y, lam, choose_delta(RidgeObjective(X, y, lam)))
    w_star, grad_norm, iters = solve_reference(objective)
    objective._w_star = w_star
    objective.reference_grad_norm = grad_norm
    objective.reference_iters = iters

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n": n,
                "d": d,
                "lam": objective.lam,
                "delta": objective.delta,
                "grad_norm": grad_norm,
                "iters": iters,
                "f_star": objective.f_star,
                "w_star": [float(v) for v in w_star],
            },
            indent=1,
        )
    )
    log(f"huber: delta = {objective.delta:.4f}, solved in {iters} iterations, written to {path}")
    return objective
```

- [ ] **Step 4: Chạy test, xác nhận nó pass**

Run: `.venv/bin/python -m pytest tests/test_huber.py -v`
Expected: PASS, 18 test.

- [ ] **Step 5: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 142 test.

- [ ] **Step 6: Commit**

```bash
git add src/huber.py tests/test_huber.py
git commit -m "feat: add Huber mini-batches and the frozen problem builder"
```

---

## Task 5: Hai nhóm thí nghiệm

**Files:**
- Modify: `src/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `build_huber` từ Task 4; `SmoothObjective` từ Task 1.
- Produces: `group_huber_newton`, `group_huber_headline`, `HUBER_BUILDERS`, `HUBER_GROUPS`; cờ `--problem {ridge,huber}` cho `python -m src.experiment`.

- [ ] **Step 1: Viết test**

Thêm vào cuối `tests/test_experiment.py`:

```python
def test_the_huber_groups_expand_to_the_configurations_the_chapter_needs():
    from src.experiment import HUBER_BUILDERS, HUBER_GROUPS, group_huber_headline, group_huber_newton
    from src.huber import HuberObjective

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8))
    y = X @ rng.standard_normal(8) + 0.2 * rng.standard_normal(200)
    objective = HuberObjective(X, y - y.mean(), lam=1e-2, delta=0.5)

    # Two directions crossed with two step rules, as in the Ridge Newton group.
    assert len(group_huber_newton(objective)) == 4
    # One configuration for each of the four methods, plus the chord variant.
    assert len(group_huber_headline(objective)) == 5

    assert set(HUBER_BUILDERS) == {"huber-newton", "huber-headline"}
    assert all(group.name.startswith("huber-") for group in HUBER_GROUPS)


def test_the_huber_groups_do_not_collide_with_the_ridge_ones():
    from src.experiment import BUILDERS, HUBER_BUILDERS

    assert not set(BUILDERS) & set(HUBER_BUILDERS)
```

- [ ] **Step 2: Chạy test, xác nhận nó hỏng**

Run: `.venv/bin/python -m pytest tests/test_experiment.py -k huber -v`
Expected: FAIL, `ImportError: cannot import name 'HUBER_BUILDERS'`

- [ ] **Step 3: Thêm hai builder vào `src/experiment.py`, ngay sau `group_headline`**

```python
# ------------------------------------------- the second problem, section 4.4


def group_huber_newton(objective: SmoothObjective) -> list[RunSpec]:
    """Newton on a function that is not a quadratic.

    On Ridge this group compares a full step against a damped one and pays for
    refactoring against reusing. Here the same four cells answer a sharper
    question: how many iterations Newton needs once it no longer lands on w* in
    one, and what reusing the Hessian costs when the Hessian actually moves.
    """
    directions: dict[str, MakeDirection] = {
        "Newton": lambda o: NewtonStep(o),
        "Newton (Hessian reused)": lambda o: NewtonStep(o, reuse_factorization=True),
    }
    steps: dict[str, MakeStep] = {
        "t = 1": lambda _: Fixed(1.0, label="t = 1"),
        "Armijo": lambda _: Armijo(c=1e-4, rho=0.5, t0=1.0),
    }
    return product(directions, steps, max_iter=60, repeats=3)


def group_huber_headline(objective: SmoothObjective) -> list[RunSpec]:
    """The configurations chosen on Ridge, rerun against the Huber objective.

    The grids are deliberately not swept again. The question here is what the
    shape of the objective does to the ranking, and re-tuning every method would
    put two changes into one figure.

    One configuration differs from the Ridge headline and has to: there the best
    Newton reused its factorisation, which is exact on a quadratic. Reusing it
    here is the chord method, a different algorithm, so the headline uses plain
    Newton and the chord variant stays in `huber-newton` where it is the subject
    rather than a stand-in.
    """
    return [
        RunSpec(SteepestDescent, lambda o: Fixed(1.9 / o.L, label="t = 1.9/L"),
                label="GD (t = 1.9/L)", max_iter=3000, repeats=3,
                params={"method": "GD, fixed step"}),
        RunSpec(SteepestDescent, lambda o: Armijo(c=0.3, rho=0.5, t0=1.0),
                label="GD (Armijo c=0.3, rho=0.5, t0=1)", max_iter=1500, repeats=3,
                params={"method": "GD, backtracking"}),
        RunSpec(lambda o: Nesterov(o, "strongly_convex", restart=True),
                lambda o: Fixed(1.0 / o.L, label="t = 1/L"),
                label="AGD (beta from t, mu, restart)", max_iter=1000, repeats=3,
                params={"method": "AGD"}),
        RunSpec(lambda o: MiniBatch(o, 2048, seed=0),
                lambda o: Decay("constant", 1.0 / batch_smoothness(o, 2048)),
                label="SGD (B = 2048, eta = 1/L_B)", max_epochs=40, repeats=3,
                params={"method": "SGD"}),
        RunSpec(lambda o: NewtonStep(o), lambda o: Fixed(1.0, label="t = 1"),
                label="Newton (t = 1)", max_iter=15, repeats=3,
                params={"method": "Newton"}),
    ]


HUBER_BUILDERS: dict[str, tuple[str, Callable[[SmoothObjective], list[RunSpec]]]] = {
    "huber-newton": ("Newton on the Huber objective", group_huber_newton),
    "huber-headline": ("Best configuration of each method, Huber objective", group_huber_headline),
}

HUBER_GROUPS = [ExperimentGroup(name, title, build) for name, (title, build) in HUBER_BUILDERS.items()]
```

- [ ] **Step 4: Nới chú thích kiểu trong `src/experiment.py`**

Đổi `RidgeObjective` thành `SmoothObjective` trong `MakeDirection`, `MakeStep`, `ExperimentGroup.build`, `run_spec`, `run_group`, `run_all`, và mọi builder đã có. Giữ nguyên `RidgeObjective` ở `run_lambda_sweep` và `run_scaling_comparison`, vì hai hàm đó dựng bài toán Ridge mới trong thân hàm chứ không nhận bài toán từ ngoài. Sửa dòng import thành:

```python
from .objective import RidgeObjective, SmoothObjective, batch_smoothness
```

- [ ] **Step 5: Thêm cờ `--problem` vào `main()`**

Trong `main()`, sau dòng `parser.add_argument("--scale", ...)`:

```python
    parser.add_argument("--problem", choices=("ridge", "huber"), default="ridge")
```

rồi thay phần dựng bài toán và chọn nhóm:

```python
    scale = SWEEP if args.scale == "sweep" else FULL
    objective, X_test, y_test, _ = load_processed(scale=scale)

    if args.problem == "huber":
        from .huber import build_huber

        objective = build_huber(objective.X, objective.y, objective.lam)
        available = HUBER_GROUPS
        print(
            f"problem=huber  scale={args.scale}  n={objective.n:,}  d={objective.d}  "
            f"lam={objective.lam:.5g}  delta={objective.delta:.4f}  L={objective.L:.4f}  "
            f"mu={objective.mu:.6f}  kappa={objective.kappa:.1f}\n"
        )
    else:
        available = GROUPS
        print(
            f"scale={args.scale}  n={objective.n:,}  d={objective.d}  "
            f"lam={objective.lam:.5g}  L={objective.L:.4f}  mu={objective.mu:.6f}  "
            f"kappa={objective.kappa:.1f}\n"
        )

    groups = [g for g in available if not args.only or g.name in args.only]
```

Đưa khối `library` hiện có vào nhánh `ridge`, vì đường cơ sở scikit-learn chỉ định nghĩa cho bài toán Ridge:

```python
    if args.problem == "ridge" and (not args.only or "library" in args.only):
```

- [ ] **Step 6: Chạy test**

Run: `.venv/bin/python -m pytest tests/test_experiment.py -v`
Expected: PASS.

- [ ] **Step 7: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 144 test.

- [ ] **Step 8: Commit**

```bash
git add src/experiment.py tests/test_experiment.py
git commit -m "feat: add the two Huber experiment groups and a --problem flag"
```

---

## Task 6: Chạy thí nghiệm, vẽ hình, đo chi phí

**Files:**
- Create: `notebooks/09_huber.ipynb`
- Tạo ra: `results/raw/huber-newton.json`, `results/raw/huber-headline.json`, `data/processed/huber_reference.json`, và năm hình trong `results/figures/`

**Interfaces:**
- Consumes: `HUBER_BUILDERS`, `HUBER_GROUPS` từ Task 5; `build_huber` từ Task 4; `convergence_pair`, `cost_figure`, `save_figure` từ `src/figures.py`.
- Produces: các file kết quả và hình mà Task 7 trích số từ đó.

- [ ] **Step 1: Chạy hai nhóm từ dòng lệnh**

Run: `.venv/bin/python -m src.experiment --problem huber`
Expected: hai nhóm chạy xong, in ra `delta = 4.0666` và `kappa` khoảng 288,3; hai file JSON xuất hiện trong `results/raw/`.

Nếu nhóm `huber-headline` chạy quá 20 phút thì phần lớn thời gian nằm ở việc ghi checkpoint của SGD, mỗi lần một lượt quét toàn bộ $X$. Đặt `record_every=20` trong `RunSpec` của SGD rồi chạy lại nhóm đó với `--force --only huber-headline`.

- [ ] **Step 2: Tạo notebook `notebooks/09_huber.ipynb`**

Ô markdown đầu:

```markdown
# 09. Hàm mục tiêu Huber

Ridge là hàm bậc hai, nên Newton xong sau một vòng và backtracking nhận ngay bước đầy đủ.
Notebook này chạy cùng bốn thuật toán trên hàm mất mát Huber, nơi cả hai đặc quyền đó
biến mất. Chương 8 của báo cáo.
```

Ô mã thứ nhất:

```python
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
pd.set_option("display.width", 160)
```

Ô mã thứ hai, dựng bài toán và in ba hằng số:

```python
from src.dataset import load_processed, SWEEP
from src.huber import build_huber

ridge, *_ = load_processed("../data/processed", SWEEP)
huber = build_huber(ridge.X, ridge.y, ridge.lam, "../data/processed/huber_reference.json")
pd.DataFrame([
    {"bài toán": "Ridge", "L": ridge.L, "mu": ridge.mu, "kappa": ridge.kappa, "f*": ridge.f_star},
    {"bài toán": "Huber", "L": huber.L, "mu": huber.mu, "kappa": huber.kappa, "f*": huber.f_star},
]).round(4)
```

Ô mã thứ ba, chạy hai nhóm và vẽ bốn hình:

```python
from src.experiment import HUBER_BUILDERS, ExperimentGroup, run_group
from src.figures import convergence_pair, cost_figure, save_figure

records = {}
for name, (title, build) in HUBER_BUILDERS.items():
    records[name] = run_group(ExperimentGroup(name, title, build), huber, "../results/raw")
    convergence_pair(records[name], name, title=title, out_dir="../results/figures")
```

Ô mã thứ tư, hình chi phí line search dựng từ bản ghi đã có:

```python
armijo = [r for group in records.values() for r in group if "Armijo" in r.step_label]
figure = cost_figure(armijo, title="Line search cost on the Huber objective")
save_figure(figure, "huber-step_cost", "../results/figures")
```

Ô markdown và ô mã thứ năm, đo chi phí dựng Hessian:

```python
import time

def timed(build, repeats=3):
    build()                                                  # warm up
    times = []
    for _ in range(repeats):
        t = time.perf_counter(); build(); times.append(time.perf_counter() - t)
    return float(np.median(times)) * 1e3

def masked_copy():
    """The other way to build the Huber Hessian: weight every row, keep them all."""
    s = (np.abs(huber.X @ huber.w_star - huber.y) <= huber.delta).astype(np.float64)
    weighted = huber.X * s[:, None]
    hessian = weighted.T @ huber.X / huber.n
    hessian.flat[:: huber.d + 1] += huber.lam
    return hessian

w_star = huber.w_star
outside = float(np.mean(np.abs(huber.X @ w_star - huber.y) > huber.delta))
pd.DataFrame([
    {"cách dựng": "Ridge, mọi dòng", "ms": timed(lambda: ridge.compute_hessian())},
    {"cách dựng": "Huber, lấy chỉ số", "ms": timed(lambda: huber.compute_hessian(w_star))},
    {"cách dựng": "Huber, nhân trọng số", "ms": timed(masked_copy)},
]).round(2)
```

Ô mã tiếp theo kiểm tra hai cách dựng cho cùng một ma trận, vì con số thời gian
chỉ có nghĩa khi cả hai tính đúng cùng một thứ:

```python
assert np.allclose(huber.compute_hessian(w_star), masked_copy(), rtol=1e-12)
print(f"tỷ lệ dòng ngoài vùng bậc hai: {outside:.4f}")
```

Cách nào nhanh hơn thì ghi con số vào mục 13 ở Task 8. Giữ nguyên cách lấy chỉ số
trong `src/huber.py` trừ khi cách nhân trọng số nhanh hơn trên 20 phần trăm; đổi
mã vì chênh lệch nhỏ hơn thế là đổi để đổi.

Ô mã thứ sáu, bảng số cho báo cáo:

```python
from src.experiment import summary_table
pd.DataFrame(summary_table(records["huber-headline"])).round(6)
```

Ô mã thứ bảy, gap từng vòng của Newton, số liệu cho bảng đuôi bậc hai:

```python
newton = next(r for r in records["huber-newton"] if r.label.startswith("Newton,") and "reused" not in r.label)
pd.DataFrame({"vòng": newton.iters, "gap": newton.gaps}).head(12)
```

- [ ] **Step 3: Chạy toàn bộ notebook từ đầu tới cuối**

Run: `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/09_huber.ipynb`
Expected: chạy xong không lỗi; năm hình xuất hiện trong `results/figures/` ở cả hai định dạng.

- [ ] **Step 4: Kiểm tra danh sách hình**

Run: `ls results/figures/huber-*`
Expected: đúng mười file, gồm `huber-newton_iters`, `huber-newton_time`, `huber-headline_iters`, `huber-headline_time`, `huber-step_cost`, mỗi tên hai đuôi `.pdf` và `.png`.

- [ ] **Step 5: Ghi lại các con số cần cho báo cáo**

Chép ra một chỗ tạm, vì Task 7 cần chúng: $\delta$, $L$, $\mu$, $\kappa$, $\mu$ tại $w^*$, $f^*$, tỷ lệ dòng ngoài vùng bậc hai, số vòng Newton, dãy gap từng vòng của Newton, số lần đánh giá hàm mỗi vòng của GD và của Newton, thời gian dựng Hessian của hai bài toán, và bảng tổng hợp của `huber-headline`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/09_huber.ipynb results/raw/huber-newton.json results/raw/huber-headline.json data/processed/huber_reference.json results/figures/huber-*
git commit -m "feat: run the Huber experiments and draw their figures"
```

---

## Task 7: Chương báo cáo và hai frame slide

**Files:**
- Modify: `report/report.tex` (chèn chương mới ngay trước dòng `\chapter{So sánh tổng hợp trên hai trục}`)
- Modify: `report/slides.tex` (hai frame sau frame về Newton, hiện ở dòng 327)
- Modify: `report/refs.bib`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: các con số ghi lại ở Task 6 bước 5; các stem hình `huber-newton`, `huber-headline`, `huber-step_cost`.
- Produces: nhãn `\label{ch:huber}` để các chương khác tham chiếu tới.

- [ ] **Step 1: Đọc hai tài liệu bắt buộc**

Đọc `docs/quy-uoc-bao-cao.md` và `docs/van-phong-tieng-viet.md`. Chương này là văn phân tích kết quả nên phải đi đủ ba lượt A, B, C.

- [ ] **Step 2: Lượt A, dựng bảng lập luận, không viết câu văn nào**

Mỗi đoạn dự kiến một dòng, bốn cột: kết luận, số liệu chống lưng, cơ chế, điều kiện đảo chiều. Bốn mục của chương, tối thiểu một dòng mỗi mục:

| Mục | Kết luận cần chống lưng |
| --- | --- |
| 1 | Huber giữ nguyên $L$ nhưng hạ $\mu$ xuống sàn $\lambda$, nên $\kappa$ tăng từ 267,5 lên 288,3 |
| 2 | Newton cần 8 vòng thay vì 1, và ba vòng cuối là hội tụ bậc hai |
| 2 | Dùng lại Hessian biến Newton thành phương pháp dây cung, đổi bậc hai lấy tuyến tính |
| 3 | Backtracking lùi bước với GD mà không lùi với Newton |
| 4 | Thứ hạng bốn thuật toán trên Huber so với trên Ridge |

Dòng nào chưa điền được cột cơ chế thì quay lại số liệu, không viết đoạn mô tả để lấp chỗ.

- [ ] **Step 3: Thêm mục vào `report/refs.bib`**

```bibtex
@article{huber1964robust,
  author  = {Huber, Peter J.},
  title   = {Robust Estimation of a Location Parameter},
  journal = {The Annals of Mathematical Statistics},
  volume  = {35},
  number  = {1},
  pages   = {73--101},
  year    = {1964},
  doi     = {10.1214/aoms/1177703732},
}
```

- [ ] **Step 4: Lượt B, viết nháp chương, chèn vào `report/report.tex`**

Chèn ngay trước dòng `\chapter{So sánh tổng hợp trên hai trục}\label{ch:headline}`. Khung bắt buộc, phần văn xuôi viết ở lượt này với bộ mẫu `style/mau/` nạp kèm:

```latex
% ===========================================================================
\chapter{Khi hàm mục tiêu không còn là hàm bậc hai}\label{ch:huber}
% ===========================================================================

\section{Mất mát Huber trên cùng bộ dữ liệu}
% Dinh nghia ham, \cite{huber1964robust}, delta chon the nao, bang tab:huber-constants

\section{Newton mất đặc quyền một vòng}
% Bang tab:huber-newton va hinh fig:huber-newton

\resultpair{huber-newton}{Newton trên hàm mục tiêu Huber, với bước đầy đủ và với
  backtracking, mỗi biến thể một lần dựng lại Hessian và một lần dùng lại nhân tử
  của điểm khởi tạo.}{fig:huber-newton}

\section{Backtracking có việc với hướng nào}
% Hinh fig:huber-cost

\resultfig{huber-step_cost}{Số lần đánh giá hàm mục tiêu mỗi vòng lặp mà điều kiện
  Armijo đòi, trên hướng gradient so với trên hướng Newton.}{fig:huber-cost}

\section{Thứ hạng bốn thuật toán trên hàm phi tuyến}
% Hinh fig:huber-headline, doi chieu voi chuong \ref{ch:headline}

\resultpair{huber-headline}{Bốn thuật toán trên hàm mục tiêu Huber, mỗi thuật toán
  giữ nguyên cấu hình đã chọn cho bài toán Ridge ở chương~\ref{ch:headline}.}{fig:huber-headline}
```

Ba dòng bình luận `%` đánh dấu chỗ phải điền văn xuôi ở lượt B. Không để lại dòng
`%` nào sau khi viết xong.

Hai bảng cần có, mỗi bảng đủ `\caption` và `\label`:

- `tab:huber-constants`: $L$, $\mu$, $\kappa$, $f^*$ của hai bài toán đặt cạnh nhau, thêm cột $\mu$ đo tại $w^*$ để thấy cận $\lambda$ lỏng bao nhiêu.
- `tab:huber-newton`: gap từng vòng của Newton, cột thứ hai là gap của biến thể dây cung, để đuôi bậc hai và đuôi tuyến tính nằm cạnh nhau.

Ba điểm bắt buộc phải có mặt trong phần văn xuôi, vì cả ba là chỗ thực nghiệm lệch khỏi dự đoán của kế hoạch:

1. Backtracking không lùi bước một lần nào với hướng Newton. Kế hoạch dự đoán ngược lại, và cơ chế là hướng Newton đã mang sẵn thông tin độ cong nên bước đầy đủ thỏa điều kiện giảm đủ ngay.
2. Sàn số học của các hình trong chương này cao hơn sàn của các chương trước, vì Huber không có dạng toàn phương cho gap. Ghi rõ sàn đo được và ghi rõ rằng đường Newton dừng ở sàn chứ không dừng vì thuật toán hết khả năng.
3. Cận $\mu \ge \lambda$ là cận cho trường hợp xấu nhất; giá trị đo tại $w^*$ lớn hơn, nên tốc độ quan sát được sẽ nhanh hơn cận lý thuyết tính từ $\kappa = 288{,}3$.

- [ ] **Step 5: Lượt C, rà theo mục 6 của `docs/van-phong-tieng-viet.md`, mỗi lần một nhóm**

Ba lượt rà riêng: nhóm 1 giàn giáo, nhóm 2 câu và chủ ngữ, nhóm 3 từ vựng và khuôn lặp. Không rà cả ba trong một lượt.

- [ ] **Step 6: Thêm hai frame vào `report/slides.tex`, sau frame "Bậc hai: một vòng lặp, sai số bằng không"**

Chèn vào sau dòng 346, trước dòng phân cách `% ======================= PHẦN 5.`.
Các con số dưới đây thay bằng số thật ghi lại ở Task 6 bước 5; chúng đang là số
của phép thăm dò và chỉ đúng tới chỗ làm tròn.

```latex
\begin{frame}{Bỏ tính bậc hai: Newton cần tám vòng, ba vòng cuối bậc hai}
  \[
    f_{\text{huber}}(w) = \frac{1}{n}\sum_{i} H_{\delta}(x_i\trans w - y_i)
      + \frac{\lambda}{2}\norm{w}^{2}, \qquad \delta = 4{,}07
  \]
  \begin{center}\small
    \begin{tabular}{rll}
      \toprule
      Vòng & $f - \fstar$, Newton & $f - \fstar$, Hessian dùng lại \\
      \midrule
      5 & $1{,}10 \cdot 10^{-2}$  & \\
      6 & $1{,}98 \cdot 10^{-5}$  & \\
      7 & $6{,}66 \cdot 10^{-11}$ & \\
      \bottomrule
    \end{tabular}
  \end{center}
  \chotlai{Số mũ nhân đôi mỗi vòng, đúng dạng hội tụ bậc hai. Dùng lại Hessian
    biến Newton thành phương pháp dây cung và đuôi đó biến mất.}
  \note{19,9 phần trăm số dòng nằm ngoài vùng bậc hai, nên hàm thật sự không còn
    là hàm bậc hai.}
\end{frame}

\begin{frame}{Backtracking có việc với gradient, không có việc với Newton}
  \centering
  \resultgraphic[0.62]{huber-step_cost}
  \chotlai{GD tốn $2{,}81$ lần đánh giá hàm mỗi vòng, Newton tốn đúng một lần và
    nhận ngay bước đầy đủ ở cả tám vòng.}
  \note{Ngược với dự đoán ở mục 4.4 của kế hoạch. Hướng Newton đã mang sẵn thông
    tin độ cong nên bước đầy đủ thỏa điều kiện giảm đủ ngay.}
\end{frame}
```

Cột thứ ba của bảng để trống trong khung này vì số của biến thể dây cung chỉ có
sau khi chạy Task 6; điền nốt ở bước này, không để trống khi commit.

Nếu hình bản báo cáo quá dày đường cho máy chiếu thì tạo bản riêng vào `results/figures/slides/` giữ nguyên tên file, theo mục 3 của `CLAUDE.md`.

- [ ] **Step 7: Chạy test báo cáo**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: PASS. Test này bắt hình thiếu `\ref`, nhãn trùng, và mục `refs.bib` không được trích dẫn.

- [ ] **Step 8: Biên dịch báo cáo và slide**

Run: `cd report && latexmk -xelatex report.tex && latexmk -xelatex slides.tex`
Expected: biên dịch xong, không còn cảnh báo tham chiếu hỏng. Kiểm tra rằng chương mới là chương 8 và các chương sau đã dịch số đúng.

- [ ] **Step 9: Commit**

```bash
git add report/report.tex report/slides.tex report/refs.bib
git commit -m "docs: add the Huber chapter to the report and two slides"
```

---

## Task 8: Cập nhật kế hoạch gốc

**Files:**
- Modify: `KE_HOACH_TRIEN_KHAI.md` (mục 4.4 ở dòng 356, mục 4.5 ở dòng 374, mục 11 ở dòng 712, mục 13 ở cuối, mục 14 ở dòng 993)

**Interfaces:**
- Consumes: các con số từ Task 6 và nội dung chương từ Task 7.
- Produces: không có mã, đây là task cuối.

- [ ] **Step 1: Đọc `docs/van-phong-tieng-viet.md`**

Phần thêm vào mục 13 là văn phân tích kết quả, nên đi đủ ba lượt như Task 7.

- [ ] **Step 2: Viết lại mục 4.4**

Bỏ câu "Phần này tùy chọn" và câu dự đoán rằng Huber sẽ làm backtracking có nội dung. Thay bằng cái đã làm: $\delta$ chọn thế nào, hai nhóm thí nghiệm nào, và kết quả thật của Newton cùng của backtracking.

- [ ] **Step 3: Sửa mục 4.5**

Ghi rõ phần thuật toán mở rộng đã bỏ theo quyết định ngày 13 tháng 8 năm 2026, giữ nguyên bảng năm ứng viên để người đọc thấy chỗ mở rộng được, và nói rõ yêu cầu khuyến khích của đề bài được đáp ứng bằng hàm mục tiêu thứ hai ở mục 4.4.

- [ ] **Step 4: Sửa dòng tương ứng trong bảng đối chiếu mục 11**

Dòng "Áp dụng thêm thuật toán khác (khuyến khích)" đổi cột thứ hai từ "Mục 4.4 và 4.5" thành "Mục 4.4, hàm mục tiêu Huber, chương 8 của báo cáo".

- [ ] **Step 5: Thêm ba mục vào mục 13**

Đánh số tiếp theo 13.21, tức 13.22 tới 13.24, mỗi mục một tiêu đề mang kết luận chứ không mang chủ đề:

- Backtracking không lùi bước với hướng Newton, ngược với dự đoán ở mục 4.4.
- Sàn số học của bài toán Huber cao hơn của Ridge, và vì sao dạng toàn phương không dùng được.
- Chi phí dựng Hessian của Huber so với của Ridge, kèm con số đo ở Task 6.

- [ ] **Step 6: Cập nhật mục 14**

Danh sách "Việc cần làm ngay" hiện mô tả trạng thái của tuần đầu tiên và đã hết đúng từ lâu. Thay bằng trạng thái hiện tại: những gì đã xong và những gì còn lại trước khi nộp.

- [ ] **Step 7: Chạy toàn bộ test lần cuối**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add KE_HOACH_TRIEN_KHAI.md
git commit -m "docs: record the Huber results and the dropped extension section"
```
