"""The single result structure every run produces, and its JSON form.

One shape for every algorithm is what makes the comparison plots safe to draw:
two curves can only be put on the same axes if they were logged at the same kind
of checkpoint, with the same clock discipline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

Status = str  # "converged" | "max_iter" | "diverged" | "stalled" | "line_search_failed"


@dataclass(slots=True)
class RunRecord:
    """History of one run, sampled at the checkpoints chosen by `record_every`.

    Entry i describes the iterate at `iters[i]`, before the step taken there. All
    cumulative quantities are measured from the start of the run.
    """

    label: str
    """Legend text, naming the parameters rather than just the method."""
    status: Status

    direction_label: str = ""
    step_label: str = ""
    """The two axes kept separate as well as joined, so figures can group by one
    of them without parsing `label` back apart."""

    iters: list[int] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    """f(w_k) - f*, from the exact quadratic form, not a difference of floats."""
    gnorms: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    """Cumulative compute seconds. Logging time is excluded by construction."""
    rows_seen: list[int] = field(default_factory=list)
    """Cumulative rows of X touched, the axis on which SGD compares fairly."""
    fevals: list[int] = field(default_factory=list)
    """Cumulative objective evaluations spent inside the line search."""

    w_final: list[float] = field(default_factory=list)
    is_stochastic: bool = False
    rows_per_epoch: int = 0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- accessors

    @property
    def epochs(self) -> np.ndarray:
        """Cumulative passes over the data, for the third axis of the SGD plots."""
        if self.rows_per_epoch <= 0:
            raise ValueError(f"{self.label}: rows_per_epoch was never set")
        return np.asarray(self.rows_seen, dtype=float) / self.rows_per_epoch

    @property
    def total_time(self) -> float:
        return self.times[-1] if self.times else 0.0

    @property
    def final_gap(self) -> float:
        return self.gaps[-1] if self.gaps else float("nan")

    @property
    def fevals_per_iter(self) -> float:
        """Average line-search probes per iteration.

        The number that explains how backtracking can win on iteration count and
        still lose on wall-clock time.
        """
        if not self.iters or self.iters[-1] == 0:
            return 0.0
        return self.fevals[-1] / self.iters[-1]

    def _first_index_below(self, threshold: float) -> int | None:
        for i, gap in enumerate(self.gaps):
            if gap < threshold:
                return i
        return None

    def iters_to_reach(self, threshold: float = 1e-6) -> int | None:
        """Iterations needed to bring f - f* below `threshold`, or None."""
        i = self._first_index_below(threshold)
        return None if i is None else self.iters[i]

    def time_to_reach(self, threshold: float = 1e-6) -> float | None:
        i = self._first_index_below(threshold)
        return None if i is None else self.times[i]

    # ------------------------------------------------------------------ i/o

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RunRecord":
        return cls(**payload)


def save_records(records: list[RunRecord], path: str | Path, meta: dict | None = None) -> Path:
    """Write one experiment group to JSON so figures can be redrawn without rerunning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "records": [r.to_dict() for r in records]}
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_records(path: str | Path) -> tuple[list[RunRecord], dict]:
    payload = json.loads(Path(path).read_text())
    return [RunRecord.from_dict(r) for r in payload["records"]], payload["meta"]
