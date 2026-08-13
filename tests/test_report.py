"""Checks on the LaTeX sources, run before every submission.

Section 5 of CLAUDE.md requires that every figure and table carry a caption and a
label and be referred to at least once, and that no entry in refs.bib go
uncited. Those rules are easy to state and easy to break silently: biblatex with
the numeric style drops an uncited entry without a word, and a figure nobody
refers to still compiles.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FIGURE_DIR = ROOT / "results" / "figures"

SOURCES = ["report.tex", "slides.tex", "preamble.tex"]


def read(name: str) -> str:
    path = REPORT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} has not been written yet")
    return path.read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    """Drop LaTeX comments so a commented-out figure does not count as used."""
    return re.sub(r"(?<!\\)%.*", "", text)


@pytest.fixture(scope="module")
def report() -> str:
    return strip_comments(read("report.tex"))


@pytest.fixture(scope="module")
def slides() -> str:
    return strip_comments(read("slides.tex"))


# ------------------------------------------------------------ labels and refs


def labels_of(text: str) -> list[str]:
    return re.findall(r"\\label\{([^}]*)\}", text) + re.findall(
        r"\\result(?:fig|pair)\{[^}]*\}\{(?:[^{}]|\{[^}]*\})*\}\{([^}]*)\}", text
    )


def refs_of(text: str) -> set[str]:
    found = set()
    for command in ("ref", "eqref", "autoref", "cref", "nameref", "pageref"):
        for group in re.findall(rf"\\{command}\{{([^}}]*)\}}", text):
            found.update(part.strip() for part in group.split(","))
    return found


def test_no_duplicate_labels(report):
    labels = labels_of(report)
    duplicates = {label for label in labels if labels.count(label) > 1}
    assert not duplicates, f"labels declared more than once: {sorted(duplicates)}"


def test_every_label_is_referenced(report):
    unused = sorted(set(labels_of(report)) - refs_of(report))
    assert not unused, f"labels never referenced: {unused}"


def test_every_reference_resolves(report):
    dangling = sorted(refs_of(report) - set(labels_of(report)))
    assert not dangling, f"references without a label: {dangling}"


# ------------------------------------------------------------------- figures


def figure_stems(text: str) -> list[str]:
    """Every results/figures stem the source asks for, expanded from the macros."""
    stems = []
    for stem in re.findall(r"\\resultfig\{([^}]*)\}", text):
        stems.append(stem)
    for stem in re.findall(r"\\resultpair\{([^}]*)\}", text):
        stems += [f"{stem}_iters", f"{stem}_time"]
    for stem in re.findall(r"\\resultgraphic(?:\[[^\]]*\])?\{([^}]*)\}", text):
        stems.append(stem)
    return stems


def test_no_figure_is_included_twice(report):
    stems = figure_stems(report)
    duplicates = {stem for stem in stems if stems.count(stem) > 1}
    assert not duplicates, f"the same figure appears under two labels: {sorted(duplicates)}"


@pytest.mark.parametrize("source", ["report.tex", "slides.tex"])
def test_every_requested_figure_exists(source):
    text = strip_comments(read(source))
    missing = [s for s in figure_stems(text) if not (FIGURE_DIR / f"{s}.pdf").exists()]
    assert not missing, (
        f"{source} asks for figures that are not in results/figures: {sorted(set(missing))}"
    )


def test_figures_are_included_through_the_macros(report):
    """A bare \\includegraphics bypasses the missing-figure fallback and the sizing."""
    bare = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", report)
    assert not bare, f"use \\resultfig or \\resultpair instead of raw includes: {bare}"


# ---------------------------------------------------------------- references


def test_every_bib_entry_is_cited(report):
    path = REPORT_DIR / "refs.bib"
    if not path.exists():
        pytest.skip("refs.bib has not been written yet")

    entries = set(re.findall(r"@\w+\{([^,]+),", path.read_text(encoding="utf-8")))
    cited = set()
    for group in re.findall(r"\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]*)\}", report):
        cited.update(part.strip() for part in group.split(","))

    uncited = sorted(entries - cited)
    assert not uncited, (
        f"refs.bib entries nobody cites: {uncited}. biblatex numeric drops them "
        "silently, so delete them or cite them."
    )
    unknown = sorted(cited - entries)
    assert not unknown, f"citations with no entry in refs.bib: {unknown}"


# ------------------------------------------------------------------ hygiene


def test_floats_are_not_pinned_in_place(report):
    """Section 5 forbids [h] and [H]: a float there cuts an argument in half."""
    bad = re.findall(r"\\begin\{(?:figure|table)\}\[([^\]]*)\]", report)
    offenders = [spec for spec in bad if "h" in spec.lower()]
    assert not offenders, f"figure or table placed with h or H: {offenders}"


def test_tables_have_no_vertical_rules(report):
    """booktabs style: no vertical rules."""
    specs = re.findall(r"\\begin\{tabular\}\{([^}]*)\}", report)
    offenders = [spec for spec in specs if "|" in spec]
    assert not offenders, f"tabular with vertical rules: {offenders}"


@pytest.mark.parametrize("source", SOURCES)
def test_sources_are_utf8_and_non_empty(source):
    assert len(read(source).strip()) > 0
