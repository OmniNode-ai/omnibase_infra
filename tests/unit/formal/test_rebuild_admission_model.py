# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the M0 model of the stop gate, WIP admission and holder (OMN-22478).

Reads the committed ``formal/rebuild_admission/results``: every base and liveness config holds,
every mutation, control and witness reports its own named property, and the digest ties the
results to the spec and cfgs as they stand. A break that stops failing fails this file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[3] / "formal" / "rebuild_admission"
RESULTS = MODEL / "results"
FAILS = ("mut_", "wit_", "ctl_")
HOLDS = ("Base_", "live_")


def _configs() -> list[str]:
    return sorted(p.stem for p in MODEL.glob("*.cfg"))


def _out(name: str) -> str:
    return (RESULTS / f"{name}.out").read_text(encoding="utf-8")


@pytest.mark.unit
def test_results_match_the_model() -> None:
    digest = hashlib.sha256()
    for path in [
        *sorted(MODEL.glob("RebuildAdmission*.tla")),
        *sorted(MODEL.glob("*.cfg")),
    ]:
        digest.update(path.read_bytes())
    assert (RESULTS / "model.sha256").read_text().strip() == digest.hexdigest()


def _infeasible() -> dict[str, str]:
    """Configs the README declares too large for a lab run, each with its reason.

    A line ``INFEASIBLE <cfg>: <reason>`` in the README declares one. Such a config is not
    proven: the test reports it skipped, and it must have no result claiming it holds.
    """
    declared = {}
    for line in (MODEL / "README.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"INFEASIBLE (\w+): (.+)", line)
        if m:
            declared[m.group(1)] = m.group(2)
    return declared


@pytest.mark.unit
def test_infeasible_declarations_name_configs() -> None:
    for name, reason in _infeasible().items():
        assert (MODEL / f"{name}.cfg").is_file(), name
        assert name.startswith(HOLDS) and reason.strip(), name


@pytest.mark.unit
@pytest.mark.parametrize("name", [c for c in _configs() if c.startswith(HOLDS)])
def test_protocol_holds_exhaustively(name: str) -> None:
    declared = _infeasible()
    if name in declared:
        out = RESULTS / f"{name}.out"
        assert not (
            out.is_file()
            and "No error has been found" in out.read_text(encoding="utf-8")
        )
        pytest.skip(f"declared infeasible, not proven: {declared[name]}")
    text = _out(name)
    assert "Model checking completed. No error has been found." in text
    assert re.search(r"distinct states found, 0 states left on queue", text)


def _expected(name: str) -> str:
    """The property a break must report, read from its own cfg's INVARIANT or PROPERTY line."""
    lines = (MODEL / f"{name}.cfg").read_text(encoding="utf-8").splitlines()
    named = [
        ln.split()[1] for ln in lines if ln.startswith(("INVARIANT ", "PROPERTY "))
    ]
    assert len(named) == 1, f"{name} must break exactly one named property"
    return named[0]


@pytest.mark.unit
@pytest.mark.parametrize("name", [c for c in _configs() if c.startswith(FAILS)])
def test_each_break_reports_its_named_property(name: str) -> None:
    text = _out(name)
    prop = _expected(name)
    assert re.search(
        rf"Invariant {prop} is violated"
        rf"|Action property {prop} is violated"
        rf"|Temporal property {prop} was violated",
        text,
    ), name
    assert "Model checking completed. No error has been found." not in text
