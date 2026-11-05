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
    for path in [MODEL / "RebuildAdmission.tla", *sorted(MODEL.glob("*.cfg"))]:
        digest.update(path.read_bytes())
    assert (RESULTS / "model.sha256").read_text().strip() == digest.hexdigest()


@pytest.mark.unit
@pytest.mark.parametrize("name", [c for c in _configs() if c.startswith(HOLDS)])
def test_protocol_holds_exhaustively(name: str) -> None:
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
        rf"Invariant {prop} is violated|Temporal property {prop} was violated", text
    ) or (prop in text and "violated" in text), name
    assert "Model checking completed. No error has been found." not in text
