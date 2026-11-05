# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TLC evidence for the M0 model of the stop gate, WIP admission and holder (OMN-22478).

Reads the committed ``formal/rebuild_admission/results``: every base config holds, every
mutation, witness and finding reports its violation, and the digest ties the results to the
spec and cfgs as they stand. A mutation that stops failing fails this file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[3] / "formal" / "rebuild_admission"
RESULTS = MODEL / "results"
FAILS = ("mut_", "wit_", "finding_")


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
@pytest.mark.parametrize("name", [c for c in _configs() if c.startswith("Base_")])
def test_base_holds_exhaustively(name: str) -> None:
    text = _out(name)
    assert "Model checking completed. No error has been found." in text
    assert re.search(r"distinct states found, 0 states left on queue", text)


@pytest.mark.unit
@pytest.mark.parametrize("name", [c for c in _configs() if c.startswith(FAILS)])
def test_mutation_reports_its_violation(name: str) -> None:
    text = _out(name)
    assert re.search(
        r"Invariant \w+ is violated|Temporal property \w+ was violated", text
    )
    named = {
        "mut_p2_wip_counts_rows_only": "ClaimsWithinLimit",
        "mut_p1_gate_no_epoch": "NoStartAgainstOrder",
        "mut_p1_stale_before_safety": "SafetyCauseAdmitted",
        "mut_p3_no_escalation": "StopClearsOrEscalates",
        "mut_p4_keys_per_engine": "RestartRepeatsNoStep",
        "mut_p4_no_fence": "ExactlyOneOwner",
        "finding_f4_wedge": "FixedSafetyCloses",
    }
    if name in named:
        assert named[name] in text
