# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19215 AC2: the committed lab pair shows the placed sibling answering only when the rung is down.

The pair is two in-process ``onex delegate`` runs on a lab host: one whose
overlay binds the local rungs to a closed port, one unmodified control. The
receipts' attempts are committed as ``tests/fixtures/omn19215_ac2_lab_pair.json``; this test
reads them, so a later edit that drops or falsifies the pair turns it red.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_EVIDENCE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "omn19215_ac2_lab_pair.json"
)


def _load() -> dict[str, Any]:
    evidence: dict[str, Any] = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    return evidence


def _verdict(evidence: Mapping[str, Any]) -> list[str]:
    """Return every way the committed pair fails to show the AC2 behavior."""
    sibling = evidence["sibling_backend_id"]
    rung_host = evidence["rung_host"]
    sibling_host = evidence["sibling_host"]
    problems: list[str] = []
    runs = evidence["runs"]
    closed = [r for r in runs if r["kind"] == "closed_port"]
    control = [r for r in runs if r["kind"] == "control"]
    if not closed or not control:
        problems.append("the pair needs at least one closed-port run and one control")
    for run in closed:
        attempts = run["attempts"]
        first, last = attempts[0], attempts[-1]
        if (
            first["host"] != rung_host
            or first["acceptance_reason"] != "provider_call_failed"
        ):
            problems.append(f"{run['run_id']}: the rung's host did not fail first")
        if last["backend_id"] != sibling or last["host"] != sibling_host:
            problems.append(f"{run['run_id']}: the sibling did not answer")
        if last["acceptance_decision"] != "accept":
            problems.append(f"{run['run_id']}: the sibling's answer was not accepted")
    for run in control:
        for attempt in run["attempts"]:
            if attempt["backend_id"] == sibling or attempt["host"] == sibling_host:
                problems.append(
                    f"{run['run_id']}: the sibling was reached with the rung up"
                )
        if run["attempts"][-1]["host"] != rung_host:
            problems.append(f"{run['run_id']}: the rung's host did not answer")
    return problems


def test_committed_lab_pair_shows_the_sibling_answering_only_when_the_rung_is_down() -> (
    None
):
    assert _verdict(_load()) == []


def test_the_pair_has_several_runs_of_each_kind() -> None:
    runs = _load()["runs"]
    assert sum(r["kind"] == "closed_port" for r in runs) >= 2
    assert sum(r["kind"] == "control" for r in runs) >= 2


def test_a_control_that_reached_the_sibling_is_refused() -> None:
    evidence = _load()
    control = next(r for r in evidence["runs"] if r["kind"] == "control")
    control["attempts"][-1].update(
        backend_id=evidence["sibling_backend_id"], host=evidence["sibling_host"]
    )
    assert _verdict(evidence)


def test_a_closed_port_run_the_sibling_did_not_answer_is_refused() -> None:
    evidence = _load()
    closed = next(r for r in evidence["runs"] if r["kind"] == "closed_port")
    closed["attempts"] = closed["attempts"][:1]
    assert _verdict(evidence)
