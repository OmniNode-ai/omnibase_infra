# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15547 incident replay for ``scripts/ci/delegation_health_check.py`` (OMN-19451).

THE REGRESSION BEING REPLAYED is the absence of any merge-time reader of the
delegation verdict. The delegation regression nightly (omnimarket
``delegation-regression-nightly.yml``) concluded ``failure`` in run
35832924275 (2026-09-23, five hard breaks against the stability-test lane), and
nothing slowed runtime merges while it stood red: auto-merge, the landing lane
and humans read no delegation verdict.

THE ARTIFACT is that run, byte for byte as the REST API returned it
(``gh api repos/OmniNode-ai/omnimarket/actions/runs/35832924275``). The
replay wraps it in the run listing the reader consumes and drives the REAL
check over it: a runtime-affecting PR must be refused, naming the run.

THE DISCRIMINATOR is run 36394450393, the newest green scheduled nightly at
capture time, captured the same way. The same check over the same code path
must admit it; a check that refused everything would replay the incident
perfectly and block every runtime PR forever.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import delegation_health_check as dh

pytestmark = pytest.mark.unit

FIXTURES = Path("tests/fixtures/omn19451")
RED = FIXTURES / "omnimarket-run-35832924275.gh-api.json.captured"
GREEN = FIXTURES / "omnimarket-run-36394450393.gh-api.json.captured"


def _drive(monkeypatch: Any, fixture: Path) -> tuple[int, str]:
    run = json.loads(fixture.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
    now = started + timedelta(hours=2)
    nightly = next(
        s
        for s in dh.load_config(Path("config/delegation_health_check.yaml")).sources
        if s.name == "delegation-regression-nightly"
    )
    monkeypatch.setattr(
        "scripts.ci.lab_pass_receipt._gh_api",
        lambda _path: json.dumps({"workflow_runs": [run]}).encode(),
    )
    out = io.StringIO()
    code = dh.evaluate_delegation_health(
        (nightly,),
        runtime_affecting=True,
        labels=(),
        out=out,
        now=now.astimezone(UTC),
        record={},
    )
    return code, out.getvalue()


def test_the_real_check_refuses_a_runtime_pr_while_the_real_red_nightly_stands(
    monkeypatch: Any,
) -> None:
    code, output = _drive(monkeypatch, RED)
    assert code == 1
    assert "35832924275" in output


def test_the_same_check_admits_the_real_green_nightly(monkeypatch: Any) -> None:
    code, output = _drive(monkeypatch, GREEN)
    assert code == 0
    assert "36394450393" in output


def test_the_two_artifacts_are_different_real_runs() -> None:
    red = json.loads(RED.read_text(encoding="utf-8"))
    green = json.loads(GREEN.read_text(encoding="utf-8"))
    assert red["conclusion"] == "failure"
    assert green["conclusion"] == "success"
    assert red["id"] != green["id"]
