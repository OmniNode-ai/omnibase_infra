# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22821 -- a cancelled or skipped run is no verdict in the delegation-health check.

The M4 C17 customer-surface verdict run 42288463416 at f16faf507e09 concluded
``cancelled`` and was the newest run, so the check read it as a failed verdict and
went red on every runtime pull request. A cancelled run measured nothing:

AC1: the newest COMPLETED verdict decides when newer runs were cancelled/skipped.
AC2: a run that completed with ``failure`` (the failing control) still fails.
AC3: a source whose every admitted run is cancelled/skipped is refused, not passed.
AC4: the shared reader's default is unchanged: staging delivery and the lab alarm
     still read a cancelled newest run as a block.

The run list is the recorded shape of 2026-11-07: the cancelled run's id and head are
the recorded ones; the older completed runs are fixture rows.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import delegation_health_check as dh
from scripts.ci.lab_pass_receipt import evaluate_workflow_verdict

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 7, 0, 52, 0, tzinfo=UTC)
CONFIG = Path("config/delegation_health_check.yaml")
CANCELLED_ID = 42288463416
CANCELLED_HEAD = "f16faf507e09" + "0" * 28
M4 = "m4-c17-customer-surface-verdict.yml"


def _run(
    run_id: int, *, conclusion: str, hours_ago: float, head: str = "0" * 40
) -> dict[str, Any]:
    stamp = (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": run_id,
        "conclusion": conclusion,
        "created_at": stamp,
        "run_started_at": stamp,
        "run_attempt": 1,
        "event": "schedule",
        "display_title": "verdict",
        "head_branch": "dev",
        "head_sha": head,
        "status": "completed",
        "html_url": f"https://example.invalid/runs/{run_id}",
    }


def _recorded_m4() -> list[dict[str, Any]]:
    """Newest first: the recorded cancelled run, then older completed fixture rows."""
    return [
        _run(CANCELLED_ID, conclusion="cancelled", hours_ago=1.0, head=CANCELLED_HEAD),
        _run(900002, conclusion="success", hours_ago=3.0),
        _run(900001, conclusion="failure", hours_ago=11.0),
    ]


def _replay(monkeypatch: Any, m4_runs: list[dict[str, Any]]) -> None:
    def fake(path: str) -> bytes:
        if f"/workflows/{M4}/" in path:
            return json.dumps({"workflow_runs": m4_runs}).encode()
        return json.dumps(
            {"workflow_runs": [_run(1, conclusion="success", hours_ago=2.0)]}
        ).encode()

    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", fake)


def _check() -> tuple[int, str]:
    out = io.StringIO()
    code = dh.evaluate_delegation_health(
        dh.load_config(CONFIG).sources,
        runtime_affecting=True,
        labels=(),
        out=out,
        now=NOW,
        record={},
    )
    return code, out.getvalue()


def test_config_declares_cancelled_and_skipped_as_no_verdict_for_both_sources() -> None:
    sources = dh.load_config(CONFIG).sources
    assert len(sources) == 2
    for source in sources:
        assert set(source.no_verdict_conclusions) == {"cancelled", "skipped"}


def test_a_cancelled_newest_run_is_skipped_and_the_completed_verdict_is_read(
    monkeypatch: Any,
) -> None:
    _replay(monkeypatch, _recorded_m4())
    code, output = _check()
    assert code == 0, output
    assert "m4-c17-customer-surface-verdict: green, run 900002" in output
    assert "other(s) ignored" in output


def test_a_skipped_newest_run_is_also_no_verdict(monkeypatch: Any) -> None:
    runs = [_run(900009, conclusion="skipped", hours_ago=0.5), *_recorded_m4()[1:2]]
    _replay(monkeypatch, runs)
    code, output = _check()
    assert code == 0, output
    assert "run 900002" in output


def test_failing_control_a_completed_failure_still_fails(monkeypatch: Any) -> None:
    runs = [
        _run(CANCELLED_ID, conclusion="cancelled", hours_ago=1.0),
        _run(900003, conclusion="failure", hours_ago=2.0),
        _run(900002, conclusion="success", hours_ago=3.0),
    ]
    _replay(monkeypatch, runs)
    code, output = _check()
    assert code == 1
    assert "m4-c17-customer-surface-verdict: red, run 900003" in output
    assert "'failure'" in output


def test_a_newer_failure_after_a_green_still_fails_even_with_cancelled_between(
    monkeypatch: Any,
) -> None:
    runs = [
        _run(900005, conclusion="timed_out", hours_ago=0.5),
        _run(CANCELLED_ID, conclusion="cancelled", hours_ago=1.0),
        _run(900002, conclusion="success", hours_ago=3.0),
    ]
    _replay(monkeypatch, runs)
    code, output = _check()
    assert code == 1
    assert "run 900005" in output


def test_only_cancelled_runs_is_an_absent_measurement_not_a_pass(
    monkeypatch: Any,
) -> None:
    runs = [
        _run(CANCELLED_ID, conclusion="cancelled", hours_ago=1.0),
        _run(900007, conclusion="skipped", hours_ago=5.0),
    ]
    _replay(monkeypatch, runs)
    code, output = _check()
    assert code == 1
    assert "an absent measurement is not a pass" in output


def test_a_stale_completed_verdict_behind_a_cancelled_run_is_still_stale(
    monkeypatch: Any,
) -> None:
    runs = [
        _run(CANCELLED_ID, conclusion="cancelled", hours_ago=1.0),
        _run(900002, conclusion="success", hours_ago=9.0),
    ]
    _replay(monkeypatch, runs)
    code, output = _check()
    assert code == 1
    assert "past the 8h bound" in output


def test_shared_reader_default_still_blocks_on_a_cancelled_newest_run(
    monkeypatch: Any,
) -> None:
    _replay(monkeypatch, _recorded_m4())
    out = io.StringIO()
    code = evaluate_workflow_verdict(
        "OmniNode-ai/omnimarket", M4, "dev", 8.0, ["schedule"], out, now=NOW
    )
    assert code == 1
    assert "concluded 'cancelled'" in out.getvalue()
