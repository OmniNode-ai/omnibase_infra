# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed deploy job and a pin near its expiry each write an alarm row (OMN-24562).

The job reading (OMN-24617) records a failed job as a NOT_WORKING reading. These tests hold
the second row: an ALARM (``src=alarm``) with an owner line for a failed dev or stability
job, and for a code-ref pin within six hours of its expiry, written on the agent's idle tick.
"""

from __future__ import annotations

import json
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import JOB_READING
from deploy_agent.events import DEPLOY_PHASE_ORDER, EnumRuntimeLane, Phase, PhaseStatus
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.job_state import JobState

pytestmark = pytest.mark.unit

runtime_mod = load_dev_boot_guard(JOB_READING)
handler_mod = runtime_mod.handler_module
request_mod = runtime_mod.request_module

DEV = EnumRuntimeLane.DEV.value
STABILITY = EnumRuntimeLane.STABILITY_TEST.value
PROD = EnumRuntimeLane.PROD.value
NOW = datetime(2026, 11, 16, 8, 0, 0, tzinfo=UTC)


def _cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    return dict(part.split("=", 1) for part in parts[2:] if "=" in part)


def _failed_request(lane: str, status: str = "failed") -> Any:
    results = dict.fromkeys(DEPLOY_PHASE_ORDER, PhaseStatus.SUCCESS)
    results[Phase.RUNTIME] = PhaseStatus.FAILED
    job = JobState(
        correlation_id=uuid4(),
        command={"runtime_lane": lane, "scope": "runtime"},
        status=status,
        phase_results=results,
        completed_at=NOW,
    )
    record = json.loads(job.model_dump_json())
    return request_mod.ModelDeployJobReadingRequest(
        correlation_id=record["correlation_id"],
        runtime_lane=lane,
        scope="runtime",
        job_outcome=status,
        phase_results=record["phase_results"],
        completed_at=record.get("completed_at"),
        published_at=NOW,
    )


def _handler(environ: dict[str, str] | None = None) -> Any:
    return handler_mod.HandlerDeployJobReading(
        settings=runtime_mod.load_settings(agent_mod), environ=environ or {}
    )


def _append_stub(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """An append command that writes each row it is given to a file, one per line."""
    sink = tmp_path / "rows.txt"
    script = tmp_path / "append.sh"
    script.write_text(f'#!/bin/sh\ncat >> "{sink}"\necho \'{{"ok": true}}\'\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    environ = {
        "ONEX_LEDGER_BUS_APPEND_COMMAND": str(script),
        "ONEX_LEDGER_HOST_NAME": "test-host",
    }
    return environ, sink


@pytest.mark.parametrize("lane", [DEV, STABILITY])
def test_failed_job_alarm_row_names_owner_and_step(lane: str) -> None:
    row = _handler().build_alarm_row(_failed_request(lane))
    assert row is not None
    cells = _cells(row)
    assert row.startswith("2026-11-16T08:00:00Z | STATUS | ")
    assert cells["src"] == "alarm"
    assert cells["alarm"] == "deploy-job-failed"
    assert cells["deploy_lane"] == lane
    assert cells["failing_step"] == "runtime"
    assert cells["owner"]
    assert cells["owner_line"]


@pytest.mark.parametrize(
    ("lane", "status"),
    [(PROD, "failed"), (DEV, "success"), (DEV, "in_progress")],
)
def test_no_alarm_for_prod_or_a_job_that_did_not_fail(lane: str, status: str) -> None:
    assert _handler().build_alarm_row(_failed_request(lane, status)) is None


def test_failed_job_appends_reading_then_alarm(tmp_path: Path) -> None:
    environ, sink = _append_stub(tmp_path)
    result = _handler(environ).handle(_failed_request(DEV))
    assert result.outcome == "appended"
    assert result.alarm_outcome == "appended"
    rows = sink.read_text().splitlines()
    assert len(rows) == 2
    assert _cells(rows[0])["src"] == "reading"
    assert _cells(rows[1])["src"] == "alarm"


def test_successful_job_writes_no_alarm(tmp_path: Path) -> None:
    environ, sink = _append_stub(tmp_path)
    result = _handler(environ).handle(_failed_request(DEV, "success"))
    assert result.alarm_outcome == "none"
    assert result.alarm_row == ""
    assert len(sink.read_text().splitlines()) == 1


def _pin(hours_left: float) -> dict[str, Any]:
    return {
        "branch": "integration/stab-integ2",
        "pr": 6012,
        "ticket": "OMN-24617",
        "registered_by": "stab-integ2-1750",
        "expires_at": NOW + timedelta(hours=hours_left),
        "now": NOW,
    }


@pytest.mark.parametrize("hours_left", [5.9, 6.0, 0.5])
def test_pin_inside_window_writes_alarm_row(hours_left: float) -> None:
    row = _handler().build_pin_alarm_row(**_pin(hours_left))
    assert row is not None
    cells = _cells(row)
    assert cells["src"] == "alarm"
    assert cells["alarm"] == "pin-near-expiry"
    assert cells["pin_owner"] == "stab-integ2-1750"
    assert cells["pin_pr"] == "6012"
    assert cells["owner"]
    assert cells["owner_line"]
    assert cells["expires_at"] == _pin(hours_left)["expires_at"].strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@pytest.mark.parametrize("hours_left", [6.1, 24.0])
def test_pin_outside_window_writes_nothing(hours_left: float, tmp_path: Path) -> None:
    environ, sink = _append_stub(tmp_path)
    assert _handler().build_pin_alarm_row(**_pin(hours_left)) is None
    assert _handler(environ).handle_pin(**_pin(hours_left)).outcome == "skipped"
    assert not sink.exists()


def test_pin_alarm_append_is_deduplicated_within_the_hour() -> None:
    handler = _handler()
    pin = _pin(3)
    first = handler.pin_request_id(pin["branch"], pin["expires_at"], NOW)
    assert first == handler.pin_request_id(
        pin["branch"], pin["expires_at"], NOW + timedelta(minutes=20)
    )
    assert first != handler.pin_request_id(
        pin["branch"], pin["expires_at"], NOW + timedelta(hours=1)
    )


def _write_pin(state_dir: Path, hours_left: float) -> None:
    guard = load_dev_boot_guard("handler_dev_lane_code_ref_pin")
    now = datetime.now(UTC)
    pin = guard.ModelDevLaneCodeRefPin(
        branch="integration/stab-integ2",
        head="a" * 40,
        pr=6012,
        ticket="OMN-24617",
        registered_by="stab-integ2-1750",
        registered_at=now - timedelta(hours=1),
        expires_at=now + timedelta(hours=hours_left),
    )
    guard.write_pin(state_dir, pin)


def test_idle_check_alarms_for_a_live_pin_inside_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ, sink = _append_stub(tmp_path)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    state = tmp_path / "state"
    monkeypatch.setenv("DEPLOY_AGENT_STATE_DIR", str(state))
    _write_pin(state, 2)
    result = runtime_mod.check_pin(force=True)
    assert result is not None
    assert result.alarm_outcome == "appended"
    assert _cells(sink.read_text().splitlines()[0])["alarm"] == "pin-near-expiry"


def test_idle_check_is_silent_for_a_far_pin_and_for_no_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ, sink = _append_stub(tmp_path)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    state = tmp_path / "state"
    monkeypatch.setenv("DEPLOY_AGENT_STATE_DIR", str(state))
    assert runtime_mod.check_pin(force=True) is None
    _write_pin(state, 48)
    result = runtime_mod.check_pin(force=True)
    assert result is not None
    assert result.outcome == "skipped"
    assert not sink.exists()


def test_idle_check_is_throttled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ, _ = _append_stub(tmp_path)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    state = tmp_path / "state"
    monkeypatch.setenv("DEPLOY_AGENT_STATE_DIR", str(state))
    _write_pin(state, 2)
    assert runtime_mod.check_pin(force=True) is not None
    assert runtime_mod.check_pin() is None


def test_agent_idle_tick_calls_the_pin_check() -> None:
    source = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert "load_dev_boot_guard(JOB_READING).check_pin" in source
    assert sys.modules[agent_mod.__name__] is agent_mod
