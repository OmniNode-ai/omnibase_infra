# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed deploy job and a pin near its expiry each write an alarm row (OMN-24562).

The job reading (OMN-24617) records a failed job as a NOT_WORKING reading. These tests hold
the second row: an ALARM (``src=alarm``) with an owner line for a failed dev or stability
job, and for a code-ref pin within six hours of its expiry, written on the agent's idle tick.
"""

from __future__ import annotations

import ast
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
    handler = handler_mod
    pin = _pin(3)
    first = handler._pin_request_id(pin["branch"], pin["expires_at"], NOW)
    assert first == handler._pin_request_id(
        pin["branch"], pin["expires_at"], NOW + timedelta(minutes=20)
    )
    assert first != handler._pin_request_id(
        pin["branch"], pin["expires_at"], NOW + timedelta(hours=1)
    )


_REFUSING_APPEND = """#!INTERPRETER
import pathlib, sys
args = sys.argv[1:]
rid = args[args.index('--request-id') + 1]
row = sys.stdin.read()
seen = pathlib.Path('LEDGER') / rid
if seen.exists():
    if seen.read_text() == row:
        print('{"duplicate": true}')
        sys.exit(0)
    open('REFUSALS', 'a').write(rid + '\\n')
    sys.stderr.write('request id already belongs to different rows')
    sys.exit(65)
seen.parent.mkdir(exist_ok=True)
seen.write_text(row)
print('{"ok": true}')
"""


def _refusing_append(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """The work-ledger append's id rule: a row under a seen request id is a duplicate when
    identical and a refusal (exit 65) when different; ``refusals`` counts the refusals."""
    ledger = tmp_path / "ledger"
    refusals = tmp_path / "refusals.txt"
    script = tmp_path / "append.py"
    script.write_text(
        _REFUSING_APPEND.replace("INTERPRETER", sys.executable)
        .replace("LEDGER", str(ledger))
        .replace("REFUSALS", str(refusals))
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    environ = {
        "ONEX_LEDGER_BUS_APPEND_COMMAND": str(script),
        "ONEX_LEDGER_HOST_NAME": "test-host",
    }
    return environ, ledger, refusals


def test_a_simulated_pin_hour_has_no_refused_append_and_one_row_per_pin(
    tmp_path: Path,
) -> None:
    environ, ledger, refusals = _refusing_append(tmp_path)
    handler = _handler(environ)
    start = datetime(2026, 11, 16, 8, 0, 5, tzinfo=UTC)
    expires = start + timedelta(hours=3, minutes=20)
    outcomes = [
        handler.handle_pin(
            branch="integration/stab-integ2",
            pr=6012,
            ticket="OMN-24617",
            registered_by="stab-integ2-1750",
            expires_at=expires,
            now=start + timedelta(seconds=300 * tick),
        ).outcome
        for tick in range(12)
    ]
    assert outcomes == ["appended"] * 12
    assert not refusals.exists()
    assert len(list(ledger.iterdir())) == 1
    row = next(ledger.iterdir()).read_text()
    assert _cells(row.split(" | req=")[0])["hours_left"] == "3.3"


def test_the_next_clock_hour_writes_a_new_row(tmp_path: Path) -> None:
    environ, ledger, refusals = _refusing_append(tmp_path)
    handler = _handler(environ)
    expires = NOW + timedelta(hours=4)
    for hours in (0, 0.5, 1, 1.5):
        handler.handle_pin(
            branch="b",
            pr=1,
            ticket="OMN-1",
            registered_by="x",
            expires_at=expires,
            now=NOW + timedelta(hours=hours),
        )
    assert not refusals.exists()
    assert len(list(ledger.iterdir())) == 2


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


def _bare_agent(lanes: set[EnumRuntimeLane]) -> Any:
    agent = object.__new__(agent_mod.DeployAgent)
    agent._allowed_lanes = lanes
    return agent


def _pin_check_gate() -> ast.If:
    """The ``if`` of the idle tick that wraps the pin check call."""
    tree = ast.parse(Path(agent_mod.__file__).read_text(encoding="utf-8"))
    run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    return next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.If)
        and any("'check_pin'" in ast.unparse(inner) for inner in node.body)
        and not any(isinstance(inner, ast.If) for inner in node.body)
    )


def test_a_prod_agent_never_runs_the_pin_check() -> None:
    """The idle tick reaches the check only behind ``DEV in self._allowed_lanes``."""
    gate = _pin_check_gate()
    assert ast.unparse(gate.test) == "EnumRuntimeLane.DEV in self._allowed_lanes"
    assert not gate.orelse
    source = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert source.count('"check_pin"') == 1


def test_a_dev_agent_runs_the_pin_check_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Module:
        @staticmethod
        def check_pin() -> None:
            calls.append("check_pin")

    monkeypatch.setattr(agent_mod, "load_dev_boot_guard", lambda name: _Module)
    gate = _pin_check_gate()
    assert ast.unparse(gate.body[0]) == (
        "await self._offload(self._job_reading, 'check_pin')"
    )
    _bare_agent({EnumRuntimeLane.DEV})._job_reading("check_pin")
    assert calls == ["check_pin"]


def test_a_pin_check_load_error_does_not_stop_the_agent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _broken(name: str) -> Any:
        raise ImportError("the module set did not load")

    monkeypatch.setattr(agent_mod, "load_dev_boot_guard", _broken)
    agent = _bare_agent({EnumRuntimeLane.DEV})
    agent._job_reading("check_pin")
    agent._job_reading("check_pin")
    assert caplog.text.count("deploy job reading check_pin failed") == 2


def test_a_terminal_publish_error_does_not_skip_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken(name: str) -> Any:
        raise OSError("the contract could not be read")

    monkeypatch.setattr(agent_mod, "load_dev_boot_guard", _broken)
    _bare_agent({EnumRuntimeLane.DEV})._job_reading("publish", object(), agent_mod)


def test_the_terminal_publish_goes_through_the_guard_before_clear_settling() -> None:
    source = Path(agent_mod.__file__).read_text(encoding="utf-8")
    publish = source.index('self._job_reading("publish"')
    assert publish < source.index("self.job_store.clear_settling(cid)", publish)


def test_the_contract_is_parsed_once_and_a_failed_read_logs_once_per_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    reads: list[Path] = []
    original = runtime_mod._contract_settings

    def _counting(path: Path) -> Any:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(runtime_mod, "_contract_settings", _counting)
    monkeypatch.setattr(runtime_mod, "_pin_settings", [])
    monkeypatch.setattr(runtime_mod, "_pin_last_check", [])
    monkeypatch.delenv("DEPLOY_AGENT_STATE_DIR", raising=False)
    for _ in range(5):
        assert runtime_mod.check_pin(force=True) is None
    assert len(reads) == 1

    def _failing(path: Path) -> Any:
        raise OSError("unreadable")

    monkeypatch.setattr(runtime_mod, "_contract_settings", _failing)
    monkeypatch.setattr(runtime_mod, "_pin_settings", [])
    monkeypatch.setattr(runtime_mod, "_pin_last_check", [])
    caplog.clear()
    for _ in range(5):
        runtime_mod.check_pin()
    assert caplog.text.count("pin alarm check skipped") == 1


def test_the_alarm_owner_is_the_item_lane_and_names_no_host() -> None:
    settings = runtime_mod.load_settings(agent_mod)
    assert settings["alarm_owner"] == "swarm-item-lane-OMN-24562"
    assert "h201" not in str(settings["alarm_job_owner_line"])
