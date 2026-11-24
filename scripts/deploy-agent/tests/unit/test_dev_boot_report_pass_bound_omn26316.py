# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316 -- a backlog of refused dev boot incidents cannot hold the agent.

After every deploy-agent re-exec the dev agent served nothing for about twenty minutes:
``DeployAgent.run`` retried the dev boot incident reports at startup, before the health site
started, and 394 pending incidents (the oldest 2026-10-27) were each refused by the bus append
command. The ledger refuses a row stamped more than a day behind its own clock, and the retried
row carried the incident's stamp, so an incident a day old could never be delivered.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane

guard_mod = executor_mod.load_dev_boot_guard()
from omnibase_infra.models.model_dev_boot_report_config import (
    ModelDevBootReportConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]

BACKLOG = 394
PASS_SECONDS = 0.5
APPEND_SECONDS = 0.05
ABANDON_AFTER = 7 * 24 * 3600.0


def _stamp(age: timedelta) -> str:
    return (datetime.now(UTC) - age).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_incident(root: Path, job: str, at: str) -> Path:
    path = root / f"incident-{job}.json"
    executor_mod._atomic_write_private_json(
        path,
        {
            "job": job,
            "commit": "a" * 40,
            "error": "BOOT FAILED",
            "project": "omnibase-infra",
            "restored_revision": "b" * 40,
            "recovery_verified": True,
            "at": at,
            "ledger_pending": True,
            "exception_pending": False,
        },
    )
    return path


def _refusing_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bus append command that takes APPEND_SECONDS, records the call and exits 65 (refused)."""
    calls = tmp_path / "append-calls.log"
    script = tmp_path / "append.py"
    script.write_text(
        "import sys, time\n"
        f"time.sleep({APPEND_SECONDS})\n"
        f"open({str(calls)!r}, 'a').write(sys.stdin.read())\n"
        "sys.exit(65)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_LEDGER_WRITE_VIA", "bus")
    monkeypatch.setenv(
        "ONEX_LEDGER_BUS_APPEND_COMMAND", f"{sys.executable} {script} --bus-lane dev"
    )
    monkeypatch.setenv("ONEX_LEDGER_HOST_NAME", "h-test")

    def routed(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=20, check=False
        )

    monkeypatch.setattr(executor_mod, "_run", routed)
    return calls


def _config(
    tmp_path: Path, *, pass_seconds: float = PASS_SECONDS
) -> ModelDevBootReportConfig:
    return ModelDevBootReportConfig(
        registry_root=tmp_path,
        plugin_dir=tmp_path / "plugin",
        ledger_via_bus=True,
        pass_seconds=pass_seconds,
        abandon_after_seconds=ABANDON_AFTER,
    )


def _call_count(calls: Path) -> int:
    return calls.read_text().count("| STATUS |") if calls.exists() else 0


def test_pass_with_394_refused_incidents_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    # Inside the abandon bound, so every one of them is retried and every one is refused.
    for index in range(BACKLOG):
        _write_incident(root, f"{index:04x}", _stamp(timedelta(hours=2)))
    calls = _refusing_append(tmp_path, monkeypatch)

    started = time.monotonic()
    guard_mod.deliver_reports(root, execution=executor_mod, config=_config(tmp_path))
    elapsed = time.monotonic() - started

    # One pass costs the bound plus the one append in flight, never BACKLOG appends.
    assert elapsed < PASS_SECONDS + 2.0, elapsed
    assert 0 < _call_count(calls) < BACKLOG // 4
    # Nothing was dropped: every incident is still pending for the next pass.
    pending = [
        json.loads(p.read_text())["ledger_pending"]
        for p in root.glob("incident-*.json")
    ]
    assert len(pending) == BACKLOG and all(pending)


def test_incident_older_than_the_bound_is_abandoned_with_one_log_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "reports"
    old = _write_incident(root, "oldjob01", "2026-10-27T10:00:00Z")
    calls = _refusing_append(tmp_path, monkeypatch)

    with caplog.at_level(logging.WARNING):
        guard_mod.deliver_reports(
            root, execution=executor_mod, config=_config(tmp_path)
        )
        guard_mod.deliver_reports(
            root, execution=executor_mod, config=_config(tmp_path)
        )

    stored = json.loads(old.read_text())
    assert stored["ledger_pending"] is False and stored["exception_pending"] is False
    assert stored["delivered_abandoned_at"]
    assert _call_count(calls) == 0
    # One line for the move; the second pass finds nothing pending and says nothing.
    lines = [r for r in caplog.records if "delivered-abandoned" in r.getMessage()]
    assert len(lines) == 1 and "oldjob01" in lines[0].getMessage()


def test_incident_inside_the_bound_is_still_retried_with_the_delivery_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    at = _stamp(timedelta(days=3))
    path = _write_incident(root, "midjob01", at)
    calls = _refusing_append(tmp_path, monkeypatch)

    guard_mod.deliver_reports(root, execution=executor_mod, config=_config(tmp_path))

    assert json.loads(path.read_text())["ledger_pending"] is True
    row = calls.read_text().strip()
    # The ledger's clock guard refuses a row stamped over a day behind: the row carries
    # the delivery time, and the incident's own time rides in the evidence.
    assert not row.startswith(at)
    assert datetime.strptime(row.split(" | ")[0], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    ) > datetime.now(UTC) - timedelta(minutes=5)
    assert f"incident_at={at}" in row


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    agent = DeployAgent(skip_self_update=True)
    agent._allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    return agent


def test_agent_report_pass_never_holds_the_consumer_or_the_health_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    for index in range(BACKLOG):
        _write_incident(root, f"{index:04x}", _stamp(timedelta(hours=2)))
    calls = _refusing_append(tmp_path, monkeypatch)
    agent = _agent(tmp_path, monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        agent_mod,
        "deliver_reports",
        lambda _root: guard_mod.deliver_reports(
            root, execution=executor_mod, config=config
        ),
    )

    started = time.monotonic()
    # The startup path (step 3) and the job-thread retry tick both come through here.
    agent._retry_pending_publishes()
    assert time.monotonic() - started < PASS_SECONDS, "a report pass held the caller"
    assert agent._report_pass._thread is not None

    # A second tick while the pass runs starts no second pass.
    first = agent._report_pass._thread
    agent._retry_pending_publishes()
    assert agent._report_pass._thread is first

    agent._report_pass.join(timeout=PASS_SECONDS + 5)
    assert not first.is_alive()
    assert 0 < _call_count(calls) < BACKLOG // 4


def test_agent_report_pass_does_not_run_on_a_lane_without_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    agent._allowed_lanes = frozenset({EnumRuntimeLane.PROD})
    deliver = Mock()
    monkeypatch.setattr(agent_mod, "deliver_reports", deliver)
    agent._retry_pending_publishes()
    assert agent._report_pass._thread is None
    deliver.assert_not_called()
