# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: a backlog of dev boot incidents against a real ledger clock guard (OMN-26316).

The deploy agent's own executor runs the handler's ``deliver_reports`` against incident files
it wrote itself and a real subprocess standing in for the bus append command. The stand-in
applies the ledger's clock guard: it exits 65 for a row stamped more than 24 hours behind its
own clock and prints an ``accepted`` receipt for any other. That is the refusal the dev agent
met at startup (394 incidents, the oldest from 2026-10-27), so the three behaviours that fix it
are exercised end to end: a pass is bounded, a row carries the delivery time, and an incident
past the abandon bound leaves the pending set.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod

from omnibase_infra.models.model_dev_boot_report_config import (
    ModelDevBootReportConfig,
)

pytestmark = pytest.mark.integration

guard_mod = executor_mod.load_dev_boot_guard()

BACKLOG = 120
PASS_SECONDS = 0.5
APPEND_SECONDS = 0.05

# A stand-in for the bus append command with the ledger's clock guard: stdin is one STATUS row.
_CLOCK_GUARD_APPEND = """\
import json, sys, time
from datetime import UTC, datetime, timedelta

time.sleep({append_seconds})
row = sys.stdin.read().strip()
stamp = datetime.strptime(row.split(" | ")[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
open({calls!r}, "a").write(row + "\\n")
if datetime.now(UTC) - stamp > timedelta(hours=24):
    sys.stderr.write("clock guard rejected append\\n")
    sys.exit(65)
print(json.dumps({{"status": "accepted"}}))
"""


def _stamp(age: timedelta) -> str:
    return (datetime.now(UTC) - age).strftime("%Y-%m-%dT%H:%M:%SZ")


def _incident(root: Path, job: str, at: str) -> Path:
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


def _bus_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    calls = tmp_path / "append-calls.log"
    script = tmp_path / "append.py"
    script.write_text(
        _CLOCK_GUARD_APPEND.format(append_seconds=APPEND_SECONDS, calls=str(calls)),
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


def _config(tmp_path: Path) -> ModelDevBootReportConfig:
    return ModelDevBootReportConfig(
        registry_root=tmp_path,
        plugin_dir=tmp_path / "plugin",
        ledger_via_bus=True,
        pass_seconds=PASS_SECONDS,
        abandon_after_seconds=7 * 24 * 3600.0,
    )


def _pending(root: Path) -> dict[str, bool]:
    return {
        p.name: json.loads(p.read_text())["ledger_pending"]
        for p in sorted(root.glob("incident-*.json"))
    }


def test_three_day_old_incident_is_delivered_under_the_ledger_clock_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    at = _stamp(timedelta(days=3))
    path = _incident(root, "midjob01", at)
    calls = _bus_append(tmp_path, monkeypatch)

    guard_mod.deliver_reports(root, execution=executor_mod, config=_config(tmp_path))

    # The row went out stamped with the delivery time, so the clock guard took it.
    assert json.loads(path.read_text())["ledger_pending"] is False
    row = calls.read_text().strip()
    assert not row.startswith(at)
    assert f"incident_at={at}" in row


def test_backlog_pass_is_bounded_and_loses_no_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    for index in range(BACKLOG):
        _incident(root, f"{index:04x}", _stamp(timedelta(hours=2)))
    calls = _bus_append(tmp_path, monkeypatch)
    config = _config(tmp_path)

    started = time.monotonic()
    guard_mod.deliver_reports(root, execution=executor_mod, config=config)
    elapsed = time.monotonic() - started

    delivered = sum(1 for pending in _pending(root).values() if not pending)
    # BACKLOG appends at APPEND_SECONDS each would take six seconds; the pass stops at the bound.
    assert elapsed < PASS_SECONDS + 2.0, elapsed
    assert 0 < delivered < BACKLOG
    assert len(calls.read_text().splitlines()) == delivered

    # Later passes finish the rest: nothing was dropped.
    for _ in range(BACKLOG):
        guard_mod.deliver_reports(root, execution=executor_mod, config=config)
        if not any(_pending(root).values()):
            break
    assert not any(_pending(root).values())


def test_incident_past_the_abandon_bound_leaves_the_pending_set_unsent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    path = _incident(root, "oldjob01", "2026-10-27T10:00:00Z")
    calls = _bus_append(tmp_path, monkeypatch)

    guard_mod.deliver_reports(root, execution=executor_mod, config=_config(tmp_path))

    stored = json.loads(path.read_text())
    assert stored["ledger_pending"] is False
    assert stored["delivered_abandoned_at"]
    assert not calls.exists()


def test_background_pass_returns_at_once_and_runs_one_pass_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    for index in range(BACKLOG):
        _incident(root, f"{index:04x}", _stamp(timedelta(hours=2)))
    _bus_append(tmp_path, monkeypatch)
    config = _config(tmp_path)
    background = guard_mod.BackgroundPass("test-reports")

    def work() -> None:
        guard_mod.deliver_reports(root, execution=executor_mod, config=config)

    started = time.monotonic()
    assert background.start(work) is True
    assert time.monotonic() - started < PASS_SECONDS / 2, "start held the caller"
    # A second start while the pass runs starts none.
    assert background.start(work) is False

    background.join(timeout=PASS_SECONDS + 5)
    assert 0 < sum(1 for pending in _pending(root).values() if not pending) < BACKLOG
    # The pass is over, so the next tick starts a fresh one.
    assert background.start(work) is True
    background.join(timeout=PASS_SECONDS + 5)
