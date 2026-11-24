# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A refused dev boot report must not hold the agent's boot or its job worker.

Measured on the dev lane agent, 2026-11-24: 394 recorded incidents under
``dev-boot-guard/omnibase-infra`` each still owed a STATUS row and an exception, and the bus
append command refused every one of them (``Dev boot STATUS delivery still pending: append
command refused``, about 2.9 s each). ``deliver_reports`` walks every incident on every call,
and it is called at the agent's start before the health port binds (step 3 of ``run``) and
again every 30 s on the single job worker. After the self-update re-exec at 06:41:43 EST the
agent logged 394 refusals in 19 minutes 20 seconds and bound its health port at 07:01:03.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from deploy_agent import executor as executor_mod

guard_mod = executor_mod.load_dev_boot_guard()

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]

INCIDENT_COUNT = 12
# The recorded cost of one refused incident is about 2.9 s; the test clock charges more per
# read so that the bound shows in a handful of incidents.
CLOCK_STEP_SECONDS = 5.0
DAY = "2026-11-{day:02d}T07:12:23Z"


def _write_incidents(root: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = root / f"incident-{index:08x}.json"
        executor_mod._atomic_write_private_json(
            path,
            {
                "job": f"{index:08x}-0000-4000-8000-000000000000",
                "commit": "0bfd8b31555c30305ad4add1bfef6e33ac792e6c",
                "error": "RuntimeError: work ledger: ledger exited 1",
                "project": "omnibase-infra",
                "restored_revision": "a" * 40,
                "recovery_verified": True,
                "at": DAY.format(day=index + 1),
                "ledger_pending": True,
                "exception_pending": True,
            },
        )
        paths.append(path)
    return paths


def _declare_bus_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, exit_code: int
) -> Path:
    """A bus append command that records each call and answers with ``exit_code``."""
    append = tmp_path / "append.py"
    seen = tmp_path / "seen.txt"
    append.write_text(
        "import sys\n"
        f"open({str(seen)!r}, 'a').write(sys.stdin.read())\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_LEDGER_WRITE_VIA", "bus")
    monkeypatch.setenv(
        "ONEX_LEDGER_BUS_APPEND_COMMAND", f"{sys.executable} {append} --bus-lane dev"
    )
    monkeypatch.setenv("ONEX_LEDGER_HOST_NAME", "h-test")
    monkeypatch.setenv("DEPLOY_AGENT_REPORT_PLUGIN_DIR", str(tmp_path / "plugin"))
    return seen


def _clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every read of the guard's clock costs CLOCK_STEP_SECONDS."""
    now = [0.0]

    def monotonic() -> float:
        now[0] += CLOCK_STEP_SECONDS
        return now[0]

    monkeypatch.setattr(
        guard_mod, "time", SimpleNamespace(monotonic=monotonic, sleep=lambda _s: None)
    )


def _route(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Real shell for the append command; the exception queue command is a Mock."""
    queue = Mock(
        return_value=subprocess.CompletedProcess([], 75, "", "bus unavailable")
    )

    def routed(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[0] != "bash":
            return queue(argv, **kwargs)
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=20, check=False
        )

    monkeypatch.setattr(executor_mod, "_run", routed)
    return queue


def test_refused_reports_stop_the_pass_after_one_report_command_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    _write_incidents(root, INCIDENT_COUNT)
    seen = _declare_bus_append(monkeypatch, tmp_path, exit_code=1)
    _route(monkeypatch)
    _clock(monkeypatch)

    executor_mod.deliver_dev_boot_reports(root)

    attempts = len(seen.read_text().splitlines())
    assert attempts < INCIDENT_COUNT
    assert attempts > 0
    assert attempts * CLOCK_STEP_SECONDS <= guard_mod.REPORT_SECONDS


def test_a_pass_tries_the_newest_incident_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    _write_incidents(root, INCIDENT_COUNT)
    seen = _declare_bus_append(monkeypatch, tmp_path, exit_code=1)
    _route(monkeypatch)
    _clock(monkeypatch)

    executor_mod.deliver_dev_boot_reports(root)

    first = seen.read_text().splitlines()[0]
    assert first.startswith(DAY.format(day=INCIDENT_COUNT))


def test_a_later_pass_delivers_every_incident_once_the_bus_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    paths = _write_incidents(root, INCIDENT_COUNT)
    seen = _declare_bus_append(monkeypatch, tmp_path, exit_code=0)
    # exit 0 without a receipt is not an accepted row: the pass sees a refusal each time.
    _route(monkeypatch)
    _clock(monkeypatch)
    executor_mod.deliver_dev_boot_reports(root)
    assert 0 < len(seen.read_text().splitlines()) < INCIDENT_COUNT

    seen.unlink()
    append = tmp_path / "append.py"
    append.write_text(
        "import json, sys\n"
        f"open({str(seen)!r}, 'a').write(sys.stdin.read())\n"
        "print(json.dumps({'status': 'accepted'}))\n",
        encoding="utf-8",
    )
    queue = _route(monkeypatch)
    queue.return_value = subprocess.CompletedProcess(
        [], 0, "EXCEPTION-QUEUE appended id=boot-failed", ""
    )
    executor_mod.deliver_dev_boot_reports(root)

    assert len(seen.read_text().splitlines()) == INCIDENT_COUNT
    for path in paths:
        stored = json.loads(path.read_text())
        assert stored["ledger_pending"] is False
        assert stored["exception_pending"] is False
