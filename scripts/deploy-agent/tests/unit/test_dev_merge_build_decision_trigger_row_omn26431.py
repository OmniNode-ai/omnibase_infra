# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent writes one ledger row per dev merge sha stating what it decided (OMN-26431).

THE FAILURE
-----------
After the dev merges of 2026-11-22 01:11Z to 01:30Z no lane could say, without GitHub reads,
whether a build had been requested for a merge sha. The agent decides that per command and per
idle tick (queued with a job id, deferred, or refused) and left no record of it.

These tests drive the agent's own decision sites (the idle converge tick and the command
execution) against a stand-in for the bus append command, and read the rows that reach it.
Select them with ``-k trigger_row``.
"""

from __future__ import annotations

import json
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import JOB_READING, DeployAgent
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.idle_converge import (
    PROBE_MARGIN,
    EnumIdleConvergeVerdict,
    load_probe_windows,
    probe_blocking,
)

# The idle converge harness of OMN-19509 builds the agent this file drives.
from .test_idle_converge_omn19509 import HEAD, QUIET, REPO_ROOT, RUNNING, _agent

pytestmark = pytest.mark.unit

runtime_mod = load_dev_boot_guard(JOB_READING)

#: The 02:24:59Z replay: a C15 run at 01:41Z that runs 65 minutes (config/lab_probe_windows.yaml).
C15_START = datetime(2026, 11, 22, 1, 41, tzinfo=UTC)
REPLAY_NOW = datetime(2026, 11, 22, 2, 24, 59, tzinfo=UTC)


def _cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    return dict(part.split("=", 1) for part in parts[2:] if "=" in part)


def _fake_append(tmp_path: Path, exit_code: int = 0) -> tuple[str, Path]:
    """A stand-in for the bus append command: one stdin row per invocation into a file."""
    sink = tmp_path / "rows.txt"
    script = tmp_path / "append.py"
    script.write_text(
        "import json, sys\n"
        f"open({str(sink)!r}, 'a').write(sys.stdin.read())\n"
        "print(json.dumps({'status': 'accepted'}))\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return f"{sys.executable} {script}", sink


def _lines(sink: Path, repo: str | None = None) -> list[str]:
    if not sink.exists():
        return []
    return [
        line
        for line in sink.read_text(encoding="utf-8").splitlines()
        if "src=build-decision" in line and (repo is None or f"repo={repo} " in line)
    ]


def _rows(sink: Path, repo: str | None = None) -> list[dict[str, str]]:
    return [_cells(line) for line in _lines(sink, repo)]


@pytest.fixture
def route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The agent's append route (the variables the node contract names) and a state dir."""
    command, sink = _fake_append(tmp_path)
    settings = runtime_mod.load_settings(agent_mod)
    monkeypatch.setenv(settings["append_command_env"], command)
    monkeypatch.setenv(settings["host_name_env"], "lab-host")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    (tmp_path / "agent-state").mkdir()
    # What the module remembers of one process's ticks is not what the next test's ticks see.
    runtime_mod._refs.clear()
    runtime_mod._not_routed_read.clear()
    return sink


def _git_diff(monkeypatch: pytest.MonkeyPatch, paths: str | None) -> None:
    """What ``git diff --name-only`` reads in the clone: ``paths``, or nothing (unreadable)."""
    real = subprocess.run

    def run(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, list) and argv[:1] == ["git"] and "diff" in argv:
            if paths is None:
                return subprocess.CompletedProcess(argv, 128, "", "no clone here")
            return subprocess.CompletedProcess(argv, 0, paths, "")
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def _dev_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, omnimarket_to_202: bool = True
) -> DeployAgent:
    agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=omnimarket_to_202)
    # The changed paths are unread unless a test says what they are.
    _git_diff(monkeypatch, None)
    return agent


def test_trigger_row_queued_for_a_converge_carries_the_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    agent = _dev_agent(tmp_path, monkeypatch)

    agent._maybe_idle_converge()

    jobs = sorted((tmp_path / "jobs").glob("*.json"))
    assert len(jobs) == 1
    rows = [r for r in _rows(route) if r.get("result") == "queued"]
    assert len(rows) == 1, f"no queued build-decision row reached the ledger: {rows}"
    row = rows[0]
    assert row["trigger_sha"] == HEAD
    assert row["class"] == "DETECTED_NOT_PUBLISHED"
    assert row["job"] == jobs[0].stem
    assert "reason" not in row


def test_trigger_row_queued_for_a_published_command_names_its_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    agent = _dev_agent(tmp_path, monkeypatch)
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="gha/omnibase_infra/runtime-rebuild",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
        git_ref="c" * 40,
    )
    agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    agent._execute_command(cmd)

    rows = _rows(route)
    assert len(rows) == 1, f"no build-decision row reached the ledger: {rows}"
    assert rows[0]["trigger_sha"] == "c" * 40
    assert rows[0]["class"] == "PUBLISHED"
    assert rows[0]["result"] == "queued"
    assert rows[0]["job"] == str(cmd.correlation_id)
    assert rows[0]["correlation"] == str(cmd.correlation_id)


def test_trigger_row_deferred_by_probe_window_has_earliest_start_from_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    agent = _dev_agent(tmp_path, monkeypatch)
    windows = load_probe_windows(REPO_ROOT)
    blocker = probe_blocking(REPLAY_NOW, windows)
    assert blocker == "C15 run at 2026-11-22T01:41Z"
    agent._idle_converge_now = lambda: REPLAY_NOW
    agent._idle_probe_blocker = lambda now: blocker

    agent._maybe_idle_converge()

    rows = _rows(route, "omnimarket")
    assert len(rows) == 1, f"no deferred build-decision row reached the ledger: {rows}"
    row = rows[0]
    assert row["result"] == "deferred"
    assert row["reason"] == "probe_window"
    c15 = next(w for w in windows if w.id == "C15")
    expected = (
        C15_START + timedelta(minutes=int(c15.max_duration_minutes)) + PROBE_MARGIN
    )
    assert row["earliest_start"] == expected.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert expected >= datetime(2026, 11, 22, 2, 46, tzinfo=UTC)
    assert "job" not in row


def test_trigger_row_refused_not_routed_elsewhere_names_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    # The committed table routes omnibase_infra rebuilds to this instance: the 02:08:45Z refusal.
    agent = _dev_agent(tmp_path, monkeypatch, omnimarket_to_202=False)
    refused = agent._decide_idle_converge("omnibase_infra", QUIET)
    assert refused.verdict is EnumIdleConvergeVerdict.NOT_ROUTED_ELSEWHERE

    rows = _rows(route, "omnibase_infra")
    assert len(rows) == 1, f"no refused build-decision row reached the ledger: {rows}"
    assert rows[0]["result"] == "refused"
    assert rows[0]["reason"] == "not_routed_elsewhere"
    assert "job" not in rows[0]


def test_trigger_row_for_a_merge_changing_no_runtime_path_is_non_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    agent = _dev_agent(tmp_path, monkeypatch)
    _git_diff(monkeypatch, "docs/overview.md\n")

    agent._maybe_idle_converge()

    rows = _rows(route, "omnimarket")
    assert len(rows) == 1, (
        f"no NON_RUNTIME build-decision row reached the ledger: {rows}"
    )
    assert rows[0]["class"] == "NON_RUNTIME"
    for cell in ("result", "job", "reason", "earliest_start"):
        assert cell not in rows[0], f"a NON_RUNTIME row must name no build {cell}"


def test_trigger_row_for_a_merge_changing_a_runtime_path_is_not_non_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    """Positive control for the class: one runtime path among the changed ones keeps the build."""
    agent = _dev_agent(tmp_path, monkeypatch)
    _git_diff(monkeypatch, "docs/overview.md\nsrc/omnibase_infra/runtime/kernel.py\n")

    agent._maybe_idle_converge()

    rows = _rows(route, "omnimarket")
    assert [r["class"] for r in rows] == ["DETECTED_NOT_PUBLISHED"]
    assert rows[0]["result"] == "queued"


def test_trigger_row_for_the_same_decision_twice_is_one_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    agent = _dev_agent(tmp_path, monkeypatch)
    agent._idle_probe_blocker = lambda now: "C15 run at 2026-11-22T01:41Z"

    for _ in range(3):
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()

    rows = _rows(route, "omnimarket")
    assert len(rows) == 1, f"a repeated decision must write one row, got {len(rows)}"

    # A new verdict for the same sha is a new row (key: sha + result + reason).
    agent._idle_probe_blocker = lambda now: None
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    results = [r["result"] for r in _rows(route, "omnimarket")]
    assert results == ["deferred", "queued"]

    # The dedupe is held in the agent's state directory, so a restart writes no second row.
    restarted = _dev_agent(tmp_path, monkeypatch)
    restarted._idle_probe_blocker = lambda now: "C15 run at 2026-11-22T01:41Z"
    restarted._maybe_idle_converge()
    probe_rows = [
        r for r in _rows(route, "omnimarket") if r.get("reason") == "probe_window"
    ]
    assert len(probe_rows) == 1


@pytest.mark.parametrize("failure", ["exit", "timeout", "no-route"])
def test_trigger_row_append_failure_changes_no_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: Path,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    settings = runtime_mod.load_settings(agent_mod)
    if failure == "exit":
        (tmp_path / "failing").mkdir()
        command, _ = _fake_append(tmp_path / "failing", exit_code=65)
        monkeypatch.setenv(settings["append_command_env"], command)
    elif failure == "timeout":
        handler = runtime_mod.handler_module

        def timed_out(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="append", timeout=30)

        monkeypatch.setattr(handler.subprocess, "run", timed_out)
    else:
        monkeypatch.delenv(settings["append_command_env"], raising=False)
    agent = _dev_agent(tmp_path, monkeypatch)

    with caplog.at_level("WARNING"):
        agent._maybe_idle_converge()

    jobs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (tmp_path / "jobs").glob("*.json")
    ]
    assert len(jobs) == 1 and jobs[0]["status"] == "success"
    assert HEAD in agent._idle_converge_attempted
    assert "build decision row" in caplog.text, "the failed append must be logged"
    assert runtime_mod.append_failures() >= 1, "the failed append must be counted"
    # never retried in a loop: once every repository has been decided, further ticks append nothing
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    count = runtime_mod.append_failures()
    for _ in range(3):
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
    assert runtime_mod.append_failures() == count


def test_trigger_row_passes_the_ledger_row_grammar_for_every_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Path
) -> None:
    """Each row is one STATUS line the ledger grammar takes: timestamp, type, lane, key=value cells."""
    agent = _dev_agent(tmp_path, monkeypatch)
    agent._idle_probe_blocker = lambda now: "C15 run at 2026-11-22T01:41Z"
    agent._maybe_idle_converge()
    agent._idle_probe_blocker = lambda now: None
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    lines = _lines(route, "omnimarket")
    assert len(lines) == 2, f"expected a deferred and a queued row, got {lines}"
    for line in lines:
        assert re.match(
            r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \| STATUS \| lane=\S+ \| ", line
        )
        cells = [c.strip() for c in line.split(" | ")]
        assert all(re.fullmatch(r"[a-z_]+=\S+", c) for c in cells[2:]), line
        assert len({c.split("=")[0] for c in cells[2:]}) == len(cells[2:]), (
            "a cell twice"
        )
        assert "\n" not in line
        assert re.search(r"\| req=[0-9a-f-]{36}$", line)
    # the grep a reader runs: one row per decision for a sha
    deferred, queued = (_cells(line) for line in lines)
    assert deferred["trigger_sha"] == queued["trigger_sha"] == HEAD
    assert RUNNING not in "".join(lines)
