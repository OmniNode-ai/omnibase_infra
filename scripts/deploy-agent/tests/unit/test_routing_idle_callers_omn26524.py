# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The routing key check and the routed-command alarm run from the agent's idle tick (OMN-26524).

OMN-26485 merged both handlers into ``node_deploy_job_reading_effect`` with no caller, so
both read NOT RUNNING on dev. The idle tick calls ``check_pin`` of the node's host
composition root; it now also runs ``check_routing`` and ``check_routed_commands`` (the
callers the node contract's ``operation_callers`` names), each writing one run line.
"""

from __future__ import annotations

import json
import logging
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import JOB_READING
from deploy_agent.events import DEPLOY_PHASE_ORDER, PhaseStatus
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.job_state import JobState

pytestmark = pytest.mark.unit

runtime_mod = load_dev_boot_guard(JOB_READING)
KEY = "unit-test-signing-key"
ALARM = "routed-command-no-job-record"


@pytest.fixture(autouse=True)
def agent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "jobs"
    state.mkdir()
    sink = tmp_path / "rows.txt"
    script = tmp_path / "append.sh"
    script.write_text(f'#!/bin/sh\ncat >> "{sink}"\necho \'{{"ok": true}}\'\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("ONEX_LEDGER_BUS_APPEND_COMMAND", str(script))
    monkeypatch.setenv("ONEX_LEDGER_HOST_NAME", "test-host")
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", KEY)
    monkeypatch.setenv("DEPLOY_AGENT_INSTANCE", "dev-201")
    monkeypatch.setattr(agent_mod, "STATE_DIR", state)
    monkeypatch.setattr(runtime_mod, "_routing_last", [])
    monkeypatch.setattr(runtime_mod, "_alarm_last", [])
    return state


def _rows(tmp_path: Path) -> list[str]:
    sink = tmp_path / "rows.txt"
    return sink.read_text().splitlines() if sink.is_file() else []


def _own_check() -> str:
    from deploy_agent.auth import hmac_key_check

    check = hmac_key_check()
    assert check
    return check


def _table() -> Any:
    settings = runtime_mod.load_settings(agent_mod)
    assert settings["operation_callers"]
    import yaml

    root = Path(agent_mod.__file__).resolve().parents[3]
    return yaml.safe_load((root / "config/deploy_lane_routing.yaml").read_text())


def _peer() -> str:
    """An available instance of the committed table other than the agent's own."""
    table = _table()
    return next(
        name
        for name, spec in table["instances"].items()
        if name != "dev-201" and spec.get("available", True)
    )


def _with_urls(monkeypatch: pytest.MonkeyPatch, urls: dict[str, str]) -> None:
    settings = dict(runtime_mod._idle_settings())
    settings["instance_health_urls"] = urls
    monkeypatch.setattr(runtime_mod, "_pin_settings", [settings])


def test_routing_key_check_reads_an_undeclared_peer_as_unread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        result = runtime_mod.check_routing(force=True)
    assert result is not None
    assert result.outcome == "mismatch"
    reasons = {m.instance: m.reason for m in result.mismatches}
    assert reasons[_peer()] == "key-check-unread"
    assert "dev-201" not in reasons
    assert any(
        "routing key check outcome=mismatch" in r.message for r in caplog.records
    )


def test_routing_key_check_passes_when_every_peer_answers_with_the_agents_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    table = _table()
    _with_urls(
        monkeypatch, {name: f"http://{name}/health" for name in table["instances"]}
    )
    monkeypatch.setattr(
        runtime_mod, "_read_key_check", lambda url, timeout: _own_check()
    )
    with caplog.at_level(logging.INFO):
        result = runtime_mod.check_routing(force=True)
    assert result is not None
    assert result.outcome == "ok"
    assert any("routing key check outcome=ok" in r.message for r in caplog.records)


def test_routing_key_check_names_a_peer_whose_key_differs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    table = _table()
    _with_urls(
        monkeypatch, {name: f"http://{name}/health" for name in table["instances"]}
    )
    monkeypatch.setattr(
        runtime_mod, "_read_key_check", lambda url, timeout: "9b0c12edb01c7b61"
    )
    with caplog.at_level(logging.INFO):
        result = runtime_mod.check_routing(force=True)
    assert result is not None
    refused = {m.instance: m for m in result.mismatches}
    assert refused[_peer()].reason == "key-check-differs"
    assert refused[_peer()].instance_key_check == "9b0c12edb01c7b61"
    assert refused[_peer()].runtime_key_check == _own_check()
    assert any(
        r.levelno == logging.WARNING and "9b0c12edb01c7b61" in r.message
        for r in caplog.records
    )


def test_routing_key_check_is_throttled_and_never_runs_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime_mod.check_routing(force=True) is not None
    assert runtime_mod.check_routing() is None
    monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET")
    assert runtime_mod.check_routing(force=True) is None


def _routed(instance: str, minutes_ago: int, command: str) -> str:
    return json.dumps(
        {
            "correlation_id": command,
            "instance": instance,
            "runtime_lane": "dev",
            "requested_by": "gha/omnimarket/main",
            "routed_at": (
                datetime.now(UTC) - timedelta(minutes=minutes_ago)
            ).isoformat(),
        }
    )


def _record(state: Path, command: str) -> None:
    job = JobState(
        correlation_id=command,
        command={"runtime_lane": "dev", "scope": "runtime"},
        status="success",
        phase_results=dict.fromkeys(DEPLOY_PHASE_ORDER, PhaseStatus.SUCCESS),
        completed_at=datetime.now(UTC),
    )
    (state / f"{command}.json").write_text(job.model_dump_json())


def test_routed_command_alarm_writes_one_row_for_a_command_with_no_job_record(
    tmp_path: Path, agent_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = agent_env
    missing, answered, waiting, elsewhere = (str(uuid4()) for _ in range(4))
    (state / "routed-commands.jsonl").write_text(
        "\n".join(
            [
                _routed("dev-201", 20, missing),
                _routed("dev-201", 20, answered),
                _routed("dev-201", 2, waiting),
                _routed(_peer(), 20, elsewhere),
            ]
        )
        + "\n"
    )
    _record(state, answered)
    with caplog.at_level(logging.INFO):
        result = runtime_mod.check_routed_commands(force=True)
    assert result is not None
    assert [str(a.correlation_id) for a in result.alarms] == [missing]
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert f"alarm={ALARM}" in rows[0]
    assert f"command={missing}" in rows[0]
    assert "instance=dev-201" in rows[0]
    assert any(
        "routed command alarm polled=3 instance=dev-201 alarms=1 waiting=1" in r.message
        for r in caplog.records
    )


def test_routed_command_alarm_with_no_routed_commands_file_polls_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        result = runtime_mod.check_routed_commands(force=True)
    assert result is not None
    assert result.alarms == ()
    assert _rows(tmp_path) == []
    assert any("routed command alarm polled=0" in r.message for r in caplog.records)
    assert runtime_mod.check_routed_commands() is None


def test_the_idle_entry_runs_both_callers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_mod, "check_routing", lambda **_: calls.append("routing")
    )
    monkeypatch.setattr(
        runtime_mod, "check_routed_commands", lambda **_: calls.append("alarm")
    )
    runtime_mod.check_pin(force=True)
    assert calls == ["routing", "alarm"]


def _serve(status: int, body: bytes) -> tuple[Any, str]:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Health(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/health"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, b'{"hmac_key_check": "e30a85fc883bc113"}', "e30a85fc883bc113"),
        (503, b'{"hmac_key_check": "e30a85fc883bc113"}', "e30a85fc883bc113"),
        (200, b'{"hmac_key_check": null}', ""),
        (200, b"not json", ""),
    ],
)
def test_a_peer_health_answer_is_read_for_its_key_check(
    status: int, body: bytes, expected: str
) -> None:
    server, url = _serve(status, body)
    try:
        assert runtime_mod._read_key_check(url, 2.0) == expected
    finally:
        server.shutdown()


def test_an_unreachable_or_non_http_peer_reads_as_unread() -> None:
    assert runtime_mod._read_key_check("http://127.0.0.1:9/health", 1.0) == ""
    assert runtime_mod._read_key_check("file:///etc/hostname", 1.0) == ""


# --- OMN-26316: the overdue dev merge build caller -----------------------------------------


def _git(repo: Path, *args: str, when: datetime | None = None) -> str:
    import os
    import subprocess

    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    if when is not None:
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = when.isoformat()
    done = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return done.stdout.strip()


@pytest.fixture
def dev_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout with a tracked upstream, standing in for the agent's deploy clone."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "-b", "dev")
    clone = tmp_path / "clone"
    clone.mkdir()
    _git(clone, "init", "-b", "dev")
    _git(clone, "remote", "add", "origin", str(origin))
    monkeypatch.setattr(agent_mod, "REPO_DIR", str(clone), raising=False)
    monkeypatch.setattr(runtime_mod, "_overdue_last", [])
    return clone


def _commit(clone: Path, path: str, when: datetime) -> str:
    target = clone / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{when.isoformat()}\n")
    _git(clone, "add", path)
    _git(clone, "commit", "-m", path, when=when)
    return _git(clone, "rev-parse", "HEAD")


def _track(clone: Path) -> None:
    _git(clone, "push", "-u", "origin", "dev")


def _by_sha(result: Any) -> dict[str, Any]:
    return {d.merge_sha: d for d in result}


def test_overdue_dev_merge_is_requested_once_its_window_has_passed_with_no_dev_build(
    dev_clone: Path, caplog: pytest.LogCaptureFixture
) -> None:
    now = datetime.now(UTC)
    _commit(dev_clone, "base.txt", now - timedelta(days=3))
    docs = _commit(dev_clone, "docs/note.md", now - timedelta(hours=3))
    old = _commit(
        dev_clone, "src/omnibase_infra/handlers/old.py", now - timedelta(hours=2)
    )
    fresh = _commit(
        dev_clone, "src/omnibase_infra/handlers/new.py", now - timedelta(minutes=10)
    )
    _track(dev_clone)
    with caplog.at_level(logging.INFO):
        result = runtime_mod.check_overdue_dev_merges(force=True)
    assert result is not None
    seen = _by_sha(result)
    assert (seen[docs].outcome, seen[docs].reason) == ("SATISFIED", "non_runtime")
    assert (seen[old].outcome, seen[old].reason) == ("REQUEST", "overdue_no_build")
    assert seen[old].request_id
    assert (seen[fresh].outcome, seen[fresh].reason) == ("WAIT", "within_window")
    assert any(
        r.levelno == logging.WARNING
        and "overdue dev merge build sha=" + old in r.message
        and str(seen[old].request_id) in r.message
        for r in caplog.records
    )
    assert any(
        "overdue dev merge builds merges=3 satisfied=1 waiting=1 requests=1"
        in r.message
        for r in caplog.records
    )


def test_a_dev_job_accepted_after_the_merge_satisfies_it_and_nothing_is_requested(
    dev_clone: Path, agent_env: Path
) -> None:
    now = datetime.now(UTC)
    merged = now - timedelta(hours=2)
    _commit(dev_clone, "base.txt", now - timedelta(days=3))
    sha = _commit(dev_clone, "src/omnibase_infra/handlers/old.py", merged)
    _track(dev_clone)
    job = {
        "correlation_id": str(uuid4()),
        "command": {"runtime_lane": "dev", "git_ref": "origin/dev", "scope": "runtime"},
        "accepted_at": (merged + timedelta(minutes=5)).isoformat(),
        "status": "success",
        # OMN-26316: a job counts only when the sha it built is the merge or descends from it.
        "lineage": {"resolved_ref": sha, "build_ref": sha},
    }
    (agent_env / f"{job['correlation_id']}.json").write_text(json.dumps(job))
    (agent_env / "agent-adoption.json").write_text('{"alarmed": false}')
    result = runtime_mod.check_overdue_dev_merges(force=True)
    assert result is not None
    assert (_by_sha(result)[sha].outcome, str(_by_sha(result)[sha].job_id)) == (
        "SATISFIED",
        job["correlation_id"],
    )


def test_overdue_check_is_throttled_and_an_unread_checkout_is_skipped(
    dev_clone: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        assert runtime_mod.check_overdue_dev_merges(force=True) is None
    assert any("overdue dev merge builds skipped" in r.message for r in caplog.records)
    _commit(dev_clone, "base.txt", datetime.now(UTC) - timedelta(days=3))
    _commit(dev_clone, "docs/note.md", datetime.now(UTC) - timedelta(hours=3))
    _track(dev_clone)
    assert runtime_mod.check_overdue_dev_merges(force=True) is not None
    assert runtime_mod.check_overdue_dev_merges() is None
