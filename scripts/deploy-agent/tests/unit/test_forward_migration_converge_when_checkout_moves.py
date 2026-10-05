# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An idle dev lane applies migration changes made outside deploy jobs."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import DEV_LANE_ONLY_MIGRATION_SERVICES, EnumRuntimeLane
from deploy_agent.executor import (
    MIGRATION_TREE_PATHS,
    RUNTIME_MIGRATION_SERVICES,
    DeployExecutor,
    read_applied_migration_fingerprint,
    read_checkout_migration_fingerprint,
    write_applied_migration_fingerprint,
)

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _commit(repo: Path) -> None:
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "test migration content")


def test_fingerprint_tracks_only_migration_content(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Migration Test")
    migrations = tmp_path / "docker/migrations/forward"
    migrations.mkdir(parents=True)
    (migrations / "001_a.sql").write_text("SELECT 1;\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run-forward-migrations.sh"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _commit(tmp_path)

    original = read_checkout_migration_fingerprint(str(tmp_path))
    expected = [
        _git(tmp_path, "rev-parse", f"HEAD:{path}") for path in MIGRATION_TREE_PATHS
    ]
    assert original == ":".join(expected)

    node = migrations / "nodes/x"
    node.mkdir(parents=True)
    (node / "0001.sql").write_text("SELECT 2;\n", encoding="utf-8")
    _commit(tmp_path)
    moved = read_checkout_migration_fingerprint(str(tmp_path))
    assert moved is not None and moved != original

    (tmp_path / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    _commit(tmp_path)
    assert read_checkout_migration_fingerprint(str(tmp_path)) == moved

    runner.write_text("#!/bin/sh\necho migrated\n", encoding="utf-8")
    _commit(tmp_path)
    assert read_checkout_migration_fingerprint(str(tmp_path)) != moved


def test_fingerprint_outside_a_git_repo_is_unknown(tmp_path: Path) -> None:
    assert read_checkout_migration_fingerprint(str(tmp_path)) is None


@pytest.mark.parametrize("stdout", ["", "a\n", "a\nb\nc\n"])
def test_fingerprint_rejects_wrong_line_count(
    stdout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        executor_mod,
        "_run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout, "")),
    )
    assert read_checkout_migration_fingerprint() is None


def test_applied_fingerprint_round_trip(tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert read_applied_migration_fingerprint(state, EnumRuntimeLane.DEV) is None
    write_applied_migration_fingerprint(state, EnumRuntimeLane.DEV, "tree:runner")
    assert (
        read_applied_migration_fingerprint(state, EnumRuntimeLane.DEV) == "tree:runner"
    )
    record = state / "forward-migration-applied.dev.json"
    document = json.loads(record.read_text(encoding="utf-8"))
    assert set(document) == {"fingerprint", "applied_at"}
    assert datetime.fromisoformat(document["applied_at"]).tzinfo == UTC
    assert list(state.iterdir()) == [record]


@pytest.mark.parametrize(
    "content", ["not json", "[]", "{}", '{"fingerprint": 3}', '{"fingerprint": ""}']
)
def test_corrupt_applied_fingerprint_is_unknown(tmp_path: Path, content: str) -> None:
    (tmp_path / "forward-migration-applied.dev.json").write_text(
        content, encoding="utf-8"
    )
    assert read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV) is None


@pytest.fixture
def migration_seams(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Run the real preflight while stubbing every Docker and environment read."""
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "t\n" if "psql" in cmd else "", "")

    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(executor_mod, "_compose_env", dict)
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", lambda *args, **kwargs: (True, [])
    )
    monkeypatch.setattr(
        executor_mod, "verify_oneshots_completed", lambda *args, **kwargs: (True, [])
    )
    monkeypatch.setattr(
        executor_mod, "read_checkout_migration_fingerprint", lambda: "old"
    )
    return calls


def test_preflight_records_fingerprint_from_before_one_shots(
    tmp_path: Path, migration_seams: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = DeployExecutor(migration_record_dir=tmp_path)
    run: Callable[..., subprocess.CompletedProcess[str]] = executor_mod._run

    def move_checkout(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "up" in cmd:
            assert (
                read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV)
                is None
            )
            monkeypatch.setattr(
                executor_mod, "read_checkout_migration_fingerprint", lambda: "new"
            )
        return run(cmd, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", move_checkout)
    assert executor._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV) == "old"
    assert migration_seams
    assert executor_mod.read_checkout_migration_fingerprint() == "new"
    assert read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV) == "old"


@pytest.mark.parametrize("failure", ["compose", "container", "oneshot", "table"])
def test_preflight_failure_preserves_last_successful_record(
    tmp_path: Path,
    migration_seams: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    write_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV, "previous")
    if failure == "compose":
        monkeypatch.setattr(
            executor_mod,
            "_run",
            Mock(return_value=subprocess.CompletedProcess([], 1, "", "apply failed")),
        )
    elif failure == "container":
        monkeypatch.setattr(
            executor_mod,
            "verify_containers_up",
            lambda *args, **kwargs: (False, ["forward-migration"]),
        )
    elif failure == "oneshot":
        monkeypatch.setattr(
            executor_mod,
            "verify_oneshots_completed",
            lambda *args, **kwargs: (False, ["cloud-migration"]),
        )
    else:
        monkeypatch.setattr(
            executor_mod,
            "_run",
            Mock(return_value=subprocess.CompletedProcess([], 0, "f\n", "")),
        )
    with pytest.raises(RuntimeError, match="migration preflight"):
        DeployExecutor(migration_record_dir=tmp_path)._ensure_runtime_migrations_ready(
            lane=EnumRuntimeLane.DEV
        )
    assert (
        read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV) == "previous"
    )


def test_preflight_does_not_record_an_unknown_fingerprint(
    tmp_path: Path, migration_seams: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        executor_mod, "read_checkout_migration_fingerprint", lambda: None
    )
    assert (
        DeployExecutor(migration_record_dir=tmp_path)._ensure_runtime_migrations_ready()
        == ""
    )
    assert read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV) is None


def test_converge_forward_migration_restarts_no_runtime_service(
    tmp_path: Path, migration_seams: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_calls: list[tuple[str, str, str, float]] = []
    locked = False

    @contextmanager
    def lock(project: str, *, lane: str, ref: str, timeout: float) -> Iterator[None]:
        nonlocal locked
        lock_calls.append((project, lane, ref, timeout))
        locked = True
        try:
            yield
        finally:
            locked = False

    def fingerprint() -> str:
        return "inside-lock" if locked else "before-lock"

    monkeypatch.setattr(executor_mod, "lane_lock", lock)
    monkeypatch.setattr(
        executor_mod, "read_checkout_migration_fingerprint", fingerprint
    )
    assert (
        DeployExecutor(migration_record_dir=tmp_path).converge_forward_migration(
            lane=EnumRuntimeLane.DEV
        )
        == "inside-lock"
    )
    assert lock_calls == [
        (
            executor_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project,
            "dev",
            "before-lock",
            executor_mod.DEFAULT_LANE_LOCK_TIMEOUT_SECONDS,
        )
    ]
    assert (
        read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV)
        == "inside-lock"
    )
    compose_calls = [cmd for cmd in migration_seams if cmd[:2] == ["docker", "compose"]]
    expected_services = (*RUNTIME_MIGRATION_SERVICES, *DEV_LANE_ONLY_MIGRATION_SERVICES)
    assert [cmd[-1] for cmd in compose_calls] == list(expected_services)
    for cmd in compose_calls:
        assert cmd[cmd.index("up") :] == [
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            cmd[-1],
        ]
        assert cmd[-1] in expected_services
        assert "projection-delegation-writer" not in cmd
        assert "omninode-runtime" not in cmd
    assert all("psql" in cmd for cmd in migration_seams if cmd not in compose_calls)


@dataclass
class _IdleHarness:
    """The agent under test plus the mocks its idle tick reads and calls."""

    agent: DeployAgent
    converge: Mock
    has_active_job: Mock
    read_checkout: Mock


@pytest.fixture
def idle_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_IdleHarness]:
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: 100.0)
    read_checkout = Mock(return_value="new")
    monkeypatch.setattr(agent_mod, "read_checkout_migration_fingerprint", read_checkout)
    monkeypatch.setattr(
        agent_mod, "read_applied_migration_fingerprint", Mock(return_value="old")
    )
    agent = DeployAgent(skip_self_update=True)
    has_active_job = Mock(return_value=False)
    converge = Mock(return_value="new")
    monkeypatch.setattr(agent.job_store, "has_active_job", has_active_job)
    monkeypatch.setattr(agent.executor, "converge_forward_migration", converge)
    yield _IdleHarness(agent, converge, has_active_job, read_checkout)
    agent._job_pool.shutdown(wait=True)
    agent._lag_pool.shutdown(wait=True)
    agent._settle_pool.shutdown(wait=True)


def _next_tick(agent: DeployAgent) -> None:
    agent._migration_converge_last_check = None
    agent._maybe_converge_forward_migration()


def test_idle_tick_applies_checkout_move_once_even_if_record_still_old(
    idle_agent: _IdleHarness, caplog: pytest.LogCaptureFixture
) -> None:
    """RED/GREEN incident: an external reset advances DDL with no rebuild job."""
    with caplog.at_level(logging.INFO):
        idle_agent.agent._maybe_converge_forward_migration()
        _next_tick(idle_agent.agent)
    idle_agent.converge.assert_called_once_with(lane=EnumRuntimeLane.DEV)
    assert idle_agent.agent._migration_converge_attempted == {"new"}
    assert "tree new differs from the last applied old" in caplog.text
    assert "no runtime service is restarted" in caplog.text
    assert "succeeded for new" in caplog.text


def test_idle_tick_with_applied_tree_never_converges(
    idle_agent: _IdleHarness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        agent_mod, "read_applied_migration_fingerprint", lambda *args: "new"
    )
    with caplog.at_level(logging.INFO):
        idle_agent.agent._maybe_converge_forward_migration()
        _next_tick(idle_agent.agent)
    idle_agent.converge.assert_not_called()
    assert caplog.text.count("checkout migration tree already applied") == 1


def test_active_job_prevents_idle_migration(idle_agent: _IdleHarness) -> None:
    idle_agent.has_active_job.return_value = True
    idle_agent.agent._maybe_converge_forward_migration()
    idle_agent.converge.assert_not_called()
    idle_agent.read_checkout.assert_not_called()


def test_unreadable_fingerprint_warns_once_per_verdict_change(
    idle_agent: _IdleHarness, caplog: pytest.LogCaptureFixture
) -> None:
    idle_agent.read_checkout.return_value = None
    idle_agent.agent._maybe_converge_forward_migration()
    _next_tick(idle_agent.agent)
    idle_agent.converge.assert_not_called()
    assert caplog.text.count("migration tree fingerprint unreadable") == 1
    assert caplog.records[0].levelno == logging.WARNING
    idle_agent.read_checkout.return_value = "new"
    _next_tick(idle_agent.agent)
    idle_agent.read_checkout.return_value = None
    _next_tick(idle_agent.agent)
    assert caplog.text.count("migration tree fingerprint unreadable") == 2


def test_failed_migration_logs_error_and_is_not_retried(
    idle_agent: _IdleHarness, caplog: pytest.LogCaptureFixture
) -> None:
    idle_agent.converge.side_effect = RuntimeError("DDL refused")
    idle_agent.agent._maybe_converge_forward_migration()
    _next_tick(idle_agent.agent)
    idle_agent.converge.assert_called_once_with(lane=EnumRuntimeLane.DEV)
    assert idle_agent.agent._migration_converge_attempted == {"new"}
    assert "forward migration converge FAILED for new: DDL refused" in caplog.text
    assert "Writers built against these migrations will fail readiness" in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_closed_load_gate_defers_without_consuming_attempt(
    idle_agent: _IdleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = Mock(return_value=False)
    monkeypatch.setattr(idle_agent.agent, "_idle_converge_gate_open", gate)
    idle_agent.agent._maybe_converge_forward_migration()
    idle_agent.converge.assert_not_called()
    assert idle_agent.agent._migration_converge_attempted == set()
    gate.return_value = True
    _next_tick(idle_agent.agent)
    idle_agent.converge.assert_called_once_with(lane=EnumRuntimeLane.DEV)


def test_idle_migration_checks_are_throttled(
    idle_agent: _IdleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    idle_agent.agent._maybe_converge_forward_migration()
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: 159.0)
    idle_agent.agent._maybe_converge_forward_migration()
    idle_agent.read_checkout.assert_called_once_with()
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: 160.0)
    idle_agent.agent._maybe_converge_forward_migration()
    assert idle_agent.read_checkout.call_count == 2


@pytest.mark.parametrize(
    "lanes",
    [{EnumRuntimeLane.PROD}, {EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST}],
)
def test_only_dev_only_agents_converge_migrations(
    idle_agent: _IdleHarness, lanes: set[EnumRuntimeLane]
) -> None:
    idle_agent.agent._allowed_lanes = lanes
    idle_agent.agent._maybe_converge_forward_migration()
    idle_agent.converge.assert_not_called()
    idle_agent.read_checkout.assert_not_called()
