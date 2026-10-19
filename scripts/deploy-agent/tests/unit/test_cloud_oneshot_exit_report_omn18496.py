# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18496: a cloud one-shot that exits non-zero is reported at once, with its error.

Real ``docker compose up -d`` of a one-shot returns 0 before the one-shot runs,
so a failing ``cloud-migration`` is seen only by the exited-0 poll. Measured on
a real daemon (``test_cloud_migration_nonfatal_real_docker_omn18496.py``), the
poll waited out its whole timeout on a container that had already exited 1,
and the job's failure said only that the one-shot "did not complete", without
the error it printed. These cases pin the poll to stop at a non-zero exit and
the failure to carry the exit code and the one-shot's own last lines.
"""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

LOG_LINE = (
    "cloud-migration-1  | ERROR: permission denied to create database "
    "(dsn postgresql://svc_cloud:hunter2@postgres:5432/omninode_cloud)"
)


def _no_sleep(seconds: float) -> None:
    raise AssertionError(f"the poll slept {seconds}s on a one-shot that had exited 1")


def test_poll_returns_at_once_on_a_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        executor_mod,
        "_compose_service_states",
        lambda lane=None: {"cloud-migration": ("exited", 1)},
    )
    monkeypatch.setattr(executor_mod.time, "sleep", _no_sleep)
    failed: dict[str, int] = {}

    ok, pending = executor_mod.verify_oneshots_completed(
        ["cloud-migration"], timeout_s=300, failed=failed
    )

    assert (ok, pending) == (False, ["cloud-migration"])
    assert failed == {"cloud-migration": 1}


def test_poll_still_waits_for_a_running_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the early return must not fire on a live one-shot."""
    reads = iter(
        [
            {"cloud-migration": ("running", None)},
            {"cloud-migration": ("exited", 0)},
        ]
    )
    monkeypatch.setattr(
        executor_mod, "_compose_service_states", lambda lane=None: next(reads)
    )
    sleep = Mock()
    monkeypatch.setattr(executor_mod.time, "sleep", sleep)
    failed: dict[str, int] = {}

    ok, pending = executor_mod.verify_oneshots_completed(
        ["cloud-migration"], timeout_s=300, failed=failed
    )

    assert (ok, pending) == (True, [])
    assert failed == {}
    sleep.assert_called_once()


def test_detached_up_then_exit_one_reports_code_and_redacted_log_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if "logs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, LOG_LINE + "\n", "")
        return subprocess.CompletedProcess(cmd, 0, "t\n" if "psql" in cmd else "", "")

    states = {
        "cloud-migration-files": ("exited", 0),
        "cloud-migration": ("exited", 1),
    }
    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *args: {})
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", Mock(return_value=(True, []))
    )
    monkeypatch.setattr(
        executor_mod, "_compose_service_states", lambda lane=None: states
    )
    monkeypatch.setattr(executor_mod.time, "sleep", _no_sleep)
    monkeypatch.setattr(
        executor_mod, "read_checkout_migration_fingerprint", lambda: "tree:runner"
    )

    executor = DeployExecutor()
    executor._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)

    failure = executor.cloud_migration_failure
    assert failure.startswith("Dev-lane migration preflight failed for cloud-migration")
    assert "exited 1" in failure
    assert "permission denied to create database" in failure
    assert "hunter2" not in failure
    assert "postgresql://***@postgres:5432/omninode_cloud" in failure
    logs = [cmd for cmd in calls if "logs" in cmd]
    assert len(logs) == 1
    assert logs[0][-1] == "cloud-migration"
    assert next(cmd for cmd in calls if "stop" in cmd)[-1] == "onex-api"
