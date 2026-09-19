# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deploy-agent verification must probe runtime health, not LLM ports."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import Phase, PhaseStatus
from deploy_agent.executor import DeployExecutor, VerificationFailedError


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _completed(
    cmd: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _health_payload(
    *,
    status: str = "healthy",
    is_running: bool = True,
    config_prefetch_status: str = "ok",
) -> str:
    return json.dumps(
        {
            "status": status,
            "details": {
                "is_running": is_running,
                "config_prefetch_status": config_prefetch_status,
            },
        }
    )


def _is_projection_table_check(cmd: list[str]) -> bool:
    return "omnidash_analytics" in cmd and any(
        f"SELECT to_regclass('public.{table}') IS NOT NULL" in cmd
        for table in ("delegation_events", "node_service_registry")
    )


def test_verify_probes_runtime_health_ports_only() -> None:
    executor = DeployExecutor()
    captured_cmds: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(cmd, stdout="t\n")
        if "http://localhost:8085/health" in cmd:
            return _completed(cmd, stdout=_health_payload())
        if "http://localhost:8086/health" in cmd:
            return _completed(cmd, stdout=_health_payload())
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    endpoints = [check.endpoint for check in checks]
    assert "http://localhost:8085/health" in endpoints
    assert "http://localhost:8086/health" in endpoints
    assert not any(":8000/health" in endpoint for endpoint in endpoints)
    assert not any(":8001/health" in endpoint for endpoint in endpoints)
    assert not any(":8002/health" in endpoint for endpoint in endpoints)
    assert not any(
        "localhost:8000" in " ".join(cmd)
        or "localhost:8001" in " ".join(cmd)
        or "localhost:8002" in " ".join(cmd)
        for cmd in captured_cmds
    )
    runtime_checks = {
        check.service: check.status
        for check in checks
        if check.service in {"omninode-runtime", "runtime-effects"}
    }
    assert runtime_checks == {
        "omninode-runtime": "pass",
        "runtime-effects": "pass",
    }


def test_verify_accepts_runtime_health_when_config_prefetch_skipped() -> None:
    executor = DeployExecutor()

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if cmd[:2] == ["docker", "ps"]:
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(cmd, stdout="t\n")
        if "http://localhost:8085/health" in cmd:
            return _completed(
                cmd, stdout=_health_payload(config_prefetch_status="skipped")
            )
        if "http://localhost:8086/health" in cmd:
            return _completed(
                cmd, stdout=_health_payload(config_prefetch_status="skipped")
            )
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    runtime_checks = {
        check.service: check.status
        for check in checks
        if check.service in {"omninode-runtime", "runtime-effects"}
    }
    assert runtime_checks == {
        "omninode-runtime": "pass",
        "runtime-effects": "pass",
    }


def test_verify_fails_postgres_check_when_node_service_registry_missing() -> None:
    executor = DeployExecutor()

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if cmd[:2] == ["docker", "ps"]:
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(cmd, stdout="f\n")
        if "http://localhost:8085/health" in cmd:
            return _completed(cmd, stdout=_health_payload())
        if "http://localhost:8086/health" in cmd:
            return _completed(cmd, stdout=_health_payload())
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")

    # OMN-18640 AC8. This used to RETURN the check list with a `fail` in it and
    # report the phase SUCCESS, so a lane missing a projection table completed
    # its deploy. The subject of the test is unchanged -- which check failed --
    # and what changed is that the job now fails with it.
    with patch("deploy_agent.executor._run", side_effect=fake_run):
        with pytest.raises(VerificationFailedError) as excinfo:
            executor.verify(on_phase_update=_noop_phase_update)

    status_by_endpoint = {
        check.endpoint: check.status for check in executor.health_checks
    }
    assert (
        status_by_endpoint["omnidash_analytics.node_service_registry exists"] == "fail"
    )
    assert "node_service_registry" in str(excinfo.value)


def test_verify_fails_runtime_health_when_config_prefetch_degraded() -> None:
    executor = DeployExecutor()

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if cmd[:2] == ["docker", "ps"]:
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(cmd, stdout="t\n")
        if "http://localhost:8085/health" in cmd:
            return _completed(cmd, stdout=_health_payload())
        if "http://localhost:8086/health" in cmd:
            return _completed(
                cmd,
                stdout=_health_payload(config_prefetch_status="degraded_error"),
            )
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")

    # OMN-18640 AC8: a degraded config prefetch answers FAST, so before this
    # change it was recorded as a failure and reported as a successful deploy.
    with patch("deploy_agent.executor._run", side_effect=fake_run):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_noop_phase_update)

    status_by_service = {
        check.service: check.status for check in executor.health_checks
    }
    assert status_by_service["omninode-runtime"] == "pass"
    assert status_by_service["runtime-effects"] == "fail"


def test_verify_fails_runtime_health_when_process_not_running() -> None:
    executor = DeployExecutor()

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if cmd[:2] == ["docker", "ps"]:
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(cmd, stdout="t\n")
        if "http://localhost:8085/health" in cmd:
            return _completed(cmd, stdout=_health_payload(is_running=False))
        if "http://localhost:8086/health" in cmd:
            return _completed(cmd, stdout=_health_payload())
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")

    # OMN-18640 AC8: same again for a runtime that reports its process is not
    # running -- an unambiguous statement that the deploy did not converge.
    with patch("deploy_agent.executor._run", side_effect=fake_run):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_noop_phase_update)

    status_by_service = {
        check.service: check.status for check in executor.health_checks
    }
    assert status_by_service["omninode-runtime"] == "fail"
    assert status_by_service["runtime-effects"] == "pass"
