# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A preflight script that could not RUN is not a preflight that FOUND something (OMN-18123).

Every dev-lane rebuild between 2026-09-10T00:49:01Z and 03:31:59Z failed with::

    REQUIRED_COMPOSE_ENV_MISSING for lane dev -- compose validation was not
    attempted. <python>: can't open file
    '.../scripts/preflight_required_compose_env.py': [Errno 2] No such file or
    directory

Nothing about that failure was a compose env problem. The deploy-source clone
had been reset to a commit predating the preflight script, so the interpreter
never opened the file. The error class named the one thing that was NOT wrong,
and the real cause was legible only in the trailing interpreter message that a
reader had to know to look past the class name for.

The method's refusal posture is correct and is not what changes here: there is
still no soft-fail branch and no "continue if the script is missing". A
preflight that can be skipped is the failure mode OMN-17530 closed. What changes
is only which error a caller is told about.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus
from deploy_agent.executor import (
    REPO_DIR,
    DeployExecutor,
    PreflightScriptUnavailableError,
)
from deploy_agent.job_state import JobState
from deploy_agent.publisher import build_completion_payload

_UNSET_VARS_MESSAGE = (
    "required compose variables are unset for lane dev: "
    "POSTGRES_PASSWORD, VALKEY_PASSWORD, KAFKA_SASL_PASSWORD"
)


@pytest.fixture
def absent_script(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> str:
    """Repoint the preflight at a path that does not exist.

    The unit conftest repoints it at THIS checkout, where the script really is
    on disk, so every other test sees a runnable preflight. This fixture is the
    other direction: the deploy-source clone sitting on a commit that predates
    the script, which is the state the five incident jobs found.
    """
    missing = f"{tmp_path}/scripts/preflight_required_compose_env.py"
    monkeypatch.setattr(
        executor_mod, "preflight_required_compose_env_script", lambda: missing
    )
    return missing


def _preflight(executor: DeployExecutor) -> None:
    executor._preflight_required_compose_env(
        lane=EnumRuntimeLane.DEV,
        compose_files=["docker/docker-compose.generated.yml"],
        timeout=60,
    )


class TestMissingScript:
    """AC1, AC2: the script could not be run, and that is its own error."""

    def test_absent_script_raises_its_own_error_class(self, absent_script: str) -> None:
        """The clone is on a commit that predates the script (AC1)."""
        executor = DeployExecutor()

        with pytest.raises(PreflightScriptUnavailableError) as excinfo:
            _preflight(executor)

        message = str(excinfo.value)
        assert absent_script in message
        assert REPO_DIR in message
        # AC1: it must NOT claim the compose environment is the problem.
        assert "REQUIRED_COMPOSE_ENV_MISSING" not in message

    def test_the_script_is_never_invoked_when_it_is_absent(
        self, absent_script: str
    ) -> None:
        """AC2: the distinction is drawn before the subprocess, not from stderr."""
        executor = DeployExecutor()
        issued: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            issued.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=2,
                stdout="",
                stderr="can't open file: [Errno 2] No such file or directory",
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with pytest.raises(PreflightScriptUnavailableError):
                _preflight(executor)

        assert issued == [], (
            "the preflight subprocess ran despite the script being absent; the "
            f"error class would then depend on stderr text: {issued}"
        )


class TestUnsetVariablesUnchanged:
    """AC3, AC4: the OMN-17530 behaviour is preserved exactly."""

    def test_unset_variables_still_raise_the_compose_env_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=_UNSET_VARS_MESSAGE
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with pytest.raises(RuntimeError) as excinfo:
                _preflight(executor)

        message = str(excinfo.value)
        assert "REQUIRED_COMPOSE_ENV_MISSING" in message
        # The complete list still rides through, not just the first variable.
        assert "POSTGRES_PASSWORD" in message
        assert "VALKEY_PASSWORD" in message
        assert "KAFKA_SASL_PASSWORD" in message

    def test_an_unset_variable_result_is_not_the_missing_script_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: the two failures are distinguishable, which is the whole point."""
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=_UNSET_VARS_MESSAGE
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with pytest.raises(RuntimeError) as excinfo:
                _preflight(executor)

        assert not isinstance(excinfo.value, PreflightScriptUnavailableError)

    def test_a_successful_preflight_raises_nothing(self) -> None:
        """AC4: neither branch is a soft-fail, but success is still success."""
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="all 31 required vars present", stderr=""
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            _preflight(executor)


class TestTerminalEvent:
    """AC6: the new class reaches the wire, not just the log."""

    def test_the_error_is_carried_in_the_completion_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
        failure = PreflightScriptUnavailableError(
            "PREFLIGHT_SCRIPT_UNAVAILABLE: "
            f"{REPO_DIR}/scripts/preflight_required_compose_env.py is not a file"
        )
        job = JobState(
            correlation_id=uuid4(),
            command={
                "correlation_id": "00000000-0000-0000-0000-000000000000",
                "requested_by": "node_redeploy_orchestrator",
                "scope": "full",
                "runtime_lane": "dev",
                "git_ref": "origin/dev",
            },
            accepted_at=datetime.now(UTC),
            current_phase=Phase.COMPOSE_GEN,
            phase_results={Phase.PREFLIGHT: PhaseStatus.SUCCESS},
            status="failed",
            errors=[str(failure)],
            completed_at=datetime.now(UTC),
        )

        payload = build_completion_payload(job, "")

        assert payload["errors"] == [str(failure)]
        assert "PREFLIGHT_SCRIPT_UNAVAILABLE" in payload["errors"][0]
        assert "REQUIRED_COMPOSE_ENV_MISSING" not in payload["errors"][0]
