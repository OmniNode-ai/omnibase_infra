# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: full runner-fleet-maintain workflow, handlers dispatched directly (OMN-13942).

Simulates the orchestrator dispatching handlers in sequence without the full
event bus/runtime -- mirrors
``test_merge_sweep_canary/test_merge_sweep_workflow_integration.py``.

No test in this module restarts, cancels, or mutates anything real: the
EFFECT's gh/ssh calls are mocked, and Increment 1 has no mutating code path
to exercise even if it weren't mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_runner_fleet_health_compute.handlers.handler_runner_fleet_health_evaluate import (
    HandlerRunnerFleetHealthEvaluate,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.handlers.handler_runner_fleet_health_verdict_complete import (
    HandlerRunnerFleetHealthVerdictComplete,
)
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.handlers.handler_runner_fleet_maintain_start import (
    HandlerRunnerFleetMaintainStart,
)
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.handlers.handler_runner_fleet_snapshot_complete import (
    HandlerRunnerFleetSnapshotComplete,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.handlers.handler_runner_fleet_snapshot import (
    HandlerRunnerFleetSnapshot,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
)


class _FakeProc:
    def __init__(
        self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def _config() -> ModelRunnerFleetConfig:
    return ModelRunnerFleetConfig(
        version="1.0",
        github_org="OmniNode-ai",
        runner_host="192.168.86.201",
        runner_group="omnibase-ci",
        runner_name_prefix="omninode-runner",
        expected_count=2,
    )


def _dispatcher(
    github_runners: list[dict[str, object]], docker_lines: list[str]
) -> AsyncMock:
    async def _dispatch(*args: object, **_kwargs: object) -> _FakeProc:
        argv = [str(a) for a in args]
        if argv[:2] == ["gh", "api"] and "/actions/runners" in argv[2]:
            lines = "\n".join(json.dumps(r) for r in github_runners)
            return _FakeProc(stdout=lines.encode())
        if argv[:2] == ["gh", "api"]:
            return _FakeProc(stdout=b"")
        if argv[:2] == ["gh", "run"]:
            return _FakeProc(stdout=b"[]")
        if argv[0] == "ssh" and "buildx" in argv[2]:
            return _FakeProc(stdout=b"OK\n")
        if argv[0] == "ssh":
            return _FakeProc(stdout=("\n".join(docker_lines)).encode())
        raise AssertionError(f"Unexpected subprocess call: {argv}")

    return AsyncMock(side_effect=_dispatch)


@pytest.mark.unit
class TestRunnerFleetMaintainWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_full_workflow_saturated_fleet_produces_completed_event(self):
        """End-to-end: start -> gather -> evaluate -> completed. No mutation anywhere."""
        cid = uuid4()

        # Step 1: Start (ORCHESTRATOR handler) -- no I/O.
        start_handler = HandlerRunnerFleetMaintainStart()
        gather_command = await start_handler.handle(correlation_id=cid)
        assert gather_command.correlation_id == cid

        # Step 2: Gather snapshot (EFFECT) -- all I/O mocked.
        github_runners = [
            {"name": "omninode-runner-1", "status": "online", "busy": True},
            {"name": "omninode-runner-2", "status": "online", "busy": True},
        ]
        docker_lines = [
            "omninode-runner-1\trunning\t0\tUp 2 days\t20",
            "omninode-runner-2\trunning\t0\tUp 2 days\t20",
        ]
        snapshot_handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch(
            "asyncio.create_subprocess_exec",
            _dispatcher(github_runners, docker_lines),
        ):
            snapshot = await snapshot_handler.handle(
                correlation_id=gather_command.correlation_id
            )
        assert len(snapshot.runners) == 2

        # Step 3: Snapshot complete (ORCHESTRATOR handler) -- no I/O.
        snapshot_complete_handler = HandlerRunnerFleetSnapshotComplete()
        evaluate_command = await snapshot_complete_handler.handle(snapshot=snapshot)
        assert evaluate_command.correlation_id == cid
        assert evaluate_command.snapshot is snapshot

        # Step 4: Evaluate health (COMPUTE) -- pure, no I/O.
        evaluate_handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await evaluate_handler.handle(evaluate_command)
        assert verdict.online_count == 2
        assert verdict.saturation_ratio == 1.0
        assert all(
            a.state == EnumRunnerFleetHealthState.SATURATED for a in verdict.assessments
        )

        # Step 5: Verdict complete (ORCHESTRATOR handler) -- terminal event, no I/O.
        verdict_complete_handler = HandlerRunnerFleetHealthVerdictComplete()
        completed = await verdict_complete_handler.handle(verdict=verdict)

        assert completed.correlation_id == cid
        assert completed.verdict is verdict
        # Increment 1 hard invariant: the terminal event carries a report, nothing else.
        assert not hasattr(completed, "mutation")
        assert not hasattr(completed, "action_taken")

    @pytest.mark.asyncio
    async def test_full_workflow_healthy_fleet_recommends_nothing(self):
        cid = uuid4()
        github_runners = [
            {"name": "omninode-runner-1", "status": "online", "busy": False}
        ]
        docker_lines = ["omninode-runner-1\trunning\t0\tUp 2 days\t20"]

        start_handler = HandlerRunnerFleetMaintainStart()
        gather_command = await start_handler.handle(correlation_id=cid)

        snapshot_handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch(
            "asyncio.create_subprocess_exec",
            _dispatcher(github_runners, docker_lines),
        ):
            snapshot = await snapshot_handler.handle(
                correlation_id=gather_command.correlation_id
            )

        evaluate_command = await HandlerRunnerFleetSnapshotComplete().handle(
            snapshot=snapshot
        )
        verdict = await HandlerRunnerFleetHealthEvaluate().handle(evaluate_command)
        completed = await HandlerRunnerFleetHealthVerdictComplete().handle(
            verdict=verdict
        )

        assert completed.verdict.recommended_actions == ()
        assert all(
            a.state == EnumRunnerFleetHealthState.HEALTHY
            for a in completed.verdict.assessments
        )
