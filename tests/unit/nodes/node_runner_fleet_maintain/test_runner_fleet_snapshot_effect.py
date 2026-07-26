# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerRunnerFleetSnapshot (OMN-13942).

Every gh/ssh call is mocked -- no test in this module makes a real GitHub
API call, SSH connection, or Docker call, and nothing here mutates any real
state. This module exercises the EFFECT node's gather logic in complete
isolation from the live fleet.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_runner_health_snapshot_effect.handlers.handler_runner_fleet_snapshot import (
    HandlerRunnerFleetSnapshot,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
)


def _config() -> ModelRunnerFleetConfig:
    return ModelRunnerFleetConfig(
        version="1.0",
        github_org="OmniNode-ai",
        runner_host="192.168.86.201",
        runner_group="omnibase-ci",
        runner_name_prefix="omninode-runner",
        expected_count=2,
    )


class _FakeProc:
    """Minimal stand-in for asyncio.subprocess.Process."""

    def __init__(
        self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def _make_dispatcher(
    *,
    github_runners: list[dict[str, object]],
    docker_lines: list[str],
    buildx_ok: bool = True,
) -> AsyncMock:
    """Build a side_effect callable that inspects argv and returns a canned _FakeProc.

    Queue-facts and codeload-scan calls (one per watched repo) return empty
    results so the test stays focused on the GitHub + Docker reconciliation
    path -- those probes get their own dedicated tests below.
    """

    async def _dispatch(*args: object, **_kwargs: object) -> _FakeProc:
        argv = [str(a) for a in args]
        if argv[:2] == ["gh", "api"] and "/actions/runners" in argv[2]:
            lines = "\n".join(json.dumps(r) for r in github_runners)
            return _FakeProc(stdout=lines.encode())
        if argv[:2] == ["gh", "api"] and "/actions/runs" in argv[2]:
            return _FakeProc(stdout=b"")
        if argv[:2] == ["gh", "run"] and argv[2] == "list":
            return _FakeProc(stdout=b"[]")
        if argv[0] == "ssh" and "buildx" in argv[2]:
            return _FakeProc(stdout=b"OK\n" if buildx_ok else b"FAIL\n")
        if argv[0] == "ssh":
            return _FakeProc(stdout=("\n".join(docker_lines)).encode())
        raise AssertionError(f"Unexpected subprocess call: {argv}")

    return AsyncMock(side_effect=_dispatch)


@pytest.mark.unit
class TestHandlerRunnerFleetSnapshotGithubDockerReconciliation:
    @pytest.mark.asyncio
    async def test_forward_pass_matches_github_to_docker(self):
        cid = uuid4()
        github_runners = [
            {"name": "omninode-runner-1", "status": "online", "busy": True},
            {"name": "omninode-runner-2", "status": "offline", "busy": False},
        ]
        docker_lines = [
            "omninode-runner-1\trunning\t0\tUp 2 days\t30",
            "omninode-runner-2\trunning\t2\tUp 1 day\t-1",
        ]
        dispatcher = _make_dispatcher(
            github_runners=github_runners, docker_lines=docker_lines
        )
        handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch("asyncio.create_subprocess_exec", dispatcher):
            snapshot = await handler.handle(correlation_id=cid)

        assert snapshot.correlation_id == cid
        assert len(snapshot.runners) == 2
        by_name = {r.name: r for r in snapshot.runners}
        assert by_name["omninode-runner-1"].github_status == "online"
        assert by_name["omninode-runner-1"].github_busy is True
        assert by_name["omninode-runner-1"].docker_restart_count == 0
        assert by_name["omninode-runner-1"].diag_heartbeat_age_seconds == 30.0
        assert by_name["omninode-runner-2"].github_status == "offline"
        assert by_name["omninode-runner-2"].docker_restart_count == 2
        assert by_name["omninode-runner-2"].diag_heartbeat_age_seconds is None
        assert snapshot.github_source_ok is True
        assert snapshot.docker_source_ok is True
        assert snapshot.buildx_available is True

    @pytest.mark.asyncio
    async def test_reverse_pass_flags_stale_registration(self):
        """A Docker container with no GitHub registration is flagged, not dropped."""
        cid = uuid4()
        dispatcher = _make_dispatcher(
            github_runners=[],
            docker_lines=["omninode-runner-1\trunning\t0\tUp 2 days\t10"],
        )
        handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch("asyncio.create_subprocess_exec", dispatcher):
            snapshot = await handler.handle(correlation_id=cid)

        assert len(snapshot.runners) == 1
        assert snapshot.runners[0].stale_registration is True
        assert snapshot.runners[0].github_status == "not_registered"

    @pytest.mark.asyncio
    async def test_runner_index_beyond_expected_count_is_excluded(self):
        cid = uuid4()
        dispatcher = _make_dispatcher(
            github_runners=[
                {"name": "omninode-runner-99", "status": "online", "busy": False}
            ],
            docker_lines=[],
        )
        handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch("asyncio.create_subprocess_exec", dispatcher):
            snapshot = await handler.handle(correlation_id=cid)
        assert snapshot.runners == ()

    @pytest.mark.asyncio
    async def test_buildx_unavailable_probe_surfaces_on_snapshot(self):
        cid = uuid4()
        dispatcher = _make_dispatcher(
            github_runners=[], docker_lines=[], buildx_ok=False
        )
        handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch("asyncio.create_subprocess_exec", dispatcher):
            snapshot = await handler.handle(correlation_id=cid)
        assert snapshot.buildx_available is False

    @pytest.mark.asyncio
    async def test_github_api_failure_surfaces_as_source_error_not_exception(self):
        cid = uuid4()

        async def _dispatch(*args: object, **_kwargs: object) -> _FakeProc:
            argv = [str(a) for a in args]
            if argv[:2] == ["gh", "api"] and "/actions/runners" in argv[2]:
                return _FakeProc(stderr=b"HTTP 403", returncode=1)
            if argv[:2] == ["gh", "api"]:
                return _FakeProc(stdout=b"")
            if argv[:2] == ["gh", "run"]:
                return _FakeProc(stdout=b"[]")
            if argv[0] == "ssh" and "buildx" in argv[2]:
                return _FakeProc(stdout=b"OK\n")
            if argv[0] == "ssh":
                return _FakeProc(stdout=b"")
            raise AssertionError(f"Unexpected call: {argv}")

        handler = HandlerRunnerFleetSnapshot(config=_config())
        with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=_dispatch)):
            snapshot = await handler.handle(correlation_id=cid)

        assert snapshot.github_source_ok is False
        assert snapshot.source_errors != ()
        # Partial failure still produces a usable (degraded) snapshot -- no exception.
        assert snapshot.runners == ()
