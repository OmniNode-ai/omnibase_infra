# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readiness-fact gathering in the snapshot EFFECT (OMN-15255).

Every gh/ssh call is mocked. Nothing here connects to a runner host, and the
assertions below include an explicit proof that the probe command sent over
SSH contains no container-mutating verb -- this node must stay read-only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_runner_health_snapshot_effect.handlers.handler_runner_fleet_snapshot import (
    HandlerRunnerFleetSnapshot,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
)

# 8-field probe line: name, status, restart_count, uptime, diag_age, health,
# listeners, orphans.
_HEALTHY_LINE = "omninode-runner-1\trunning\t0\tUp 2 days (healthy)\t120\thealthy\t1\t0"
_UNHEALTHY_LINE = (
    "omninode-runner-2\trunning\t0\tUp 2 days (unhealthy)\t120\tunhealthy\t2\t1"
)
# Pre-OMN-15255 shape: the three readiness fields are simply absent.
_LEGACY_LINE = "omninode-runner-1\trunning\t0\tUp 2 days\t120"


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
    def __init__(
        self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def _dispatcher(
    *,
    docker_lines: list[str],
    disk_stdout: bytes = b"42\n",
    disk_returncode: int = 0,
    seen: list[list[str]] | None = None,
) -> AsyncMock:
    """Route mocked subprocess calls by argv, recording each one."""

    async def _dispatch(*args: object, **_kwargs: object) -> _FakeProc:
        argv = [str(a) for a in args]
        if seen is not None:
            seen.append(argv)
        if argv[:2] == ["gh", "api"]:
            return _FakeProc(stdout=b"")
        if argv[:2] == ["gh", "run"]:
            return _FakeProc(stdout=b"[]")
        if argv[0] == "ssh" and "buildx" in argv[2]:
            return _FakeProc(stdout=b"OK\n")
        if argv[0] == "ssh" and "df -P" in argv[2]:
            return _FakeProc(stdout=disk_stdout, returncode=disk_returncode)
        if argv[0] == "ssh":
            return _FakeProc(stdout=("\n".join(docker_lines)).encode())
        raise AssertionError(f"Unexpected subprocess call: {argv}")

    return AsyncMock(side_effect=_dispatch)


async def _gather(dispatcher: AsyncMock):
    handler = HandlerRunnerFleetSnapshot(config=_config())
    with patch("asyncio.create_subprocess_exec", dispatcher):
        return await handler.handle(correlation_id=uuid4())


@pytest.mark.unit
class TestReadinessFactsAreGathered:
    @pytest.mark.asyncio
    async def test_docker_health_and_listener_topology_land_on_the_fact(self):
        snapshot = await _gather(
            _dispatcher(docker_lines=[_HEALTHY_LINE, _UNHEALTHY_LINE])
        )
        by_name = {r.name: r for r in snapshot.runners}
        assert by_name["omninode-runner-1"].docker_health == "healthy"
        assert by_name["omninode-runner-1"].listener_process_count == 1
        assert by_name["omninode-runner-1"].orphaned_listener_count == 0
        assert by_name["omninode-runner-2"].docker_health == "unhealthy"
        assert by_name["omninode-runner-2"].listener_process_count == 2
        assert by_name["omninode-runner-2"].orphaned_listener_count == 1

    @pytest.mark.asyncio
    async def test_host_disk_percent_lands_on_the_snapshot(self):
        snapshot = await _gather(
            _dispatcher(docker_lines=[_HEALTHY_LINE], disk_stdout=b"91\n")
        )
        assert snapshot.host_disk_used_percent == 91.0

    @pytest.mark.asyncio
    async def test_disk_probe_failure_is_unknown_not_zero(self):
        """A failed probe must not read as an empty disk -- that would PASS."""
        snapshot = await _gather(
            _dispatcher(
                docker_lines=[_HEALTHY_LINE], disk_stdout=b"", disk_returncode=255
            )
        )
        assert snapshot.host_disk_used_percent is None
        assert any("host disk probe" in e for e in snapshot.source_errors)

    @pytest.mark.asyncio
    async def test_unparseable_disk_output_is_unknown(self):
        snapshot = await _gather(
            _dispatcher(docker_lines=[_HEALTHY_LINE], disk_stdout=b"Filesystem\n")
        )
        assert snapshot.host_disk_used_percent is None
        assert snapshot.source_errors != ()

    @pytest.mark.asyncio
    async def test_minus_one_sentinels_read_unknown_not_zero(self):
        """`0 listeners` and `could not look` are different facts.

        Collapsing the sentinel to 0 would turn a failed exec into a confident
        LISTENER_TOPOLOGY failure and bounce a healthy runner.
        """
        line = "omninode-runner-1\trunning\t0\tUp 2 days\t120\tunknown\t-1\t-1"
        snapshot = await _gather(_dispatcher(docker_lines=[line]))
        fact = snapshot.runners[0]
        assert fact.listener_process_count is None
        assert fact.orphaned_listener_count is None
        assert fact.docker_health == ""

    @pytest.mark.asyncio
    async def test_pre_rollout_probe_output_degrades_to_unknown(self):
        """A host still running the old 5-field probe must not fabricate a PASS."""
        snapshot = await _gather(_dispatcher(docker_lines=[_LEGACY_LINE]))
        fact = snapshot.runners[0]
        assert fact.docker_restart_count == 0
        assert fact.diag_heartbeat_age_seconds == 120.0
        assert fact.docker_health == ""
        assert fact.listener_process_count is None
        assert fact.orphaned_listener_count is None


@pytest.mark.unit
class TestSnapshotProbeStaysReadOnly:
    @pytest.mark.asyncio
    async def test_no_mutating_docker_verb_reaches_the_runner_host(self):
        """Asserted against the command actually dispatched, not the source text."""
        seen: list[list[str]] = []
        await _gather(_dispatcher(docker_lines=[_HEALTHY_LINE], seen=seen))
        ssh_commands = [argv[2] for argv in seen if argv[0] == "ssh"]
        assert ssh_commands, "expected at least one ssh probe"
        forbidden = (
            "docker restart",
            "docker stop",
            "docker rm",
            "docker kill",
            "docker start",
            "docker compose",
            "docker tag",
            "docker commit",
            "--force-recreate",
        )
        for command in ssh_commands:
            for verb in forbidden:
                assert verb not in command, (
                    f"mutating verb {verb!r} in probe: {command}"
                )

    @pytest.mark.asyncio
    async def test_probe_requests_health_and_listener_topology(self):
        seen: list[list[str]] = []
        await _gather(_dispatcher(docker_lines=[_HEALTHY_LINE], seen=seen))
        docker_probe = next(
            argv[2] for argv in seen if argv[0] == "ssh" and "docker ps -a" in argv[2]
        )
        assert ".State.Health.Status" in docker_probe
        # `[R]unner.Listener` is the self-excluding grep pattern -- matching the
        # literal would also match the grep process itself and always count 1.
        assert "[R]unner.Listener" in docker_probe
        assert 'grep -c "^ *1 "' in docker_probe
