# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner health collector -- GitHub API + SSH Docker inspection."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)
from omnibase_infra.observability.runner_health.model_runner_status import (
    ModelRunnerStatus,
)


class CollectorRunnerHealth:
    """Collects runner health from GitHub API + SSH Docker inspection.

    Cross-references GitHub Actions runner registrations with Docker
    container status to compute ``EnumRunnerHealthState`` per runner.
    Manages partial-source failure gracefully -- if one source fails,
    the snapshot surfaces the degradation explicitly.
    """

    def __init__(
        self,
        github_org: str,
        runner_host: str,
        runner_count: int,
        runner_prefix: str = "omninode-runner",
    ) -> None:
        self._github_org = github_org
        self._runner_host = runner_host
        self._runner_count = runner_count
        self._runner_prefix = runner_prefix

    def _classify_runner(
        self,
        github_status: str,
        github_busy: bool,
        docker_status: str,
        docker_uptime: str,
    ) -> EnumRunnerHealthState:
        """Compute health state from GitHub and Docker status signals."""
        if docker_status == "not_found":
            return EnumRunnerHealthState.MISSING
        if (
            "restarting" in docker_status.lower()
            or "restarting" in docker_uptime.lower()
        ):
            return EnumRunnerHealthState.CRASH_LOOPING
        if docker_status not in ("healthy", "running"):
            return EnumRunnerHealthState.DOCKER_UNHEALTHY
        if github_status == "offline":
            return EnumRunnerHealthState.GITHUB_OFFLINE
        return EnumRunnerHealthState.HEALTHY

    async def _fetch_github_runners(self) -> tuple[list[dict[str, object]], str | None]:
        """Fetch runner list from GitHub API. Returns (runners, error)."""
        proc = await asyncio.create_subprocess_exec(
            "gh",
            "api",
            f"/orgs/{self._github_org}/actions/runners",
            "--jq",
            ".runners[] | {name, status, busy}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return [], (
                f"GitHub API exit code {proc.returncode}: "
                f"{stderr.decode().strip()[:200]}"
            )
        runners: list[dict[str, object]] = []
        for line in stdout.decode().strip().splitlines():
            if line.strip():
                runners.append(json.loads(line))
        return runners, None

    async def _fetch_docker_status(
        self,
    ) -> tuple[dict[str, dict[str, str]], str | None]:
        """Fetch Docker container status via SSH. Returns (statuses, error)."""
        cmd = (
            f"docker ps -a --filter 'name={self._runner_prefix}' "
            f"--format '{{{{.Names}}}}\\t{{{{.Status}}}}'"
        )
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._runner_host,
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {}, (
                f"SSH/Docker exit code {proc.returncode}: "
                f"{stderr.decode().strip()[:200]}"
            )
        result: dict[str, dict[str, str]] = {}
        for line in stdout.decode().strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                name, uptime = parts
                status = (
                    "healthy"
                    if "(healthy)" in uptime
                    else ("restarting" if "Restarting" in uptime else "running")
                )
                result[name] = {"status": status, "uptime": uptime}
        return result, None

    async def _fetch_host_disk(self) -> float:
        """Fetch host disk usage percentage via SSH."""
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._runner_host,
            "df --output=pcent / | tail -1 | tr -d ' %'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        try:
            return float(stdout.decode().strip())
        except ValueError:
            return 0.0

    async def collect(self, correlation_id: UUID) -> ModelRunnerHealthSnapshot:
        """Collect a point-in-time health snapshot of all runners.

        Performs bidirectional reconciliation:
        1. Forward pass: GitHub runners -> look up Docker status
        2. Reverse pass: Docker containers not in GitHub -> STALE_REGISTRATION
        """
        gh_result, docker_result, disk_pct = await asyncio.gather(
            self._fetch_github_runners(),
            self._fetch_docker_status(),
            self._fetch_host_disk(),
        )
        github_runners, gh_error = gh_result
        docker_statuses, docker_error = docker_result

        source_errors: list[str] = []
        if gh_error:
            source_errors.append(gh_error)
        if docker_error:
            source_errors.append(docker_error)

        statuses: list[ModelRunnerStatus] = []

        # Forward pass: GitHub runners -> look up Docker status
        seen_docker_names: set[str] = set()
        for gh in github_runners:
            name = str(gh["name"])
            seen_docker_names.add(name)
            docker = docker_statuses.get(name, {"status": "not_found", "uptime": ""})
            state = self._classify_runner(
                github_status=str(gh["status"]),
                github_busy=bool(gh["busy"]),
                docker_status=docker["status"],
                docker_uptime=docker["uptime"],
            )
            statuses.append(
                ModelRunnerStatus(
                    name=name,
                    github_status=str(gh["status"]),
                    github_busy=bool(gh["busy"]),
                    docker_status=docker["status"],
                    docker_uptime=docker["uptime"],
                    state=state,
                )
            )

        # Reverse pass: Docker containers not in GitHub -> orphaned
        for docker_name, docker_info in docker_statuses.items():
            if docker_name not in seen_docker_names:
                statuses.append(
                    ModelRunnerStatus(
                        name=docker_name,
                        github_status="not_registered",
                        github_busy=False,
                        docker_status=docker_info["status"],
                        docker_uptime=docker_info["uptime"],
                        state=EnumRunnerHealthState.STALE_REGISTRATION,
                        error="Docker container exists but not registered in GitHub",
                    )
                )

        healthy = sum(1 for s in statuses if s.state == EnumRunnerHealthState.HEALTHY)

        return ModelRunnerHealthSnapshot(
            correlation_id=correlation_id,
            collected_at=datetime.now(tz=UTC),
            runners=tuple(statuses),
            expected_runners=self._runner_count,
            observed_runners=len(statuses),
            healthy_count=healthy,
            degraded_count=len(statuses) - healthy,
            github_source_ok=gh_error is None,
            docker_source_ok=docker_error is None,
            source_errors=tuple(source_errors),
            host=self._runner_host,
            host_disk_percent=disk_pct,
        )


__all__ = ["CollectorRunnerHealth"]
