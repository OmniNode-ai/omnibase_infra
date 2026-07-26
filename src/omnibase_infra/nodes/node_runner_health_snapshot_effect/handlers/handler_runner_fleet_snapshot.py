# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that gathers a read-only runner-fleet snapshot (OMN-13942).

This is an EFFECT handler -- it performs ALL external I/O for the
runner-fleet-maintain workflow (GitHub API, SSH/Docker inspection). It is
strictly read-only: no path in this module mutates the fleet, cancels a run,
or touches the merge queue. Recovery/mutation is Increment 2, design-only,
not implemented here.

Ports and extends ``collector_runner_health.py``'s GitHub + Docker
reconciliation with:
  - Docker RestartCount (crash-loop signal, OMN-13109/OMN-13912)
  - Runner.Listener ``_diag`` heartbeat freshness (OMN-13915)
  - Oldest queued self-hosted job age across watched repos + zombie-run
    candidates (OMN-13109 SILENT-WEDGE cross-reference)
  - Docker buildx availability probe (NEW -- closes an OMN-13932 blind spot)
  - Codeload-throttle failure-signature probe (NEW -- closes an OMN-13932
    blind spot)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_zombie_run_candidate import (
    ModelZombieRunCandidate,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
    load_runner_fleet_config,
)

logger = logging.getLogger(__name__)

# Env-overridable thresholds, defaults mirrored from runner-monitor.sh /
# healthcheck.sh (OMN-13109, OMN-13912, OMN-13915) so the node and the bash
# surfaces agree on what "stale"/"old" means during the trust-building period.
_WEDGE_QUEUE_AGE_SECONDS = int(os.environ.get("WEDGE_QUEUE_AGE_SECONDS", "600"))
_DEFAULT_WATCH_REPOS = (
    "OmniNode-ai/omnibase_infra",
    "OmniNode-ai/omnibase_core",
    "OmniNode-ai/omniclaude",
    "OmniNode-ai/omnimarket",
)
_CODELOAD_FAILURE_SIGNATURES = (
    "codeload.github.com",
    "GnuTLS",
    "fetch-pack",
    "the remote end hung up unexpectedly",
)
_CODELOAD_SCAN_LIMIT = int(os.environ.get("RUNNER_CODELOAD_SCAN_LIMIT", "5"))


def _watch_repos() -> tuple[str, ...]:
    raw = os.environ.get("WEDGE_WATCH_REPOS", "")
    if not raw:
        return _DEFAULT_WATCH_REPOS
    return tuple(raw.split())


class HandlerRunnerFleetSnapshot:
    """Gathers a read-only, facts-only runner-fleet snapshot.

    All probes are best-effort: a failed probe surfaces as a per-source
    error (``source_errors``) or a ``None``/empty fact rather than raising,
    so a partial-source outage still produces a usable (degraded) snapshot
    -- matching ``ModelRunnerHealthSnapshot``'s existing partial-failure
    contract.
    """

    def __init__(self, config: ModelRunnerFleetConfig | None = None) -> None:
        self._config = config or load_runner_fleet_config()
        self._watch_repos = _watch_repos()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _runner_index(self, name: str) -> int | None:
        prefix = f"{self._config.runner_name_prefix}-"
        if not name.startswith(prefix):
            return None
        suffix = name.removeprefix(prefix)
        if not suffix.isdigit():
            return None
        return int(suffix)

    async def _fetch_github_runners(self) -> tuple[list[dict[str, object]], str | None]:
        """Fetch runner list from the GitHub org runners API."""
        proc = await asyncio.create_subprocess_exec(
            "gh",
            "api",
            f"/orgs/{self._config.github_org}/actions/runners",
            "--paginate",
            "--jq",
            ".runners[] | {name, status, busy}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return [], (
                f"GitHub runners API exit code {proc.returncode}: "
                f"{stderr.decode(errors='replace').strip()[:200]}"
            )
        runners: list[dict[str, object]] = []
        for line in stdout.decode(errors="replace").strip().splitlines():
            if line.strip():
                runners.append(json.loads(line))
        return runners, None

    async def _fetch_docker_facts(
        self,
    ) -> tuple[dict[str, dict[str, str]], str | None]:
        """Fetch Docker container status + RestartCount + _diag heartbeat age via SSH.

        The heartbeat probe execs into the container to check
        ``${RUNNER_HOME}/_diag`` (default ``/home/runner/actions-runner``) --
        mirroring the anchored-path check OMN-13915 added to
        ``healthcheck.sh``. Reports ``-1`` when the container/probe cannot
        determine an age (treated as unknown, not zero, by the caller).
        """
        prefix = self._config.runner_name_prefix
        cmd = (
            f"for name in $(docker ps -a --filter 'name={prefix}' "
            "--format '{{.Names}}'); do "
            "status=$(docker inspect --format '{{.State.Status}}' \"$name\"); "
            "restart_count=$(docker inspect --format '{{.RestartCount}}' \"$name\"); "
            "uptime=$(docker ps -a --filter \"name=^/${name}$\" --format '{{.Status}}'); "
            'diag_age=$(docker exec "$name" bash -c '
            '\'f=$(ls -t "${RUNNER_HOME:-/home/runner/actions-runner}/_diag"/*.log '
            "2>/dev/null | head -1); "
            'if [ -n "$f" ]; then echo $(( $(date +%s) - $(stat -c %Y "$f") )); '
            "else echo -1; fi' 2>/dev/null || echo -1); "
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$name" "$status" "$restart_count" "$uptime" "$diag_age"; '
            "done"
        )
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._config.runner_host,
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {}, (
                f"SSH/Docker exit code {proc.returncode}: "
                f"{stderr.decode(errors='replace').strip()[:200]}"
            )
        result: dict[str, dict[str, str]] = {}
        for line in stdout.decode(errors="replace").strip().splitlines():
            parts = line.split("\t", 4)
            if len(parts) == 5:
                name, status, restart_count, uptime, diag_age = parts
                result[name] = {
                    "status": status,
                    "restart_count": restart_count,
                    "uptime": uptime,
                    "diag_age": diag_age,
                }
        return result, None

    async def _fetch_queue_facts(
        self,
    ) -> tuple[float | None, tuple[ModelZombieRunCandidate, ...], str | None]:
        """Fetch oldest-queued-job age + zombie-run candidates across watched repos.

        Cross-references OMN-13109's SILENT-WEDGE signal: a queued job aged
        past ``WEDGE_QUEUE_AGE_SECONDS`` is a zombie-run candidate regardless
        of per-runner state; the health COMPUTE node decides what (if
        anything) to recommend.
        """
        oldest_age: float | None = None
        candidates: list[ModelZombieRunCandidate] = []
        errors: list[str] = []
        now = datetime.now(tz=UTC)

        for repo in self._watch_repos:
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "api",
                f"/repos/{repo}/actions/runs",
                "--jq",
                '.workflow_runs[] | select(.status=="queued" or .status=="in_progress")'
                " | {id, name, status, created_at}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                errors.append(
                    f"queue probe failed for {repo}: "
                    f"{stderr.decode(errors='replace').strip()[:200]}"
                )
                continue
            for line in stdout.decode(errors="replace").strip().splitlines():
                if not line.strip():
                    continue
                run = json.loads(line)
                created_at = run.get("created_at", "")
                if not created_at:
                    continue
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age = (now - created).total_seconds()
                if oldest_age is None or age > oldest_age:
                    oldest_age = age
                if age >= _WEDGE_QUEUE_AGE_SECONDS:
                    candidates.append(
                        ModelZombieRunCandidate(
                            repo=repo,
                            run_id=int(run["id"]),
                            workflow_name=str(run.get("name", "")),
                            status=run["status"],
                            age_seconds=age,
                        )
                    )

        error_summary = "; ".join(errors) if errors else None
        return oldest_age, tuple(candidates), error_summary

    async def _fetch_buildx_available(self) -> tuple[bool | None, str | None]:
        """Probe Docker buildx availability on the runner host (OMN-13932)."""
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._config.runner_host,
            "docker buildx inspect >/dev/null 2>&1 && echo OK || echo FAIL",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return None, (
                f"buildx probe SSH exit code {proc.returncode}: "
                f"{stderr.decode(errors='replace').strip()[:200]}"
            )
        result = stdout.decode(errors="replace").strip()
        if result == "OK":
            return True, None
        if result == "FAIL":
            return False, None
        return None, f"unexpected buildx probe output: {result!r}"

    async def _fetch_codeload_throttle_signals(
        self,
    ) -> tuple[int, tuple[str, ...], str | None]:
        """Grep recent failed runs for codeload-throttle failure signatures (OMN-13932)."""
        examples: list[str] = []
        errors: list[str] = []

        for repo in self._watch_repos:
            list_proc = await asyncio.create_subprocess_exec(
                "gh",
                "run",
                "list",
                "--repo",
                repo,
                "--status",
                "failure",
                "--limit",
                str(_CODELOAD_SCAN_LIMIT),
                "--json",
                "databaseId,displayTitle",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            list_stdout, list_stderr = await list_proc.communicate()
            if list_proc.returncode != 0:
                errors.append(
                    f"codeload scan failed for {repo}: "
                    f"{list_stderr.decode(errors='replace').strip()[:200]}"
                )
                continue
            try:
                runs = json.loads(list_stdout.decode(errors="replace") or "[]")
            except json.JSONDecodeError as exc:
                errors.append(f"codeload scan invalid JSON for {repo}: {exc}")
                continue

            for run in runs:
                run_id = run.get("databaseId")
                if run_id is None:
                    continue
                log_proc = await asyncio.create_subprocess_exec(
                    "gh",
                    "run",
                    "view",
                    str(run_id),
                    "--repo",
                    repo,
                    "--log-failed",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                log_stdout, _log_stderr = await log_proc.communicate()
                log_text = log_stdout.decode(errors="replace")
                for signature in _CODELOAD_FAILURE_SIGNATURES:
                    if signature.lower() in log_text.lower():
                        examples.append(f"{repo}#{run_id}: matched {signature!r}")
                        break

        error_summary = "; ".join(errors) if errors else None
        return len(examples), tuple(examples), error_summary

    async def handle(self, correlation_id: UUID) -> ModelRunnerFleetSnapshot:
        """Gather a read-only, point-in-time runner-fleet snapshot.

        Args:
            correlation_id: Workflow correlation ID.

        Returns:
            ModelRunnerFleetSnapshot with facts only -- no classification.
        """
        logger.info(
            "Gathering runner-fleet snapshot (correlation_id=%s)", correlation_id
        )

        (
            gh_result,
            docker_result,
            queue_result,
            buildx_result,
            codeload_result,
        ) = await asyncio.gather(
            self._fetch_github_runners(),
            self._fetch_docker_facts(),
            self._fetch_queue_facts(),
            self._fetch_buildx_available(),
            self._fetch_codeload_throttle_signals(),
        )
        github_runners, gh_error = gh_result
        docker_facts, docker_error = docker_result
        oldest_queued_age, zombie_candidates, queue_error = queue_result
        buildx_available, buildx_error = buildx_result
        codeload_signal_count, codeload_examples, codeload_error = codeload_result

        source_errors: list[str] = []
        for error in (
            gh_error,
            docker_error,
            queue_error,
            buildx_error,
            codeload_error,
        ):
            if error:
                source_errors.append(error)

        facts: list[ModelRunnerFleetRunnerFact] = []
        seen_docker_names: set[str] = set()

        # Forward pass: GitHub registrations -> look up Docker facts.
        for gh in github_runners:
            name = str(gh["name"])
            index = self._runner_index(name)
            if index is None or index > self._config.expected_count:
                continue
            seen_docker_names.add(name)
            docker = docker_facts.get(
                name,
                {
                    "status": "not_found",
                    "restart_count": "0",
                    "uptime": "",
                    "diag_age": "-1",
                },
            )
            diag_age_raw = docker.get("diag_age", "-1")
            diag_age = float(diag_age_raw) if diag_age_raw not in ("-1", "") else None
            try:
                restart_count = int(docker.get("restart_count", "0"))
            except ValueError:
                restart_count = 0
            facts.append(
                ModelRunnerFleetRunnerFact(
                    name=name,
                    github_status=str(gh["status"]),
                    github_busy=bool(gh["busy"]),
                    docker_status=docker["status"],
                    docker_uptime=docker.get("uptime", ""),
                    docker_restart_count=restart_count,
                    diag_heartbeat_age_seconds=diag_age,
                )
            )

        # Reverse pass: Docker containers with no GitHub registration.
        for docker_name, docker_info in docker_facts.items():
            index = self._runner_index(docker_name)
            if index is None or index > self._config.expected_count:
                continue
            if docker_name in seen_docker_names:
                continue
            diag_age_raw = docker_info.get("diag_age", "-1")
            diag_age = float(diag_age_raw) if diag_age_raw not in ("-1", "") else None
            try:
                restart_count = int(docker_info.get("restart_count", "0"))
            except ValueError:
                restart_count = 0
            facts.append(
                ModelRunnerFleetRunnerFact(
                    name=docker_name,
                    github_status="not_registered",
                    github_busy=False,
                    docker_status=docker_info["status"],
                    docker_uptime=docker_info.get("uptime", ""),
                    docker_restart_count=restart_count,
                    diag_heartbeat_age_seconds=diag_age,
                    stale_registration=True,
                    error="Docker container exists but not registered in GitHub",
                )
            )

        snapshot = ModelRunnerFleetSnapshot(
            correlation_id=correlation_id,
            collected_at=datetime.now(tz=UTC),
            host=self._config.runner_host,
            expected_count=self._config.expected_count,
            runners=tuple(facts),
            oldest_queued_job_age_seconds=oldest_queued_age,
            zombie_run_candidates=zombie_candidates,
            buildx_available=buildx_available,
            codeload_throttle_signal_count=codeload_signal_count,
            codeload_throttle_examples=codeload_examples,
            github_source_ok=gh_error is None,
            docker_source_ok=docker_error is None,
            source_errors=tuple(source_errors),
        )
        logger.info(
            "Runner-fleet snapshot gathered: %d runners observed (correlation_id=%s)",
            len(facts),
            correlation_id,
        )
        return snapshot


__all__ = ["HandlerRunnerFleetSnapshot"]
