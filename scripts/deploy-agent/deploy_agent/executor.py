# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executor with phase timeouts for deploy operations."""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from deploy_agent.events import (
    ModelHealthCheck,
    Phase,
    PhaseStatus,
    Scope,
    services_for_scope,
)

logger = logging.getLogger(__name__)

REPO_DIR = "/data/omninode/omnibase_infra"
COMPOSE_FILE = f"{REPO_DIR}/docker/docker-compose.infra.yml"
COMPOSE_PROJECT = "omnibase-infra"

PHASE_TIMEOUTS = {
    Phase.PREFLIGHT: 30,
    Phase.GIT: 60,
    Phase.CORE: 300,
    Phase.RUNTIME: 300,
    Phase.VERIFICATION: 120,
}

PhaseCallback = Callable[[Phase, PhaseStatus], None]


def _run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


class DeployExecutor:
    def preflight(self, on_phase_update: PhaseCallback) -> None:
        on_phase_update(Phase.PREFLIGHT, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.PREFLIGHT]

        # Check git remote is reachable
        result = _run(
            ["git", "-C", REPO_DIR, "ls-remote", "--exit-code", "origin"],
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git remote unreachable: {result.stderr}")

        # Check docker is available
        result = _run(["docker", "info"], timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Docker unavailable: {result.stderr}")

        on_phase_update(Phase.PREFLIGHT, PhaseStatus.SUCCESS)

    def git_pull(self, git_ref: str, on_phase_update: PhaseCallback) -> str:
        on_phase_update(Phase.GIT, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.GIT]

        # Fetch with 1 retry
        result = _run(
            ["git", "-C", REPO_DIR, "fetch", "--all", "--prune"],
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning("Git fetch failed, retrying in 5s...")
            time.sleep(5)
            result = _run(
                ["git", "-C", REPO_DIR, "fetch", "--all", "--prune"],
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git fetch failed: {result.stderr}")

        # Reset to ref
        result = _run(
            ["git", "-C", REPO_DIR, "reset", "--hard", git_ref],
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git reset --hard {git_ref} failed: {result.stderr}")

        # Get SHA
        result = _run(
            ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
            timeout=timeout,
        )
        sha = result.stdout.strip()

        on_phase_update(Phase.GIT, PhaseStatus.SUCCESS)
        return sha

    def seed_infisical(self, on_phase_update: PhaseCallback) -> None:
        """Seed Infisical with required secrets before runtime containers start.

        Non-fatal: if Infisical is unreachable or seed fails, logs a warning and
        continues — the runtime containers will fall back to env-var resolution.
        This prevents a broken Infisical from blocking deploys entirely.
        """
        from deploy_agent.events import Phase, PhaseStatus

        on_phase_update(Phase.SEED, PhaseStatus.IN_PROGRESS)
        timeout = 120  # 2 minutes max for seed

        seed_script = f"{REPO_DIR}/scripts/seed-infisical.py"
        venv_python = f"{REPO_DIR}/.venv/bin/python"

        import shutil

        # Resolve Python: prefer venv, fall back to uv, then system python3
        python_bin = (
            venv_python
            if Path(venv_python).is_file()
            else ((shutil.which("uv") and "uv run python") or "python3")
        )
        if python_bin == venv_python:
            cmd = [
                python_bin,
                seed_script,
                "--contracts-dir",
                f"{REPO_DIR}/src/omnibase_infra/nodes",
                "--create-missing-keys",
                "--execute",
            ]
        else:
            cmd = [
                "uv",
                "run",
                "--project",
                REPO_DIR,
                "python",
                seed_script,
                "--contracts-dir",
                f"{REPO_DIR}/src/omnibase_infra/nodes",
                "--create-missing-keys",
                "--execute",
            ]

        result = _run(
            cmd,
            timeout=timeout,
            env={**__import__("os").environ, "PYTHONPATH": f"{REPO_DIR}/src"},
        )
        if result.returncode != 0:
            logger.warning(
                "Infisical seed returned non-zero (exit=%d). Runtime will fall back to env-var resolution. stderr: %s",
                result.returncode,
                result.stderr[:500],
            )
        else:
            logger.info("Infisical seed complete. stdout: %s", result.stdout[-500:])

        # Mark success regardless — seed failure is non-fatal
        on_phase_update(Phase.SEED, PhaseStatus.SUCCESS)

    def rebuild_scope(
        self,
        scope: Scope,
        services: list[str],
        on_phase_update: PhaseCallback,
        *,
        git_sha: str = "",
    ) -> list[str]:
        phase = Phase.CORE if scope == Scope.CORE else Phase.RUNTIME
        if scope == Scope.FULL:
            # Build images first (both scopes), then bring them up.
            # _compose_build passes --build-arg GIT_SHA so Docker invalidates
            # the COPY src/ layer even when the file-system mtime is cached.
            self._compose_build(Scope.CORE, git_sha, on_phase_update)
            self._compose_build(Scope.RUNTIME, git_sha, on_phase_update)
            self._compose_up(Phase.CORE, Scope.CORE, [], on_phase_update)
            self._compose_up(Phase.RUNTIME, Scope.RUNTIME, [], on_phase_update)
            return services_for_scope(Scope.FULL)

        self._compose_build(scope, git_sha, on_phase_update)
        self._compose_up(phase, scope, services, on_phase_update)
        return services if services else services_for_scope(scope)

    def _compose_build(
        self,
        scope: Scope,
        git_sha: str,
        on_phase_update: PhaseCallback,
    ) -> None:
        """Build images with --build-arg GIT_SHA to bust the COPY src/ layer cache.

        Without this arg, Docker serves a cached layer even after git pull, so
        the running container silently ships pre-pull code (root cause: PR #1231).
        """
        timeout = PHASE_TIMEOUTS.get(
            Phase.CORE if scope == Scope.CORE else Phase.RUNTIME, 300
        )
        profile = "core" if scope == Scope.CORE else "runtime"
        cmd = [
            "docker",
            "compose",
            "-f",
            COMPOSE_FILE,
            "-p",
            COMPOSE_PROJECT,
            "--profile",
            profile,
            "build",
            "--build-arg",
            f"GIT_SHA={git_sha}",
        ]
        result = _run(cmd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Docker compose build failed: {result.stderr}")

    def _compose_up(
        self,
        phase: Phase,
        scope: Scope,
        services: list[str],
        on_phase_update: PhaseCallback,
    ) -> None:
        on_phase_update(phase, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS.get(phase, 300)

        profile = "core" if scope == Scope.CORE else "runtime"
        cmd = [
            "docker",
            "compose",
            "-f",
            COMPOSE_FILE,
            "-p",
            COMPOSE_PROJECT,
            "--profile",
            profile,
            "up",
            "-d",
            "--force-recreate",
            "--pull",
            "always",
        ]
        if services:
            cmd.extend(services)

        result = _run(cmd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Docker compose up failed: {result.stderr}")

        on_phase_update(phase, PhaseStatus.SUCCESS)

    def verify(self, on_phase_update: PhaseCallback) -> list[ModelHealthCheck]:
        on_phase_update(Phase.VERIFICATION, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.VERIFICATION]
        checks: list[ModelHealthCheck] = []

        # Check for unhealthy containers
        result = _run(
            ["docker", "ps", "--filter", "health=unhealthy", "--format", "{{.Names}}"],
            timeout=timeout,
        )
        unhealthy = result.stdout.strip()
        if unhealthy:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter health=unhealthy",
                    status="fail",
                    latency_ms=0,
                )
            )
        else:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter health=unhealthy",
                    status="pass",
                    latency_ms=0,
                )
            )

        # Check for restarting containers
        result = _run(
            ["docker", "ps", "--filter", "status=restarting", "--format", "{{.Names}}"],
            timeout=timeout,
        )
        restarting = result.stdout.strip()
        if restarting:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter status=restarting",
                    status="fail",
                    latency_ms=0,
                )
            )
        else:
            checks.append(
                ModelHealthCheck(
                    service="docker",
                    endpoint="docker ps --filter status=restarting",
                    status="pass",
                    latency_ms=0,
                )
            )

        # Check psql registration count
        result = _run(
            [
                "docker",
                "exec",
                "postgres",
                "psql",
                "-U",
                "omninode",
                "-d",
                "omninode",
                "-tAc",
                "SELECT count(*) FROM handler_registry",
            ],
            timeout=timeout,
        )
        try:
            count = int(result.stdout.strip())
            checks.append(
                ModelHealthCheck(
                    service="postgres",
                    endpoint="handler_registry count",
                    status="pass" if count > 0 else "fail",
                    latency_ms=0,
                )
            )
        except (ValueError, TypeError):
            checks.append(
                ModelHealthCheck(
                    service="postgres",
                    endpoint="handler_registry count",
                    status="fail",
                    latency_ms=0,
                )
            )

        # Health endpoint checks
        for service, port in [
            ("runtime", 8000),
            ("intelligence-api", 8001),
            ("contract-resolver", 8002),
        ]:
            start = time.monotonic()
            result = _run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                timeout=10,
            )
            latency = int((time.monotonic() - start) * 1000)
            checks.append(
                ModelHealthCheck(
                    service=service,
                    endpoint=f"http://localhost:{port}/health",
                    status="pass" if result.returncode == 0 else "fail",
                    latency_ms=latency,
                )
            )

        on_phase_update(Phase.VERIFICATION, PhaseStatus.SUCCESS)
        return checks
