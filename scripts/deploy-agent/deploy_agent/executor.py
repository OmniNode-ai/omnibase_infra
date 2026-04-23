# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executor with phase timeouts for deploy operations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from deploy_agent.events import (
    BuildSource,
    ModelHealthCheck,
    Phase,
    PhaseStatus,
    Scope,
    services_for_scope,
)

# Maps deploy scope to catalog bundle names used by compose_gen.
# Scope.FULL regenerates both core and runtime bundles.
SCOPE_BUNDLES: dict[Scope, list[str]] = {
    Scope.CORE: ["core"],
    Scope.RUNTIME: ["core", "runtime"],
    Scope.FULL: ["core", "runtime"],
}

logger = logging.getLogger(__name__)

REPO_DIR = "/data/omninode/omnibase_infra"
DEPLOY_AGENT_DIR = "/data/omninode/deploy-agent"
COMPOSE_FILE = f"{REPO_DIR}/docker/docker-compose.infra.yml"
COMPOSE_PROJECT = "omnibase-infra"
OMNI_HOME = os.environ.get("OMNI_HOME", "/data/omninode/omni_home")
RELEASE_MANIFEST_PATH = "docker/runtime-release-manifest.json"
TAINT_MARKER = Path(
    os.environ.get(
        "DEPLOY_AGENT_RUNTIME_TAINT_FILE",
        f"{DEPLOY_AGENT_DIR}/state/runtime-taint.json",
    )
)
SIBLING_SOURCE_SUBDIRS: dict[str, str] = {
    "omnibase_compat": "src/omnibase_compat",
    "onex_change_control": "src/onex_change_control",
    "omnimarket": "src/omnimarket",
}

PHASE_TIMEOUTS = {
    Phase.PREFLIGHT: 30,
    Phase.GIT: 60,
    Phase.COMPOSE_GEN: 120,
    Phase.CORE: 300,
    Phase.RUNTIME: 300,
    Phase.VERIFICATION: 120,
}

PhaseCallback = Callable[[Phase, PhaseStatus], None]


def _resolved_repo_dir() -> Path:
    configured = Path(REPO_DIR)
    if configured.is_dir():
        return configured
    return Path(__file__).resolve().parents[3]


def _digest_tree(root: Path) -> str:
    if not root.is_dir():
        msg = f"Missing source tree: {root}"
        raise FileNotFoundError(msg)

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _workspace_source_roots(omni_home: str) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for repo, subdir in SIBLING_SOURCE_SUBDIRS.items():
        source_root = Path(omni_home) / repo / subdir
        if not source_root.is_dir():
            msg = f"Workspace source root missing for {repo}: {source_root}"
            raise FileNotFoundError(msg)
        roots[repo] = source_root
    return roots


def _workspace_build_metadata(omni_home: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for repo, source_root in _workspace_source_roots(omni_home).items():
        metadata[f"{repo.upper()}_DIGEST"] = _digest_tree(source_root)
    return metadata


def _repo_dirty(repo_root: Path) -> bool:
    result = _run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        timeout=15,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _validate_release_manifest() -> None:
    manifest_path = _resolved_repo_dir() / RELEASE_MANIFEST_PATH
    if not manifest_path.is_file():
        msg = f"Release manifest missing: {manifest_path}"
        raise FileNotFoundError(msg)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    deps = data.get("dependencies")
    if not isinstance(deps, dict):
        msg = f"Release manifest dependencies missing or invalid: {manifest_path}"
        raise ValueError(msg)
    for repo in SIBLING_SOURCE_SUBDIRS:
        if repo not in deps:
            msg = f"Release manifest missing dependency entry for {repo}"
            raise ValueError(msg)


def _requested_services_for_up(scope: Scope, services: list[str]) -> list[str]:
    """Return the explicit service list compose should recreate for this scope.

    Runtime scope must always target the runtime service list directly so a
    runtime-only rebuild cannot recreate core infra dependencies via compose's
    dependency graph. Core and full scopes retain the historical behavior
    (compose chooses services from the active profile) by returning an empty
    list when no explicit service selection was provided.

    OMN-9455: a runtime rebuild on 2026-04-22 invoked ``docker compose
    --profile runtime up -d --force-recreate --pull always`` without a service
    list or ``--no-deps``. Compose recreated dependency graph services and
    collided with the live ``omnibase-infra-infisical`` container, breaking
    Redpanda/Postgres/Valkey/Phoenix. Forcing an explicit runtime service list
    combined with ``--no-deps`` in ``_compose_up`` prevents that regression.
    """
    if services:
        return services
    if scope == Scope.RUNTIME:
        return services_for_scope(scope)
    return []


def _run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


def verify_containers_up(
    expected_containers: list[str], timeout_s: int = 120
) -> tuple[bool, list[str]]:
    """Poll docker ps until all expected containers are running or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.State}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "verify_containers_up: docker ps failed: %s", result.stderr[:200]
            )
            time.sleep(2)
            continue
        states = {
            parts[0]: parts[1]
            for line in result.stdout.strip().splitlines()
            if (parts := line.split("\t", 1)) and len(parts) == 2
        }
        missing = [c for c in expected_containers if states.get(c) != "running"]
        if not missing:
            return True, []
        logger.info(
            "verify_containers_up: waiting for %d container(s): %s",
            len(missing),
            missing,
        )
        time.sleep(2)
    # Final check to report exactly what is still not running
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.State}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    states = {
        parts[0]: parts[1]
        for line in result.stdout.strip().splitlines()
        if (parts := line.split("\t", 1)) and len(parts) == 2
    }
    missing = [c for c in expected_containers if states.get(c) != "running"]
    return False, missing


class DeployExecutor:
    def _load_runtime_taint(self) -> dict[str, str] | None:
        if not TAINT_MARKER.is_file():
            return None
        try:
            data = json.loads(TAINT_MARKER.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"status": "tainted", "reason": "unparseable_taint_marker"}
        if not isinstance(data, dict):
            return {"status": "tainted", "reason": "invalid_taint_marker"}
        return {str(k): str(v) for k, v in data.items()}

    def mark_runtime_tainted(
        self, *, reason: str, source: str = "manual_patch"
    ) -> None:
        TAINT_MARKER.parent.mkdir(parents=True, exist_ok=True)
        TAINT_MARKER.write_text(
            json.dumps(
                {"status": "tainted", "reason": reason, "source": source},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _clear_runtime_taint(self) -> None:
        if TAINT_MARKER.exists():
            TAINT_MARKER.unlink()

    def _provenance_summary(self, build_source: BuildSource) -> dict[str, object]:
        taint = self._load_runtime_taint()
        summary: dict[str, object] = {
            "build_source": build_source.value,
            "taint": taint or {"status": "clean"},
        }
        legacy_worktrees = Path(OMNI_HOME) / "worktrees"
        if legacy_worktrees.exists() or legacy_worktrees.is_symlink():
            summary["legacy_worktrees"] = {
                "path": str(legacy_worktrees),
                "kind": "symlink" if legacy_worktrees.is_symlink() else "path",
            }

        if build_source == BuildSource.WORKSPACE:
            repos: list[dict[str, object]] = []
            for repo, source_root in _workspace_source_roots(OMNI_HOME).items():
                repo_root = Path(OMNI_HOME) / repo
                repos.append(
                    {
                        "repo": repo,
                        "source_root": str(source_root),
                        "dirty": _repo_dirty(repo_root),
                        "digest": _digest_tree(source_root),
                    }
                )
            summary["repos"] = repos
            return summary

        manifest_path = _resolved_repo_dir() / RELEASE_MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        deps = manifest.get("dependencies", {})
        summary["manifest_path"] = str(manifest_path)
        summary["manifest_digest"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        summary["repos"] = [
            {
                "repo": repo,
                "distribution": dep.get("distribution", ""),
                "version": dep.get("version", ""),
            }
            for repo, dep in sorted(deps.items())
            if isinstance(dep, dict)
        ]
        return summary

    def _log_provenance_summary(self, build_source: BuildSource) -> None:
        summary = self._provenance_summary(build_source)
        logger.info("Runtime build provenance: %s", json.dumps(summary, sort_keys=True))

    def self_update(self, *, skip: bool = False) -> None:
        """Pull and re-exec deploy-agent itself if behind origin/main.

        Called as the first step of every rebuild_scope() invocation so that
        a bug-fix merged to main is picked up before the next deploy runs.

        Safety rails:
        - Skipped entirely when DEPLOY_AGENT_NO_SELF_UPDATE=1 is set.
        - Skipped when the working tree is dirty (would discard uncommitted work).
        - skip=True (--skip-self-update CLI flag) bypasses the check.
        - Container mode (DEPLOY_AGENT_MODE=container) exits with code 42
          instead of os.execv so the supervisor can respawn from the new binary.
        """
        if skip or os.environ.get("DEPLOY_AGENT_NO_SELF_UPDATE") == "1":
            logger.info("self_update: skipped (kill-switch active)")
            return

        agent_dir = os.environ.get("DEPLOY_AGENT_DIR", DEPLOY_AGENT_DIR)
        timeout = 60

        # Abort if working tree is dirty — never silently discard changes.
        status_result = _run(
            ["git", "-C", agent_dir, "status", "--porcelain"],
            timeout=timeout,
        )
        if status_result.returncode != 0:
            logger.warning(
                "self_update: git status failed (exit=%d), skipping update",
                status_result.returncode,
            )
            return
        if status_result.stdout.strip():
            logger.warning(
                "self_update: working tree is dirty, skipping update to avoid data loss"
            )
            return

        # Fetch latest origin/main.
        fetch_result = _run(
            ["git", "-C", agent_dir, "fetch", "origin", "main"],
            timeout=timeout,
        )
        if fetch_result.returncode != 0:
            logger.warning(
                "self_update: git fetch failed (exit=%d), skipping update: %s",
                fetch_result.returncode,
                fetch_result.stderr[:200],
            )
            return

        head_result = _run(
            ["git", "-C", agent_dir, "rev-parse", "HEAD"],
            timeout=timeout,
        )
        remote_result = _run(
            ["git", "-C", agent_dir, "rev-parse", "origin/main"],
            timeout=timeout,
        )
        if head_result.returncode != 0 or remote_result.returncode != 0:
            logger.warning("self_update: rev-parse failed, skipping update")
            return

        local_sha = head_result.stdout.strip()
        remote_sha = remote_result.stdout.strip()

        if local_sha == remote_sha:
            logger.info(
                "self_update: already at origin/main (%s), nothing to do",
                local_sha[:12],
            )
            return

        logger.info(
            "self_update: behind origin/main (local=%s remote=%s), pulling and re-execing",
            local_sha[:12],
            remote_sha[:12],
        )

        pull_result = _run(
            ["git", "-C", agent_dir, "pull", "--ff-only", "origin", "main"],
            timeout=timeout,
        )
        if pull_result.returncode != 0:
            logger.warning(
                "self_update: git pull failed (exit=%d), skipping re-exec: %s",
                pull_result.returncode,
                pull_result.stderr[:200],
            )
            return

        # Sync deps so new imports are available after re-exec.
        uv_result = _run(
            ["uv", "sync", "--project", agent_dir],
            timeout=120,
        )
        if uv_result.returncode != 0:
            logger.warning(
                "self_update: uv sync failed (exit=%d), proceeding with re-exec anyway: %s",
                uv_result.returncode,
                uv_result.stderr[:200],
            )

        mode = os.environ.get("DEPLOY_AGENT_MODE", "host")
        if mode == "container":
            # Let systemd/compose restart us from the freshly-pulled source.
            logger.info(
                "self_update: container mode — exiting with code 42 for supervisor respawn"
            )
            sys.exit(42)
        else:
            logger.info("self_update: host mode — re-execing process image")
            os.execv(sys.executable, [sys.executable] + sys.argv)  # noqa: S606

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

    def compose_gen(self, bundles: list[str], on_phase_update: PhaseCallback) -> None:
        """Regenerate docker-compose.infra.yml from the catalog CLI.

        Runs ``uv run python -m omnibase_infra.docker.catalog.cli generate
        <bundles> --output <COMPOSE_FILE>`` so every deploy reflects the current
        catalog state rather than a static snapshot. This closes the drift window
        where catalog changes (e.g. new LLM_* env vars) were merged but never
        landed in the running containers (OMN-8430).

        Non-fatal if the catalog CLI binary is unavailable — logs a warning and
        continues so that a missing virtualenv does not block a deploy entirely.
        """
        on_phase_update(Phase.COMPOSE_GEN, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS[Phase.COMPOSE_GEN]

        selected = bundles if bundles else ["core", "runtime"]
        cmd = [
            "uv",
            "run",
            "--project",
            REPO_DIR,
            "python",
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            *selected,
            "--output",
            COMPOSE_FILE,
        ]

        result = _run(cmd, timeout=timeout, cwd=REPO_DIR)
        if result.returncode != 0:
            logger.warning(
                "compose_gen returned non-zero (exit=%d) — continuing with existing compose file. stderr: %s",
                result.returncode,
                result.stderr[:500],
            )
        else:
            logger.info("compose_gen complete: %s", result.stdout.strip())

        on_phase_update(Phase.COMPOSE_GEN, PhaseStatus.SUCCESS)

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
        build_source: BuildSource = BuildSource.RELEASE,
        skip_self_update: bool = False,
    ) -> list[str]:
        self.self_update(skip=skip_self_update)
        phase = Phase.CORE if scope == Scope.CORE else Phase.RUNTIME
        self._log_provenance_summary(build_source)
        if build_source == BuildSource.WORKSPACE:
            _workspace_source_roots(OMNI_HOME)
        else:
            _validate_release_manifest()
        if scope == Scope.FULL:
            # Build images first (both scopes), then bring them up.
            # _compose_build passes --build-arg GIT_SHA so Docker invalidates
            # the COPY src/ layer even when the file-system mtime is cached.
            self._compose_build(Scope.CORE, git_sha, build_source, on_phase_update)
            self._compose_build(Scope.RUNTIME, git_sha, build_source, on_phase_update)
            self._compose_up(Phase.CORE, Scope.CORE, [], on_phase_update)
            self._compose_up(Phase.RUNTIME, Scope.RUNTIME, [], on_phase_update)
            return services_for_scope(Scope.FULL)

        self._compose_build(scope, git_sha, build_source, on_phase_update)
        self._compose_up(phase, scope, services, on_phase_update)
        self._clear_runtime_taint()
        return services if services else services_for_scope(scope)

    def _compose_build(
        self,
        scope: Scope,
        git_sha: str,
        build_source: BuildSource,
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
            "--build-arg",
            f"BUILD_SOURCE={build_source.value}",
            "--build-arg",
            f"RELEASE_MANIFEST_PATH={RELEASE_MANIFEST_PATH}",
        ]
        if build_source == BuildSource.WORKSPACE:
            for key, value in _workspace_build_metadata(OMNI_HOME).items():
                cmd.extend(["--build-arg", f"{key}={value}"])
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
        requested_services = _requested_services_for_up(scope, services)
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
        # OMN-9455: runtime scope must pass --no-deps so compose cannot recreate
        # the core infra services (postgres/redpanda/valkey/infisical) declared
        # as depends_on targets of the runtime services.
        if scope == Scope.RUNTIME:
            cmd.append("--no-deps")
        if requested_services:
            cmd.extend(requested_services)

        result = _run(cmd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Docker compose up failed: {result.stderr}")

        # Verify containers actually reached running state — docker compose up exits 0
        # even when containers land in Created state (hit twice in production, 01:33 + 04:48).
        # For runtime scope, verification MUST be bounded by the requested runtime
        # service list so the runtime-only rebuild never implicitly waits on core
        # infra containers (OMN-9455).
        expected = (
            requested_services if requested_services else services_for_scope(scope)
        )
        logger.info(
            "Verifying %d container(s) reached running state: %s",
            len(expected),
            expected,
        )
        ok, stuck = verify_containers_up(expected, timeout_s=120)
        if not ok:
            logger.warning(
                "Containers stuck after compose up — attempting docker start recovery: %s",
                stuck,
            )
            for name in stuck:
                start_result = subprocess.run(
                    ["docker", "start", name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if start_result.returncode == 0:
                    logger.info("docker start %s: ok", name)
                else:
                    logger.warning(
                        "docker start %s failed: %s", name, start_result.stderr[:200]
                    )
            ok, stuck = verify_containers_up(expected, timeout_s=60)
            if not ok:
                raise RuntimeError(
                    f"Containers still not running after docker start recovery: {stuck}"
                )
            logger.info("Recovery succeeded — all containers now running")

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
