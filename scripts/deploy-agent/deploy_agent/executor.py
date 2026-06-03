# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executor with phase timeouts for deploy operations."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel, ConfigDict

from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelHealthCheck,
    ModelRebuildRequested,
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

REPO_DIR = os.environ.get(
    "DEPLOY_AGENT_REPO_DIR", "/data/omninode/omni_home/omnibase_infra"
)
DEPLOY_AGENT_DIR = os.environ.get(
    "DEPLOY_AGENT_DIR", "/data/omninode/omnibase_infra/scripts/deploy-agent"
)
COMPOSE_FILE = f"{REPO_DIR}/docker/docker-compose.infra.yml"
COMPOSE_PROJECT = "omnibase-infra"
RUNTIME_POLICY_ENV_FILE = Path(REPO_DIR) / "docker" / "runtime-policy.env"

PHASE_TIMEOUTS = {
    Phase.PREFLIGHT: 30,
    Phase.GIT: 60,
    Phase.COMPOSE_GEN: 120,
    Phase.CORE: 300,
    Phase.RUNTIME: 300,
    Phase.VERIFICATION: 120,
}

PhaseCallback = Callable[[Phase, PhaseStatus], None]

RUNTIME_HEALTH_TARGETS: tuple[tuple[str, int], ...] = (
    ("omninode-runtime", 8085),
    ("runtime-effects", 8086),
)

_BUILD_SOURCE_ALLOWED = ", ".join(source.value for source in BuildSource)

# Promotion-lineage guard (OMN-12626, R1). Loaded from scripts/ by file path
# because scripts/ is not an importable package. The guard refuses to build a
# prod-bound (release-mode) image from a dirty or non-promoted source tree.
_PROMOTION_GUARD_PATH = Path(REPO_DIR) / "scripts" / "check_prod_promotion_lineage.py"


def _load_promotion_guard() -> ModuleType:
    """Load the prod promotion-lineage guard module from scripts/ by path.

    Raises RuntimeError (fail-fast) when the guard is missing so a release
    build can never silently skip the clean-tree + promoted-lineage check.
    """
    mod_name = "check_prod_promotion_lineage"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if not _PROMOTION_GUARD_PATH.is_file():
        raise RuntimeError(
            "prod promotion-lineage guard not found at "
            f"{_PROMOTION_GUARD_PATH}; cannot verify clean+promoted build source."
        )
    spec = importlib.util.spec_from_file_location(mod_name, _PROMOTION_GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load prod promotion-lineage guard from {_PROMOTION_GUARD_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def assert_release_build_promoted(
    build_source: BuildSource, *, repo_dir: str = REPO_DIR
) -> None:
    """Enforce clean + promoted source for prod-bound (release-mode) builds.

    Release-mode builds produce the digest that is later pinned and promoted to
    prod. They MUST come from a clean working tree whose HEAD is an
    ancestor-of/equal-to origin/main. Workspace builds (local dev iteration) are
    exempt by design — they never reach prod.

    Raises the guard's ``ProdLineageError`` when the source is dirty or
    not promoted. Fails the build CLOSED before any docker build side effects.
    """
    if build_source != BuildSource.RELEASE:
        return
    guard = _load_promotion_guard()
    sha = guard.assert_prod_build_promoted(Path(repo_dir))
    logger.info(
        "assert_release_build_promoted: release build source %s is clean and "
        "promoted (HEAD %s is ancestor-of/equal-to origin/main)",
        repo_dir,
        sha[:12],
    )


class DigestMismatchError(RuntimeError):
    """Raised when the running container image digest != the requested digest.

    Fails the deploy closed before any health check runs — a lane must never be
    marked healthy while serving an artifact that does not match the pinned
    (stability-proven) digest.
    """


class ProdStabilityDigestMissingError(RuntimeError):
    """Raised when a prod deploy request lacks a matching stability READY digest.

    This is a boundary-level guard: it must fire before any deploy effect runs,
    not just before health checks.
    """


class ModelLaneConfig(BaseModel):
    """Per-lane compose file(s), compose project, and health targets.

    The base ``docker-compose.infra.yml`` is always the first compose file;
    non-dev lanes layer their overlay (``docker-compose.<lane>.yml``) on top so
    the overlay's container names, project, and host port bindings win.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: EnumRuntimeLane
    compose_files: tuple[str, ...]
    compose_project: str
    postgres_container: str
    runtime_health_targets: tuple[tuple[str, int], ...]


_STABILITY_OVERLAY = f"{REPO_DIR}/docker/docker-compose.stability-test.yml"
_PROD_OVERLAY = f"{REPO_DIR}/docker/docker-compose.prod.yml"

_LANE_CONFIGS: dict[EnumRuntimeLane, ModelLaneConfig] = {
    EnumRuntimeLane.DEV: ModelLaneConfig(
        lane=EnumRuntimeLane.DEV,
        compose_files=(COMPOSE_FILE,),
        compose_project=COMPOSE_PROJECT,
        postgres_container="omnibase-infra-postgres",
        runtime_health_targets=RUNTIME_HEALTH_TARGETS,
    ),
    EnumRuntimeLane.STABILITY_TEST: ModelLaneConfig(
        lane=EnumRuntimeLane.STABILITY_TEST,
        compose_files=(COMPOSE_FILE, _STABILITY_OVERLAY),
        compose_project="omnibase-infra-stability-test",
        postgres_container="omnibase-infra-stability-test-postgres",
        runtime_health_targets=(
            ("omninode-runtime", 18085),
            ("runtime-effects", 18086),
        ),
    ),
    EnumRuntimeLane.PROD: ModelLaneConfig(
        lane=EnumRuntimeLane.PROD,
        compose_files=(COMPOSE_FILE, _PROD_OVERLAY),
        compose_project="omnibase-infra-prod",
        postgres_container="omnibase-infra-prod-postgres",
        runtime_health_targets=(
            ("omninode-runtime", 28085),
            ("runtime-effects", 28086),
        ),
    ),
}


def lane_config_for(lane: EnumRuntimeLane) -> ModelLaneConfig:
    """Return the compose/project/health configuration for a runtime lane."""
    return _LANE_CONFIGS[lane]


def _compose_file_args(lane: EnumRuntimeLane) -> list[str]:
    """Return the ``-f <file>`` token sequence for a lane's compose invocation."""
    args: list[str] = []
    for compose_file in lane_config_for(lane).compose_files:
        args.extend(["-f", compose_file])
    return args


def assert_prod_request_has_stability_digest(
    cmd: ModelRebuildRequested, *, stability_ready_digest: str | None
) -> None:
    """Reject a prod request lacking a matching stability READY digest.

    Boundary-level guard: this is invoked before any deploy effect so a prod
    deploy can never start without a stability-proven artifact. Non-prod lanes
    are unaffected.
    """
    if cmd.runtime_lane != EnumRuntimeLane.PROD:
        return
    if stability_ready_digest is None:
        raise ProdStabilityDigestMissingError(
            "prod deploy rejected: no stability-test READY digest is available; "
            "production may only deploy a digest already proven in stability-test"
        )
    if cmd.image_digest != stability_ready_digest:
        raise ProdStabilityDigestMissingError(
            "prod deploy rejected: requested image_digest "
            f"{cmd.image_digest!r} does not equal the stability-test READY digest "
            f"{stability_ready_digest!r}"
        )


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


def _coerce_build_source(value: BuildSource | str, *, layer: str) -> BuildSource:
    try:
        return BuildSource(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {layer} BUILD_SOURCE={value!r}; expected one of: {_BUILD_SOURCE_ALLOWED}"
        ) from exc


def _build_source_build_args(
    build_source: BuildSource | str,
    *,
    expected_build_source: BuildSource | str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return validated immutable build-source args for docker compose build."""
    selected = _coerce_build_source(build_source, layer="deploy-agent")
    expected = _coerce_build_source(
        selected if expected_build_source is None else expected_build_source,
        layer="expected",
    )
    if selected != expected:
        raise RuntimeError(
            "BUILD_SOURCE selector disagreement: "
            f"deploy-agent selected {selected.value!r}, "
            f"Dockerfile expected {expected.value!r}"
        )

    source_env = os.environ if env is None else env
    omni_home = source_env.get("OMNI_HOME", "").strip()
    if selected == BuildSource.WORKSPACE and not omni_home:
        raise RuntimeError("BUILD_SOURCE=workspace requires OMNI_HOME before build")

    return [
        "--build-arg",
        f"BUILD_SOURCE={selected.value}",
        "--build-arg",
        f"EXPECTED_BUILD_SOURCE={expected.value}",
        "--build-arg",
        f"OMNI_HOME={omni_home}",
    ]


def _run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


def _load_runtime_policy_env(path: Path | None = None) -> dict[str, str]:
    """Load contract-rendered runtime policy env values."""
    env_path = RUNTIME_POLICY_ENV_FILE if path is None else path
    if not env_path.exists():
        return {}

    policy_env: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        policy_env[key.strip()] = value.strip()
    return policy_env


def _compose_env() -> dict[str, str]:
    env = dict(os.environ)
    for key, value in _load_runtime_policy_env().items():
        env.setdefault(key, value)
    postgres_host = env.get("POSTGRES_HOST", "127.0.0.1")
    postgres_port = env.get("POSTGRES_PORT", "5436")
    postgres_dsn = env.get("OMNIDASH_ANALYTICS_DB_URL") or (
        "postgresql://postgres:"
        f"{env.get('POSTGRES_PASSWORD', 'postgres')}@{postgres_host}:{postgres_port}/omnidash_analytics"
    )
    defaults = {
        "CI_CALLBACK_TOKEN": "deploy-agent-compose-parse-only",
        "LINEAR_WEBHOOK_SECRET": "deploy-agent-compose-parse-only",
        "WAITLIST_NOTIFIER_SLACK_BOT_TOKEN": "deploy-agent-compose-parse-only",
        "WAITLIST_NOTIFIER_SLACK_CHANNEL_ID": "deploy-agent-compose-parse-only",
        "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN": postgres_dsn,
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def _runtime_health_passed(result: subprocess.CompletedProcess) -> bool:
    """Return whether a runtime /health response proves deploy readiness."""
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    details = payload.get("details")
    if not isinstance(details, dict):
        return False
    return (
        payload.get("status") == "healthy"
        and details.get("is_running") is True
        and details.get("config_prefetch_status") in {"ok", "skipped"}
    )


def _compose_service_states(
    lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
) -> dict[str, tuple[str, int | None]]:
    result = subprocess.run(
        [
            "docker",
            "compose",
            *_compose_file_args(lane),
            "-p",
            lane_config_for(lane).compose_project,
            "ps",
            "-a",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_compose_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:200])
    states: dict[str, tuple[str, int | None]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        service = str(payload.get("Service") or "")
        if not service:
            continue
        exit_code = payload.get("ExitCode")
        states[service] = (
            str(payload.get("State") or ""),
            exit_code if isinstance(exit_code, int) else None,
        )
    return states


def _service_satisfied(state: str, exit_code: int | None) -> bool:
    if state == "running":
        return True
    return state == "exited" and exit_code == 0


def verify_containers_up(
    expected_containers: list[str],
    timeout_s: int = 120,
    *,
    lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
) -> tuple[bool, list[str]]:
    """Poll compose until services are running or completed successfully."""
    deadline = time.monotonic() + timeout_s
    last_states: dict[str, tuple[str, int | None]] = {}
    while time.monotonic() < deadline:
        try:
            last_states = _compose_service_states(lane)
        except RuntimeError as exc:
            logger.warning("verify_containers_up: docker compose ps failed: %s", exc)
            time.sleep(2)
            continue
        missing = [
            service
            for service in expected_containers
            if not _service_satisfied(*last_states.get(service, ("missing", None)))
        ]
        if not missing:
            return True, []
        logger.info(
            "verify_containers_up: waiting for %d service(s): %s",
            len(missing),
            missing,
        )
        time.sleep(2)
    try:
        last_states = _compose_service_states(lane)
    except RuntimeError:
        last_states = {}
    missing = [
        service
        for service in expected_containers
        if not _service_satisfied(*last_states.get(service, ("missing", None)))
    ]
    return False, missing


class DeployExecutor:
    def validate_llm_endpoint_env_contract(self) -> None:
        """Fail runtime deploys when configured LLM endpoints drift from contract."""
        script = f"{REPO_DIR}/scripts/check_llm_endpoint_env_contract.py"
        venv_python = f"{REPO_DIR}/.venv/bin/python"
        base_cmd = (
            [venv_python, script]
            if Path(venv_python).is_file()
            else ["uv", "run", "--project", REPO_DIR, "python", script]
        )

        env_files: list[Path] = []
        explicit_env_file = os.environ.get("OMNIBASE_ENV_FILE")
        fallback_candidates = [
            Path.home() / ".omnibase" / ".env",
            Path(REPO_DIR) / ".env",
        ]
        candidates = (
            [Path(explicit_env_file), *fallback_candidates]
            if explicit_env_file
            else fallback_candidates
        )
        for candidate in candidates:
            if candidate.is_file() and candidate not in env_files:
                env_files.append(candidate)
        if explicit_env_file and not any(
            f == Path(explicit_env_file) for f in env_files
        ):
            raise RuntimeError(
                f"LLM endpoint env contract check failed: OMNIBASE_ENV_FILE does not exist: {explicit_env_file}"
            )

        commands = [base_cmd]
        commands.extend([*base_cmd, "--env-file", str(path)] for path in env_files)
        for cmd in commands:
            result = _run(cmd, timeout=PHASE_TIMEOUTS[Phase.PREFLIGHT])
            if result.returncode != 0:
                source = "process environment"
                if "--env-file" in cmd:
                    source = cmd[cmd.index("--env-file") + 1]
                raise RuntimeError(
                    "LLM endpoint env contract check failed for "
                    f"{source}: {result.stderr.strip() or result.stdout.strip()}"
                )

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

        result = _run(cmd, timeout=timeout, cwd=REPO_DIR, env=_compose_env())
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
            env={**_compose_env(), "PYTHONPATH": f"{REPO_DIR}/src"},
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
        build_source: BuildSource | str = BuildSource.RELEASE,
        skip_self_update: bool = False,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        image_digest: str | None = None,
    ) -> list[str]:
        self.self_update(skip=skip_self_update)
        phase = Phase.CORE if scope == Scope.CORE else Phase.RUNTIME

        # prod deploys the stability-proven digest — it pulls the pinned image
        # and never rebuilds from a ref (the digest is the authority).
        if lane == EnumRuntimeLane.PROD:
            if not image_digest:
                raise ProdStabilityDigestMissingError(
                    "prod rebuild_scope requires a pinned image_digest"
                )
            self._pull_pinned_image(image_digest, lane)
            if scope == Scope.FULL:
                self._compose_up(Phase.CORE, Scope.CORE, [], on_phase_update, lane=lane)
                self._compose_up(
                    Phase.RUNTIME, Scope.RUNTIME, [], on_phase_update, lane=lane
                )
                return services_for_scope(Scope.FULL)
            self._compose_up(phase, scope, services, on_phase_update, lane=lane)
            return services if services else services_for_scope(scope)

        if scope == Scope.FULL:
            # Build images first (both scopes), then bring them up.
            # _compose_build passes --build-arg GIT_SHA so Docker invalidates
            # the COPY src/ layer even when the file-system mtime is cached.
            self._compose_build(
                Scope.CORE, git_sha, on_phase_update, build_source=build_source
            )
            self._compose_build(
                Scope.RUNTIME, git_sha, on_phase_update, build_source=build_source
            )
            self._compose_up(Phase.CORE, Scope.CORE, [], on_phase_update, lane=lane)
            self._compose_up(
                Phase.RUNTIME, Scope.RUNTIME, [], on_phase_update, lane=lane
            )
            return services_for_scope(Scope.FULL)

        self._compose_build(scope, git_sha, on_phase_update, build_source=build_source)
        self._compose_up(phase, scope, services, on_phase_update, lane=lane)
        return services if services else services_for_scope(scope)

    def _pull_pinned_image(self, image_digest: str, lane: EnumRuntimeLane) -> None:
        """Pull the exact stability-proven image digest for a prod deploy.

        Production never rebuilds from a ref; it resolves and pulls the pinned
        digest so the artifact is byte-identical to the one proven in
        stability-test.
        """
        result = _run(
            ["docker", "pull", image_digest],
            timeout=PHASE_TIMEOUTS[Phase.RUNTIME],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker pull {image_digest} failed for lane {lane.value}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info(
            "_pull_pinned_image: pulled pinned digest %s for lane %s",
            image_digest,
            lane.value,
        )

    def verify_running_image_digest(
        self, *, lane: EnumRuntimeLane, expected_digest: str
    ) -> None:
        """Verify the running runtime container's image digest == requested.

        FAILS CLOSED (raises ``DigestMismatchError``) on any mismatch. Must run
        before health checks so a lane is never marked healthy while serving an
        artifact that does not match the pinned digest.
        """
        config = lane_config_for(lane)
        runtime_container, _ = config.runtime_health_targets[0]
        result = _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{index .Image}}{{range .RepoDigests}} {{.}}{{end}}",
                runtime_container,
            ],
            timeout=PHASE_TIMEOUTS[Phase.VERIFICATION],
        )
        if result.returncode != 0:
            raise DigestMismatchError(
                f"could not inspect running image digest for {runtime_container} "
                f"(lane {lane.value}): {result.stderr.strip() or result.stdout.strip()}"
            )
        observed = result.stdout.strip()
        if expected_digest not in observed:
            raise DigestMismatchError(
                f"running container {runtime_container} image digest {observed!r} "
                f"does not contain requested digest {expected_digest!r} "
                f"(lane {lane.value}); failing closed"
            )
        logger.info(
            "verify_running_image_digest: %s matches requested digest %s (lane %s)",
            runtime_container,
            expected_digest,
            lane.value,
        )

    def deploy_and_verify(
        self,
        *,
        lane: EnumRuntimeLane,
        expected_digest: str,
        on_phase_update: PhaseCallback,
    ) -> list[ModelHealthCheck]:
        """Verify the running digest, then run health checks.

        Digest verification runs first and fails closed: a mismatch aborts
        before any health check, so a lane serving the wrong artifact can never
        be reported healthy.
        """
        self.verify_running_image_digest(lane=lane, expected_digest=expected_digest)
        return self.verify(on_phase_update=on_phase_update, lane=lane)

    @staticmethod
    def _resolve_plugin_ref(repo_dir: str) -> str:
        """Return the HEAD SHA of a plugin repo for uv cache busting (OMN-10728).

        BuildKit's uv cache mount is keyed on the install URL, not the resolved
        git HEAD. Passing @main always hits the stale cache entry. Passing the
        full SHA forces a cache miss and a fresh fetch every time main advances.

        Falls back to "main" so manual docker builds without omni_home still work.
        """
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning(
            "_resolve_plugin_ref: git rev-parse failed for %s (exit=%d): %s — falling back to branch default",
            repo_dir,
            result.returncode,
            result.stderr[:200],
        )
        return "main"

    @staticmethod
    def _stage_workspace(repo_dir: str, omni_home: str) -> None:
        """Stage sibling repos into the Docker build context for workspace mode.

        Runs docker/runtime_build/stage_workspace.sh from the repo root so that
        workspace/sibling-repos/ is populated before `docker compose build`.
        Raises RuntimeError on failure.
        """
        script = Path(repo_dir) / "scripts" / "runtime_build" / "stage_workspace.sh"
        if not script.exists():
            raise RuntimeError(
                f"workspace staging script not found: {script}. "
                "Cannot proceed with BUILD_SOURCE=workspace."
            )
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_dir,
            env={**os.environ, "OMNI_HOME": omni_home},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Workspace staging failed (exit={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info("_stage_workspace: %s", result.stdout.strip())

    def _compose_build(
        self,
        scope: Scope,
        git_sha: str,
        on_phase_update: PhaseCallback,
        *,
        build_source: BuildSource | str = BuildSource.RELEASE,
        expected_build_source: BuildSource | str | None = None,
    ) -> None:
        """Build images with --build-arg GIT_SHA to bust the COPY src/ layer cache.

        Without this arg, Docker serves a cached layer even after git pull, so
        the running container silently ships pre-pull code (root cause: PR #1231).

        Also passes OMNIBASE_COMPAT_REF, OMNIMARKET_REF, and
        ONEX_CHANGE_CONTROL_REF as full commit SHAs so the uv cache mount
        (keyed on URL) misses and fetches fresh code every time main advances
        (OMN-10728 / OMN-11542).

        For BUILD_SOURCE=workspace, stages sibling repos into the build context
        via stage_workspace.sh before invoking docker compose build (OMN-9470).
        """
        timeout = PHASE_TIMEOUTS.get(
            Phase.CORE if scope == Scope.CORE else Phase.RUNTIME, 300
        )
        profile = "core" if scope == Scope.CORE else "runtime"

        # Validate build-source selector agreement before any side effects.
        # This surfaces selector mismatch and missing OMNI_HOME before staging.
        validated_args = _build_source_build_args(
            build_source,
            expected_build_source=expected_build_source,
        )

        selected_source = _coerce_build_source(build_source, layer="deploy-agent")
        omni_home = os.environ.get("OMNI_HOME", "").strip()

        # OMN-12626 (R1): release-mode builds produce the digest that is later
        # pinned/promoted to prod. Refuse to build one from a dirty or
        # non-promoted (dev-only) source tree before any docker side effects.
        assert_release_build_promoted(selected_source)

        if selected_source == BuildSource.WORKSPACE:
            if not omni_home:
                raise RuntimeError(
                    "BUILD_SOURCE=workspace requires OMNI_HOME before build"
                )
            self._stage_workspace(REPO_DIR, omni_home)

        omnimarket_ref = (
            self._resolve_plugin_ref(f"{omni_home}/omnimarket") if omni_home else "dev"
        )
        compat_ref = (
            self._resolve_plugin_ref(f"{omni_home}/omnibase_compat")
            if omni_home
            else "main"
        )
        occ_ref = (
            self._resolve_plugin_ref(f"{omni_home}/onex_change_control")
            if omni_home
            else "main"
        )
        logger.info(
            "_compose_build: BUILD_SOURCE=%s OMNIBASE_COMPAT_REF=%s OMNIMARKET_REF=%s ONEX_CHANGE_CONTROL_REF=%s",
            selected_source.value,
            compat_ref[:12],
            omnimarket_ref[:12],
            occ_ref[:12],
        )

        import datetime

        build_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

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
            f"VCS_REF={git_sha}",
            "--build-arg",
            f"BUILD_DATE={build_date}",
            "--build-arg",
            f"OMNIBASE_COMPAT_REF={compat_ref}",
            "--build-arg",
            f"OMNIMARKET_REF={omnimarket_ref}",
            "--build-arg",
            f"ONEX_CHANGE_CONTROL_REF={occ_ref}",
        ]
        cmd.extend(validated_args)
        result = _run(cmd, timeout=timeout, env=_compose_env())
        if result.returncode != 0:
            raise RuntimeError(f"Docker compose build failed: {result.stderr}")
        if scope == Scope.RUNTIME:
            tag_result = _run(
                [
                    "docker",
                    "tag",
                    "omnibase-infra-omninode-runtime:latest",
                    "runtime:latest",
                ],
                timeout=30,
            )
            if tag_result.returncode != 0:
                raise RuntimeError(
                    f"Docker runtime image tag failed: {tag_result.stderr}"
                )

    def _compose_up(
        self,
        phase: Phase,
        scope: Scope,
        services: list[str],
        on_phase_update: PhaseCallback,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> None:
        on_phase_update(phase, PhaseStatus.IN_PROGRESS)
        timeout = PHASE_TIMEOUTS.get(phase, 300)

        config = lane_config_for(lane)
        profile = "core" if scope == Scope.CORE else "runtime"
        requested_services = _requested_services_for_up(scope, services)
        cmd = [
            "docker",
            "compose",
            *_compose_file_args(lane),
            "-p",
            config.compose_project,
            "--profile",
            profile,
            "up",
            "-d",
            "--force-recreate",
            "--pull",
            "never" if scope == Scope.RUNTIME else "always",
        ]
        # OMN-9455: runtime scope must pass --no-deps so compose cannot recreate
        # the core infra services (postgres/redpanda/valkey/infisical) declared
        # as depends_on targets of the runtime services.
        if scope == Scope.RUNTIME:
            cmd.append("--no-deps")
        if requested_services:
            cmd.extend(requested_services)

        result = _run(cmd, timeout=timeout, env=_compose_env())
        compose_up_error = result.stderr.strip() if result.returncode != 0 else ""
        if compose_up_error:
            logger.warning(
                "Docker compose up returned non-zero; verifying live service state before failing: %s",
                compose_up_error[:500],
            )

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
        ok, stuck = verify_containers_up(expected, timeout_s=120, lane=lane)
        if not ok:
            logger.warning(
                "Containers stuck after compose up — attempting docker start recovery: %s",
                stuck,
            )
            for name in stuck:
                start_result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        *_compose_file_args(lane),
                        "-p",
                        config.compose_project,
                        "up",
                        "-d",
                        "--no-deps",
                        name,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=_compose_env(),
                )
                if start_result.returncode == 0:
                    logger.info("docker start %s: ok", name)
                else:
                    logger.warning(
                        "docker start %s failed: %s", name, start_result.stderr[:200]
                    )
            ok, stuck = verify_containers_up(expected, timeout_s=60, lane=lane)
            if not ok:
                detail = (
                    f"Containers still not running after docker start recovery: {stuck}"
                )
                if compose_up_error:
                    detail = f"Docker compose up failed: {compose_up_error}; {detail}"
                raise RuntimeError(detail)
            logger.info("Recovery succeeded — all containers now running")

        on_phase_update(phase, PhaseStatus.SUCCESS)

    def verify(
        self,
        on_phase_update: PhaseCallback,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> list[ModelHealthCheck]:
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
                lane_config_for(lane).postgres_container,
                "psql",
                "-U",
                "postgres",
                "-d",
                "omnibase_infra",
                "-tAc",
                "SELECT to_regclass('public.node_service_registry') IS NOT NULL",
            ],
            timeout=timeout,
        )
        checks.append(
            ModelHealthCheck(
                service="postgres",
                endpoint="node_service_registry exists",
                status="pass" if result.stdout.strip() == "t" else "fail",
                latency_ms=0,
            )
        )

        # Runtime health endpoint checks.
        #
        # OMN-9728: deployment readiness is owned by the runtime health servers.
        # Ports 8000/8001/8002 are LLM endpoints and cannot prove that the
        # runtime or runtime-effects processes are healthy. The host ports vary
        # per lane (dev 8085/8086, stability-test 18085/18086, prod 28085/28086).
        for service, port in lane_config_for(lane).runtime_health_targets:
            start = time.monotonic()
            result = _run(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    "10",
                    f"http://localhost:{port}/health",
                ],
                timeout=10,
            )
            latency = int((time.monotonic() - start) * 1000)
            checks.append(
                ModelHealthCheck(
                    service=service,
                    endpoint=f"http://localhost:{port}/health",
                    status="pass" if _runtime_health_passed(result) else "fail",
                    latency_ms=latency,
                )
            )

        on_phase_update(Phase.VERIFICATION, PhaseStatus.SUCCESS)
        return checks
