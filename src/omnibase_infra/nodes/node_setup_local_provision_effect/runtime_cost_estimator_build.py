# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host I/O and composition root for the provisioning effect's estimator build."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from subprocess import TimeoutExpired
from types import ModuleType

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_cost_estimator_build import (
    REVISION_LABEL,
    SOURCE_LABEL,
    HandlerCostEstimatorBuild,
    validate_source,
)
from omnibase_infra.handlers.handler_cost_estimator_build import (
    dev_lane_build_settings as parse_build_settings,
)
from omnibase_infra.models.model_cost_estimator_build_config import (
    ModelCostEstimatorBuildConfig,
)
from omnibase_infra.models.model_cost_estimator_build_request import (
    ModelCostEstimatorBuildRequest,
)
from omnibase_infra.protocols.protocol_cost_estimator_build_io import (
    ProtocolCostEstimatorBuildIO,
)

logger = logging.getLogger(__name__)


class CostEstimatorBuildIO:
    """Only the host effect spawns processes or reads the filesystem."""

    def execute_build_command(
        self, argv: list[str], timeout: int, *, allow_failure: bool = False
    ) -> str | None:
        result = subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=scrub_git_location_env(),
        )
        if result.returncode:
            if allow_failure:
                return None
            raise RuntimeError(
                f"estimator {argv[0]} {argv[1]} exited {result.returncode}"
            )
        return result.stdout.strip()

    def build_image(self, argv: list[str], timeout: int, identity_path: str) -> str:
        self.execute_build_command(argv, timeout)
        return Path(identity_path).read_text(encoding="utf-8").strip()

    def workspace(self) -> AbstractContextManager[str]:
        return tempfile.TemporaryDirectory(prefix="estimator-build-")


def _declaration(execution: ModuleType) -> dict[str, object]:
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError("estimator build requires the deploy agent source location")
    contract = (
        Path(module_file).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    declaration: dict[str, object] = yaml.safe_load(
        contract.read_text(encoding="utf-8")
    )["config"]["cost_estimator_build"]
    return declaration


def dev_lane_build_settings(
    execution: ModuleType, lane: str, scope: object = None
) -> tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None:
    """The estimator build settings, or ``None``; a given ``scope`` must be RUNTIME."""
    if (
        lane != execution.EnumRuntimeLane.DEV
        or scope not in (None, execution.Scope.RUNTIME)
        or "cost-estimator-web" in execution.lane_config_for(lane).disabled_services
    ):
        return None
    return parse_build_settings(execution, lane, _declaration(execution))


class ModelCostEstimatorWebFollow(BaseModel):
    """The declared web source, with the two reads its idle follower decides on.

    Built from the build declaration: the repository, url and branch are the
    build's own settings; ``container`` is the contract's ``follow`` block.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    container: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    revision_label: str = Field(min_length=1)
    read_timeout_seconds: int = Field(gt=0)

    def running_ref(self) -> str | None:
        """The web commit the running estimator image was built from, or ``None``.

        ``None`` for an absent container, labels that are not JSON, an image built
        from another source repository, or a revision that is not a full commit
        sha: an unread revision proves nothing and the decision refuses.
        """
        output = _read(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                self.container,
            ],
            self.read_timeout_seconds,
        )
        try:
            labels = json.loads(output or "null")
        except ValueError:
            return None
        if (
            not isinstance(labels, dict)
            or labels.get(self.source_label) != self.repository
        ):
            return None
        return _sha(labels.get(self.revision_label))

    def head_ref(self) -> str | None:
        """The head of the declared branch on the declared url, or ``None``."""
        output = _read(
            [
                "git",
                "ls-remote",
                "--",
                self.repository_url,
                f"refs/heads/{self.branch}",
            ],
            self.read_timeout_seconds,
        )
        return _sha(next(iter((output or "").split()), None))


def _read(argv: list[str], timeout: int) -> str | None:
    try:
        result = subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=scrub_git_location_env(),
        )
    except (OSError, TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _sha(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    return None


def dev_lane_web_follow(
    execution: ModuleType, lane: str
) -> ModelCostEstimatorWebFollow | None:
    """The web source the idle follower tracks, from the declaration (OMN-22808).

    ``None`` when the lane is not dev, the estimator service is disabled, the
    declaration has no ``follow`` block, the lane fragment names no source yet
    (the build itself refuses that, naming the missing settings, before it
    touches anything), or the declaration cannot be read: the follower then
    follows nothing and says why, and the agent's other followed repositories
    are unaffected. Nothing here reads the gate secret.
    """
    if (
        lane != execution.EnumRuntimeLane.DEV
        or "cost-estimator-web" in execution.lane_config_for(lane).disabled_services
    ):
        return None
    try:
        declaration = _declaration(execution)
        follow = declaration.get("follow")
        build = declaration["build"]
        if not isinstance(follow, dict) or not isinstance(build, dict):
            return None
        environment = execution._compose_env()
        values = {
            field: environment.get(str(build[f"{field}_env"]), "").strip()
            for field in ("repository", "repository_url")
        }
        if not all(values.values()):
            return None
        validate_source(values["repository"], values["repository_url"])
        return ModelCostEstimatorWebFollow(
            **values,
            branch=str(build["branch"]),
            container=str(follow["container"]),
            source_label=SOURCE_LABEL,
            revision_label=REVISION_LABEL,
            read_timeout_seconds=int(build["read_timeout_seconds"]),
        )
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, yaml.YAMLError):
        logger.warning("estimator web follow declaration unreadable", exc_info=True)
        return None


#: OMN-24903: the projection writers the stability lane's compose overlay
#: declares (OMN-17562). ``services_for_scope`` names them on the dev lane only, so a
#: stability deploy rebuilt and recreated the runtime and never these: on
#: 2026-11-16 the stability runtime carried omnimarket 0.4.321 while
#: projection-delegation-writer still ran 0.4.283 from 2026-11-12, refused every
#: delegate-skill terminal of the nightly as malformed, and left each
#: delegation_events row with no terminal_ok.
STABILITY_LANE_WRITER_SERVICES: tuple[str, ...] = (
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "tenant-projection-writer",
)


def stability_lane_writer_services(execution: ModuleType, lane: str) -> tuple[str, ...]:
    """The writers this lane runs: none unless its compose files carry the overlay."""
    config = execution.lane_config_for(lane)
    if execution._STABILITY_OVERLAY not in config.compose_files:
        return ()
    return tuple(
        service
        for service in STABILITY_LANE_WRITER_SERVICES
        if service not in config.disabled_services
    )


def build_stability_lane_writers(
    execution: ModuleType,
    build_images: Callable[..., None],
    git_sha: str,
    on_phase_update: Callable[..., None],
    *,
    build_source: str,
    lane: str,
    git_ref: str,
) -> tuple[str, ...]:
    """Build the stability lane's writers at the deploy's sha. Nothing is recreated.

    Called from ``build_dev_lane_services``, which the executor runs on every
    RUNTIME and FULL rebuild AFTER the runtime image build, so the workspace is
    already staged at the deploy's sha (``stage_workspace=False`` reuses it) and
    BEFORE ``_preflight_runtime_boot``. A build touches no running container, so
    a refused preflight leaves the lane as it was; the recreate is
    :func:`roll_writers`, which the executor calls only once
    the preflight has passed and, on a FULL rebuild, the deps leg has converged.
    """
    services = stability_lane_writer_services(execution, lane)
    if not services:
        return ()
    build_images(
        execution.Scope.RUNTIME,
        git_sha,
        on_phase_update,
        build_source=build_source,
        runtime_lane=lane,
        git_ref=git_ref,
        compose_files=execution.lane_config_for(lane).compose_files,
        services=services,
        stage_workspace=False,
    )
    # OMN-25440: the images the roll will inspect are read here, before the boot
    # preflight and the deps leg, so a writer built under another tag than the
    # one the roll reads refuses the deploy before anything is converged.
    _assert_images_at_deploy_sha(
        execution, lane, services, execution._compose_env(), git_sha
    )
    return services


#: The image label docker/Dockerfile.runtime stamps in its runtime stage from the
#: ``VCS_REF`` build argument, which ``_compose_build`` passes as the full deploy
#: sha beside ``GIT_SHA``.
IMAGE_REVISION_LABEL = "org.opencontainers.image.revision"
#: The runtime main, whose freshly built image the writers must match.
RUNTIME_SERVICE = "omninode-runtime"
#: ``healthcheck`` of a writer: 60 s start period, then 30 s interval, 3 retries.
WRITER_HEALTH_WAIT_SECONDS = 240
WRITER_HEALTH_POLL_SECONDS = 5
_INSPECT_FORMAT = (
    "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}"
    "{{else}}none{{end}}|{{.RestartCount}}"
)


def _compose_prefix(execution: ModuleType, lane: str) -> list[str]:
    """The one argv prefix of every compose command the roll runs.

    ``up``, ``config`` and ``ps`` read the same project and the same profile:
    the writers sit in the ``runtime`` profile, so a command without it does not
    see them.
    """
    return [*execution._compose_argv(lane), "--profile", "runtime"]


def writer_roll(
    execution: ModuleType, lane: str, scope: object, git_sha: str
) -> dict[str, Callable[..., tuple[str, ...]]]:
    """The roll as the ``after_migrations`` step ``_compose_up`` runs, or ``{}``.

    The runtime ``_compose_up`` runs ``_ensure_runtime_migrations_ready`` and
    ``rerun_changed_oneshots`` first; the writers must start on the migrated
    schema, so the executor hands this to ``_compose_up`` instead of rolling
    them before it. ``_compose_up`` runs outside the build-context lock, and
    the step receives the environment the runtime up runs under. ``{}`` when
    the lane has no writers or the scope builds none: ``_compose_up`` is then
    called exactly as before.

    Called at the top of ``rebuild_scope``: the roll compares every image's
    revision label with ``git_sha``, so an empty one is refused here, before
    any build, boot preflight or deps up, instead of after them.
    """
    scopes = (execution.Scope.RUNTIME, execution.Scope.FULL)
    if scope not in scopes or not stability_lane_writer_services(execution, lane):
        return {}
    if not git_sha:
        raise RuntimeError(_NO_SHA)
    return {"after_migrations": _HandedOverRoll(execution, lane, git_sha)}


_NO_SHA = "stability lane writer check needs the deploy's git sha"
_NOT_ROLLED = (
    "stability lane writer roll was handed to the runtime up and did not run; "
    "the writers were not rolled"
)


class _HandedOverRoll:
    """The ``after_migrations`` step; it records whether ``_compose_up`` ran it."""

    def __init__(self, execution: ModuleType, lane: str, git_sha: str) -> None:
        self.execution, self.lane, self.git_sha = execution, lane, git_sha
        self.ran = False

    def __call__(self, extra_env: dict[str, str] | None = None) -> tuple[str, ...]:
        rolled = roll_writers(self.execution, self.lane, self.git_sha, extra_env)
        self.ran = True
        return rolled


def handed_over(
    compose_up: Callable[..., None], roll: dict[str, Callable[..., object]]
) -> Callable[..., None]:
    """``compose_up`` with ``roll`` handed over; raises when a handed roll did not run.

    OMN-25440: ``_compose_up`` runs ``after_migrations`` only inside its RUNTIME
    block and returns early, before that block, for a request of retired roles
    only. A roll skipped that way would leave the old writers running under a
    passing deploy, so the deploy fails instead.
    """

    def up(*args: object, **kwargs: object) -> None:
        compose_up(*args, **kwargs, **roll)
        step = roll.get("after_migrations")
        if isinstance(step, _HandedOverRoll) and not step.ran:
            raise RuntimeError(_NOT_ROLLED)

    return up


def roll_writers(
    execution: ModuleType,
    lane: str,
    git_sha: str,
    extra_env: dict[str, str] | None = None,
    *,
    wait_seconds: float = WRITER_HEALTH_WAIT_SECONDS,
    poll_seconds: float = WRITER_HEALTH_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, ...]:
    """Recreate the writers on their fresh images, then read them healthy.

    The images are checked BEFORE anything is replaced: every writer image must
    carry the runtime image's revision, and that revision must be the deploy's
    sha, so a mismatch leaves the running writers untouched. ``up -d`` exiting 0
    says only that compose started a container, so the roll then fails the
    deploy unless every writer is running, healthy and has not restarted.

    ``extra_env`` is the environment the executor's builder returned, which the
    executor passes to ``_compose_up`` for the runtime up that follows; the
    writers' commands run under ``_compose_env(extra_env)``, the same environment
    ``run_compose_up_steps`` builds for that up.
    """
    services = stability_lane_writer_services(execution, lane)
    if not services:
        return ()
    environment = execution._compose_env(extra_env)
    _assert_images_at_deploy_sha(execution, lane, services, environment, git_sha)
    argv = [
        *_compose_prefix(execution, lane),
        "up",
        "-d",
        "--no-deps",
        "--no-build",
        "--force-recreate",
        *services,
    ]
    result = execution._run(
        argv,
        timeout=execution.PHASE_TIMEOUTS[execution.Phase.RUNTIME],
        env=environment,
    )
    if result.returncode:
        raise RuntimeError(
            f"stability lane writer recreate exited {result.returncode} "
            f"for {', '.join(services)}"
        )
    _wait_writers_healthy(
        execution, lane, services, environment, wait_seconds, poll_seconds, sleep
    )
    return services


def _read_docker(
    execution: ModuleType, argv: list[str], environment: dict[str, str]
) -> str:
    result = execution._run(
        argv, timeout=execution.RUNTIME_HEALTH_PROBE_TIMEOUT_SECONDS, env=environment
    )
    if result.returncode:
        raise RuntimeError(f"stability lane writer check: {' '.join(argv[:3])} failed")
    return str(result.stdout).strip()


def _writer_container(
    execution: ModuleType, lane: str, service: str, environment: dict[str, str]
) -> str:
    container = _read_docker(
        execution,
        [*_compose_prefix(execution, lane), "ps", "-a", "-q", service],
        environment,
    )
    if not container:
        raise RuntimeError(f"stability lane writer {service} has no container")
    return container.splitlines()[0]


def _writer_state(
    execution: ModuleType, container: str, environment: dict[str, str]
) -> tuple[str, str, int]:
    status, health, restarts = _read_docker(
        execution,
        ["docker", "inspect", "--format", _INSPECT_FORMAT, container],
        environment,
    ).split("|")
    return status, health, int(restarts)


def _read_writer(
    execution: ModuleType, lane: str, service: str, environment: dict[str, str]
) -> tuple[str, str]:
    """One writer's (status, health); raises when it is crashing or has no health."""
    container = _writer_container(execution, lane, service, environment)
    status, health, restarts = _writer_state(execution, container, environment)
    if (
        status in ("restarting", "exited", "dead")
        or restarts
        or health in ("unhealthy", "none")
    ):
        raise RuntimeError(
            f"stability lane writer {service} is not stable after the "
            f"recreate: status={status} health={health} restarts={restarts}"
        )
    return status, health


def _wait_writers_healthy(
    execution: ModuleType,
    lane: str,
    services: tuple[str, ...],
    environment: dict[str, str],
    wait_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None],
) -> None:
    deadline = time.monotonic() + wait_seconds
    pending = list(services)
    while pending:
        for service in tuple(pending):
            if _read_writer(execution, lane, service, environment) == (
                "running",
                "healthy",
            ):
                pending.remove(service)
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "stability lane writer(s) not healthy within "
                f"{wait_seconds:g}s: {', '.join(pending)}"
            )
        sleep(poll_seconds)
    # A writer read healthy early is not read again by the loop above, and a
    # later one may take a while: the roll passes only when every writer is
    # still running, healthy and unrestarted on one last read after the others.
    unhealthy = [
        service
        for service in services
        if _read_writer(execution, lane, service, environment) != ("running", "healthy")
    ]
    if unhealthy:
        raise RuntimeError(
            f"stability lane writer(s) not healthy on the final read: "
            f"{', '.join(unhealthy)}"
        )


def _service_image(
    execution: ModuleType, lane: str, service: str, environment: dict[str, str]
) -> str:
    """The image compose will start ``service`` from, read from its own entry.

    ``config --images <service>`` also prints the images of the service's
    dependencies (a writer lists the broker's after its own), so a line of that
    output cannot be told apart from the service's. ``config --format json``
    keys each service by name; a service with ``build:`` and no ``image:`` has
    no image field there, and compose names its image ``<project>-<service>``.
    """
    rendered = json.loads(
        _read_docker(
            execution,
            [
                *_compose_prefix(execution, lane),
                "config",
                "--format",
                "json",
                service,
            ],
            environment,
        )
    )
    body = rendered.get("services", {}).get(service)
    if not isinstance(body, dict):
        raise RuntimeError(f"stability lane writer check: {service} is not rendered")
    return str(body.get("image") or f"{rendered['name']}-{service}")


def _service_revision(
    execution: ModuleType, lane: str, service: str, environment: dict[str, str]
) -> str:
    """The revision label of the image compose will start ``service`` from."""
    image = _service_image(execution, lane, service, environment)
    return _read_docker(
        execution,
        [
            "docker",
            "image",
            "inspect",
            "--format",
            f'{{{{index .Config.Labels "{IMAGE_REVISION_LABEL}"}}}}',
            image,
        ],
        environment,
    )


def _assert_images_at_deploy_sha(
    execution: ModuleType,
    lane: str,
    services: tuple[str, ...],
    environment: dict[str, str],
    git_sha: str,
) -> None:
    """Each writer image carries the revision of the runtime image this deploy built.

    Runs before the recreate. The runtime image's label is read (never taken
    from ``git_sha`` alone) and must equal the deploy's sha; each writer's label
    must equal the runtime's, so a writer left on an older image than the
    runtime this deploy is rolling out is refused with nothing replaced.
    """
    if not git_sha:
        raise RuntimeError(_NO_SHA)
    runtime = _service_revision(execution, lane, RUNTIME_SERVICE, environment)
    if runtime != git_sha:
        raise RuntimeError(
            f"stability lane runtime image revision {runtime[:12] or 'unset'} is "
            f"not the deploy's {git_sha[:12]}; no writer was recreated"
        )
    stale = [
        service
        for service in services
        if _service_revision(execution, lane, service, environment) != runtime
    ]
    if stale:
        raise RuntimeError(
            "stability lane writer(s) not on the runtime's image revision "
            f"{runtime[:12]}; no writer was recreated: {', '.join(stale)}"
        )


def build_dev_lane_services(
    execution: ModuleType,
    build_images: Callable[..., None],
    git_sha: str,
    on_phase_update: Callable[..., None],
    *,
    build_source: str,
    lane: str,
    git_ref: str,
    read_settings: Callable[
        [], tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None
    ]
    | None = None,
) -> dict[str, str]:
    """Validate first, then build dev images before the existing recreate."""
    build_stability_lane_writers(
        execution,
        build_images,
        git_sha,
        on_phase_update,
        build_source=build_source,
        lane=lane,
        git_ref=git_ref,
    )
    if lane != execution.EnumRuntimeLane.DEV:
        return {}
    settings = (
        read_settings()
        if read_settings is not None
        else dev_lane_build_settings(execution, lane)
    )
    if execution.DEV_LANE_BUILD_SERVICES:
        build_images(
            execution.Scope.RUNTIME,
            git_sha,
            on_phase_update,
            build_source=build_source,
            runtime_lane=lane,
            git_ref=git_ref,
            compose_files=execution.lane_config_for(lane).compose_files,
            services=execution.DEV_LANE_BUILD_SERVICES,
            stage_workspace=False,
        )
    if settings is None:
        return {}
    recipe, environment = settings
    container = ModelONEXContainer()
    try:
        io_key: type[object] = ProtocolCostEstimatorBuildIO
        run_coro_sync(
            container.service_registry.register_instance(
                io_key,
                CostEstimatorBuildIO(),
            )
        )
        run_coro_sync(
            container.service_registry.register_instance(
                ModelCostEstimatorBuildConfig,
                recipe,
            )
        )
        run_coro_sync(
            container.service_registry.register_instance(
                HandlerCostEstimatorBuild,
                HandlerCostEstimatorBuild(container),
            )
        )
        handler = container.get_service(HandlerCostEstimatorBuild)
        result = handler.handle(
            ModelCostEstimatorBuildRequest(
                lane="dev",
                enabled=True,
                image=environment.get("COST_ESTIMATOR_WEB_IMAGE"),
            )
        )
        if result.image is None:
            raise RuntimeError("estimator build returned no image identity")
        return {
            "COST_ESTIMATOR_WEB_IMAGE": result.image,
            "FEATURE_ESTIMATOR_GATE_ENABLED": environment.get(
                "FEATURE_ESTIMATOR_GATE_ENABLED", "true"
            ),
            "ESTIMATOR_GATE_PASSWORD": environment.get("ESTIMATOR_GATE_PASSWORD", ""),
        }
    finally:
        container.close()
