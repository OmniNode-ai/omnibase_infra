# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Start-first replacement of one dev runtime container (OMN-22151).

OMN-22006 stopped recreating runtimes whose image and config had not moved, but
a runtime whose image did move is still replaced stop-first: compose stops the
old container, creates the new one, and the lane answers nothing until the new
boot reaches /ready. Deploy-agent job 0c915304 (2026-11-02) created the dev
runtime at 16:10:31Z; :8085 refused connections at 16:11:41Z, /ready read 503
at 16:13:45Z and 16:14:16Z, and 200 from 16:14:46Z on.

This effect orders the overlap. It starts the replacement beside the running
container, polls the replacement's own /ready until it reads 200 or the declared
bound runs out, and only then stops the old one. A replacement that does not
start, exits, or does not read 200 inside the bound is removed, the old one is
left running, and the result fails with a named reason. The overlap exists only
inside this effect: no compose file declares a second runtime.

The replacement is a ``compose run`` container of the same service, so it
carries the service's image, environment, mounts and networks but publishes no
host port (compose publishes a run container's ports only with
``--service-ports``); the old container keeps the host port for the overlap.
Its /ready is read inside the replacement, at the URL the request names.

Not claimed: the deploy agent's runtime phase does not call this effect yet.
Two prerequisites of the OMN-22151 design are not built: a front that owns the
host ports and routes to whichever runtime is ready (without it the replacement
holds no host port once the old one stops), and a standby boot that keeps the
replacement out of the old runtime's consumer groups until handover (without it
the replacement's join moves partitions away from the old runtime and drops the
old runtime's /ready during the overlap).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import ModuleType

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runtime_start_first_request import (
    ModelRuntimeStartFirstRequest,
)
from omnibase_infra.models.model_runtime_start_first_result import (
    ModelRuntimeStartFirstResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

DOCKER_SECONDS = 10
COMPOSE_SECONDS = 300
REPLACEMENT_SUFFIX = "-start-first"


def _run_errors(execution: ModuleType) -> tuple[type[Exception], ...]:
    """The errors one executor ``_run`` call can raise.

    Read from the executor module so this handler imports no process-spawning
    module: only EFFECT node packages may (check-no-io-outside-effects).
    """
    return (OSError, execution.subprocess.SubprocessError)


def _running(container: str, *, execution: ModuleType) -> bool:
    argv = ["docker", "inspect", "-f", "{{.State.Running}}", container]
    result = execution._run(argv, timeout=DOCKER_SECONDS)
    return bool(result.returncode == 0 and result.stdout.strip() == "true")


def _remove(container: str, *, execution: ModuleType) -> None:
    execution._run(["docker", "rm", "-f", container], timeout=COMPOSE_SECONDS)


def _start(service: str, container: str, *, execution: ModuleType) -> str:
    """Start the replacement; ``""`` on success, else why it did not start."""
    lane = execution.EnumRuntimeLane.DEV
    argv = [
        *execution._compose_argv(lane),
        *("run", "-d", "--no-deps", "--name", container, service),
    ]
    try:
        result = execution._run(
            argv, timeout=COMPOSE_SECONDS, env=execution._compose_env()
        )
    except _run_errors(execution) as exc:
        return sanitize_error_message(exc)
    if result.returncode == 0:
        return ""
    # Compose stderr can carry interpolated secrets; the exit code suffices.
    return f"compose run exited {result.returncode}"


def _ready_status(container: str, url: str, *, execution: ModuleType) -> int:
    argv = ["docker", "exec", container, "curl", "-s", "-o", "/dev/null"]
    argv += ["-w", "%{http_code}", "--max-time", str(DOCKER_SECONDS), url]
    try:
        result = execution._run(argv, timeout=DOCKER_SECONDS + 5)
    except _run_errors(execution):
        return 0
    if result.returncode != 0:
        return 0
    code = str(result.stdout).strip()
    return int(code) if code.isdigit() else 0


class HandlerRuntimeStartFirst:
    """Replace one DEV runtime container start-first under its caller's lock."""

    def __init__(
        self,
        *,
        execution: ModuleType,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.execution = execution
        self.clock = clock
        self.sleep = sleep

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure handler over the host's Docker compose project."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting: starts, stops and removes dev runtime containers."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelRuntimeStartFirstRequest
    ) -> ModelRuntimeStartFirstResult:
        old = request.old_container
        new = f"{old}{REPLACEMENT_SUFFIX}"
        run = self.execution
        base = {"service": request.service, "old_container": old, "replacement": new}
        if not _running(old, execution=run):
            return ModelRuntimeStartFirstResult(
                **base,
                status="fail",
                failure="old-runtime-not-running",
                detail=f"{old} is not running, so there is nothing to keep serving",
            )
        # A job killed mid-overlap leaves its replacement behind; never adopt it.
        _remove(new, execution=run)
        not_started = _start(request.service, new, execution=run)
        if not_started:
            _remove(new, execution=run)
            return ModelRuntimeStartFirstResult(
                **base,
                status="fail",
                failure="replacement-not-started",
                detail=f"{new} did not start ({not_started}); {old} left running",
            )
        started = self.clock()
        probes = 0
        while True:
            status = _ready_status(new, request.ready_url, execution=run)
            probes += 1
            waited = self.clock() - started
            if status == 200:
                break
            exited = not _running(new, execution=run)
            if exited or waited + request.poll_seconds > request.bound_seconds:
                _remove(new, execution=run)
                return ModelRuntimeStartFirstResult(
                    **base,
                    status="fail",
                    failure=(
                        "replacement-exited-before-ready"
                        if exited
                        else "replacement-never-ready"
                    ),
                    detail=(
                        f"{new} did not read /ready 200 within "
                        f"{request.bound_seconds:.0f}s ({probes} probe(s), last "
                        f"status {status}); removed, {old} left running"
                    ),
                    probes=probes,
                    waited_seconds=waited,
                )
            self.sleep(request.poll_seconds)
        old_running = _running(old, execution=run)
        run._run(["docker", "stop", old], timeout=COMPOSE_SECONDS)
        stopped = not _running(old, execution=run)
        return ModelRuntimeStartFirstResult(
            **base,
            status="pass" if stopped else "fail",
            failure="" if stopped else "old-runtime-not-stopped",
            detail=f"{new} read /ready 200 after {waited:.0f}s; {old} stopped={stopped}",
            probes=probes,
            waited_seconds=waited,
            old_running_at_first_ready=old_running,
        )
