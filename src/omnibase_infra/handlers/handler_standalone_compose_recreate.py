# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recreate one service of one declared standalone compose file, and nothing else.

The deploy agent runs this on the host that serves the compose file. The node contract declares the
lanes, compose files and services the route may touch; a request outside them is refused before
docker is called. Before the recreate, each file the target declares as carried is written into the
host's tree from the requested git ref, so a bind-mounted config no merge refreshes cannot go stale.
The recreate runs with no dependencies. The host's containers are read before and after, and the
result is a scope violation unless exactly the one declared container id changed.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.enums.enum_standalone_compose_recreate_outcome import (
    EnumStandaloneComposeRecreateOutcome,
)
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.models.model_standalone_compose_docker_config import (
    ModelStandaloneComposeDockerConfig,
)
from omnibase_infra.models.model_standalone_compose_recreate_policy import (
    ModelStandaloneComposeRecreatePolicy,
)
from omnibase_infra.models.model_standalone_compose_recreate_request import (
    ModelStandaloneComposeRecreateRequest,
)
from omnibase_infra.models.model_standalone_compose_recreate_result import (
    ModelStandaloneComposeRecreateResult,
)
from omnibase_infra.models.model_standalone_compose_recreate_target import (
    ModelStandaloneComposeRecreateTarget,
)
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,
    sanitize_error_string,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


def _declared(
    policy: ModelStandaloneComposeRecreatePolicy,
    request: ModelStandaloneComposeRecreateRequest,
) -> ModelStandaloneComposeRecreateTarget | None:
    for target in policy.targets:
        if (
            target.compose_file == request.compose_file
            and target.service == request.service
        ):
            return target
    return None


def _resolve(
    docker: HandlerStandaloneComposeDocker,
    policy: ModelStandaloneComposeRecreatePolicy,
    request: ModelStandaloneComposeRecreateRequest,
) -> tuple[ModelStandaloneComposeRecreateTarget | None, str | None, str | None]:
    """The declared target, this host's env file and the refusal reason (set when refused)."""
    if request.runtime_lane not in policy.allowed_lanes:
        return None, None, f"runtime lane {request.runtime_lane!r} is not served"
    if request.compose_file not in {t.compose_file for t in policy.targets}:
        return None, None, f"compose file {request.compose_file!r} is not declared"
    target = _declared(policy, request)
    if target is None:
        return (
            None,
            None,
            f"service {request.service!r} is not declared for {request.compose_file!r}",
        )
    env_file = docker.resolve_env_file(target.env_file_variable)
    if env_file is None:
        return None, None, f"this host does not declare {target.env_file_variable}"
    return target, env_file, None


def _changed(before: dict[str, str], after: dict[str, str]) -> tuple[str, ...]:
    names = set(before) | set(after)
    return tuple(sorted(n for n in names if before.get(n) != after.get(n)))


def refusal(
    docker: HandlerStandaloneComposeDocker,
    policy: ModelStandaloneComposeRecreatePolicy,
    request: ModelStandaloneComposeRecreateRequest,
) -> str | None:
    """Why the request must not run on this host, or None when it may."""
    return _resolve(docker, policy, request)[2]


def serves(
    docker: HandlerStandaloneComposeDocker,
    policy: ModelStandaloneComposeRecreatePolicy,
    request: ModelStandaloneComposeRecreateRequest,
) -> bool:
    """False when the target is declared but this host does not declare its env file variable.

    An undeclared target is served by no host, so every host answers it with a refusal.
    """
    target = _declared(policy, request)
    return (
        target is None or docker.resolve_env_file(target.env_file_variable) is not None
    )


def recreate_standalone_compose(
    docker: HandlerStandaloneComposeDocker,
    policy: ModelStandaloneComposeRecreatePolicy,
    request: ModelStandaloneComposeRecreateRequest,
) -> ModelStandaloneComposeRecreateResult:
    """Refuse anything undeclared; recreate one service; prove nothing else moved."""
    target, env_file, reason = _resolve(docker, policy, request)
    if target is None or env_file is None:
        return ModelStandaloneComposeRecreateResult(
            correlation_id=request.correlation_id,
            outcome=EnumStandaloneComposeRecreateOutcome.REFUSED,
            message=reason or "request refused",
        )

    carried: list[str] = []
    try:
        for path in target.carried_files:
            if docker.carry_file(request.git_ref, path):
                carried.append(path)
    except InfraConnectionError as exc:
        return ModelStandaloneComposeRecreateResult(
            correlation_id=request.correlation_id,
            outcome=EnumStandaloneComposeRecreateOutcome.FAILED,
            message=f"carrying {path!r} failed: {sanitize_error_message(exc)}",
            carried_files=tuple(carried),
        )

    before = docker.list_containers()
    try:
        docker.recreate_service(request.compose_file, env_file, request.service)
    except InfraConnectionError as exc:
        after_failed = docker.list_containers()
        return ModelStandaloneComposeRecreateResult(
            correlation_id=request.correlation_id,
            outcome=EnumStandaloneComposeRecreateOutcome.FAILED,
            message=sanitize_error_message(exc),
            containers_before=before,
            containers_after=after_failed,
            changed=_changed(before, after_failed),
            carried_files=tuple(carried),
        )
    after = docker.list_containers()
    changed = _changed(before, after)
    key = target.container_key
    violation = [name for name in changed if name != key]
    if violation or key not in changed or key not in after:
        return ModelStandaloneComposeRecreateResult(
            correlation_id=request.correlation_id,
            outcome=EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION,
            message=(
                f"container ids other than the service's changed: {sorted(violation)}"
                if violation
                else f"container {key!r} was not recreated"
            ),
            containers_before=before,
            containers_after=after,
            changed=changed,
            carried_files=tuple(carried),
        )
    return ModelStandaloneComposeRecreateResult(
        correlation_id=request.correlation_id,
        outcome=EnumStandaloneComposeRecreateOutcome.RECREATED,
        message=f"recreated {request.service!r}; no other container id changed",
        service_container_before=before.get(key, ""),
        service_container_after=after.get(key, ""),
        containers_before=before,
        containers_after=after,
        changed=changed,
        carried_files=tuple(carried),
    )


class HandlerStandaloneComposeRecreate:
    """Recreate one declared standalone compose service, resolving docker and policy from the container."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelStandaloneComposeRecreateRequest
    ) -> ModelStandaloneComposeRecreateResult:
        docker = self._container.get_service(HandlerStandaloneComposeDocker)
        policy = self._container.get_service(ModelStandaloneComposeRecreatePolicy)
        return recreate_standalone_compose(docker, policy, request)


class HandlerStandaloneComposeDocker:
    """Read the whole host and force-recreate exactly one compose service.

    Admission remains in the handler, using the node contract's policy. Commands
    use an argument vector, an explicit working directory, and a bounded timeout.
    The executable prefix is injectable for process-level integration tests.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        self._config = container.get_service(ModelStandaloneComposeDockerConfig)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _error(self, message: str, operation: str) -> InfraConnectionError:
        return InfraConnectionError(
            message,
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation=operation,
            ),
        )

    def _run(self, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                [*self._config.command, *arguments],
                cwd=self._config.repo_root,
                env=self._config.environment,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise self._error(sanitize_error_message(exc), arguments[0]) from exc
        if completed.returncode:
            raise self._error(
                f"docker {arguments[0]} exited {completed.returncode}: "
                f"{sanitize_error_string(completed.stderr)}",
                arguments[0],
            )
        return completed.stdout

    def list_containers(self) -> dict[str, str]:
        """All containers, including stopped containers, with full IDs."""
        output = self._run("ps", "--all", "--no-trunc", "--format", "{{json .}}")
        containers: dict[str, str] = {}
        try:
            for line in output.splitlines():
                row = json.loads(line)
                name, container_id = row["Names"], row["ID"]
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(container_id, str)
                    or not container_id
                    or name in containers
                ):
                    raise ValueError("empty, duplicate or invalid Docker census row")
                containers[name] = container_id
        except (ValueError, KeyError, TypeError) as exc:
            raise self._error("unreadable Docker container census", "ps") from exc
        return containers

    def carry_file(self, git_ref: str, relative_path: str) -> bool:
        """Write ``relative_path`` as the source clone holds it at ``git_ref`` into the host's tree.

        The write is a temporary file renamed into place, so a reader never sees a partial file,
        and a file already holding those bytes is left alone. Returns whether the bytes changed.
        """
        if git_ref.startswith("-"):
            raise self._error("git ref may not start with '-'", "show")
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self._config.source_root),
                    "show",
                    f"{git_ref}:{relative_path}",
                ],
                env=scrub_git_location_env(self._config.environment),
                capture_output=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise self._error(sanitize_error_message(exc), "show") from exc
        if completed.returncode:
            raise self._error(
                f"git show exited {completed.returncode}: "
                f"{sanitize_error_string(completed.stderr.decode(errors='replace'))}",
                "show",
            )
        destination = self._config.repo_root / relative_path
        if destination.is_file() and destination.read_bytes() == completed.stdout:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.carry")
        try:
            temporary.write_bytes(completed.stdout)
            temporary.chmod(0o644)
            temporary.replace(destination)
        except OSError as exc:
            raise self._error(sanitize_error_message(exc), "write") from exc
        return True

    def recreate_service(self, compose_file: str, env_file: str, service: str) -> None:
        """Force-recreate one admitted service without starting its dependencies."""
        self._run(
            "compose",
            "--file",
            compose_file,
            "--env-file",
            env_file,
            "up",
            "--detach",
            "--no-deps",
            "--force-recreate",
            "--",
            service,
        )

    def resolve_env_file(self, variable: str) -> str | None:
        """Return the host's declared env file; empty declarations are absent."""
        return self._config.environment.get(variable) or None
