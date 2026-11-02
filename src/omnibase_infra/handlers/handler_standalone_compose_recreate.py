# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recreate one service of one declared standalone compose file, and nothing else.

The deploy agent runs this on the host that serves the compose file. The node contract declares the
lanes, compose files and services the route may touch; a request outside them is refused before
docker is called. The recreate runs with no dependencies. The host's containers are read before
and after, and the result is a scope violation unless exactly the one declared container id changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_standalone_compose_recreate_outcome import (
    EnumStandaloneComposeRecreateOutcome,
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
from omnibase_infra.protocols.protocol_standalone_compose_docker import (
    ProtocolStandaloneComposeDocker,
    StandaloneComposeDockerError,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

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
    docker: ProtocolStandaloneComposeDocker,
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
    docker: ProtocolStandaloneComposeDocker,
    policy: ModelStandaloneComposeRecreatePolicy,
    request: ModelStandaloneComposeRecreateRequest,
) -> str | None:
    """Why the request must not run on this host, or None when it may."""
    return _resolve(docker, policy, request)[2]


def serves(
    docker: ProtocolStandaloneComposeDocker,
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
    docker: ProtocolStandaloneComposeDocker,
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

    before = docker.list_containers()
    try:
        docker.recreate_service(request.compose_file, env_file, request.service)
    except StandaloneComposeDockerError as exc:
        after_failed = docker.list_containers()
        return ModelStandaloneComposeRecreateResult(
            correlation_id=request.correlation_id,
            outcome=EnumStandaloneComposeRecreateOutcome.FAILED,
            message=sanitize_error_message(exc),
            containers_before=before,
            containers_after=after_failed,
            changed=_changed(before, after_failed),
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
        docker_interface: type = ProtocolStandaloneComposeDocker
        docker = cast(
            "ProtocolStandaloneComposeDocker",
            self._container.get_service(docker_interface),
        )
        policy = self._container.get_service(ModelStandaloneComposeRecreatePolicy)
        return recreate_standalone_compose(docker, policy, request)
