# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The standalone compose recreate effect, against a fake docker (OMN-22172 AC1, AC2, AC4)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_standalone_compose_recreate_outcome import (
    EnumStandaloneComposeRecreateOutcome,
)
from omnibase_infra.handlers.handler_standalone_compose_recreate import (
    HandlerStandaloneComposeRecreate,
    recreate_standalone_compose,
    refusal,
    serves,
)
from omnibase_infra.models.model_standalone_compose_recreate_policy import (
    ModelStandaloneComposeRecreatePolicy,
)
from omnibase_infra.models.model_standalone_compose_recreate_request import (
    ModelStandaloneComposeRecreateRequest,
)
from omnibase_infra.nodes import node_standalone_compose_recreate_effect
from omnibase_infra.protocols.protocol_standalone_compose_docker import (
    ProtocolStandaloneComposeDocker,
    StandaloneComposeDockerError,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

pytestmark = pytest.mark.unit

CONTRACT = (
    Path(node_standalone_compose_recreate_effect.__file__).parent / "contract.yaml"
)
POLICY = ModelStandaloneComposeRecreatePolicy.from_contract(CONTRACT)
TARGET = POLICY.targets[0]
OTHERS = {
    "omninode-runtime": "id-runtime-0",
    "omnibase-infra-postgres": "id-postgres-0",
    "omnibase-infra-redpanda": "id-redpanda-0",
}


class FakeDocker:
    """A host's containers. ``recreate_service`` swaps the one service's id, plus whatever
    ``also_changes`` names, so a scope violation can be staged."""

    def __init__(
        self,
        *,
        env_file: str | None = "/host/floor.env",
        also_changes: tuple[str, ...] = (),
        fail: bool = False,
        recreates: bool = True,
    ) -> None:
        self.containers = {**OTHERS, TARGET.container_key: "id-floor-0"}
        self._env_file = env_file
        self._also = also_changes
        self._fail = fail
        self._recreates = recreates
        self.calls: list[tuple[str, str, str]] = []

    def list_containers(self) -> dict[str, str]:
        return dict(self.containers)

    def recreate_service(self, compose_file: str, env_file: str, service: str) -> None:
        self.calls.append((compose_file, env_file, service))
        if self._fail:
            raise StandaloneComposeDockerError("compose exited 1")
        if self._recreates:
            self.containers[TARGET.container_key] = "id-floor-1"
        for name in self._also:
            self.containers[name] = "id-changed"

    def resolve_env_file(self, variable: str) -> str | None:
        assert variable == TARGET.env_file_variable
        return self._env_file


class FakeContainer:
    def __init__(self, docker: FakeDocker) -> None:
        self._services: dict[type, object] = {
            ProtocolStandaloneComposeDocker: docker,
            ModelStandaloneComposeRecreatePolicy: POLICY,
        }

    def get_service(self, interface: type) -> object:
        return self._services[interface]


def _request(**overrides: str) -> ModelStandaloneComposeRecreateRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "requested_by": "unit-test",
        "runtime_lane": "dev",
        "compose_file": TARGET.compose_file,
        "service": TARGET.service,
        "git_ref": "origin/dev",
    }
    fields.update(overrides)
    return ModelStandaloneComposeRecreateRequest.model_validate(fields)


def test_recreate_changes_only_the_declared_services_container_id() -> None:
    docker = FakeDocker()
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.RECREATED
    assert docker.calls == [(TARGET.compose_file, "/host/floor.env", TARGET.service)]
    assert result.changed == (TARGET.container_key,)
    assert result.service_container_before == "id-floor-0"
    assert result.service_container_after == "id-floor-1"
    for name, container_id in OTHERS.items():
        assert result.containers_before[name] == container_id
        assert result.containers_after[name] == container_id


def test_another_containers_id_changing_is_a_scope_violation() -> None:
    docker = FakeDocker(also_changes=("omninode-runtime",))
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert "omninode-runtime" in result.message


def test_a_recreate_that_changed_nothing_is_not_reported_as_recreated() -> None:
    docker = FakeDocker(recreates=False)
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert result.changed == ()


def test_a_compose_failure_is_a_typed_failure_with_the_ids_read_after() -> None:
    docker = FakeDocker(fail=True)
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.FAILED
    assert result.containers_after == result.containers_before


@pytest.mark.parametrize(
    "overrides",
    [
        {"compose_file": "docker/docker-compose.other.yml"},
        {"service": "some-other-service"},
        {"runtime_lane": "prod"},
        {"runtime_lane": "canary"},
    ],
)
def test_undeclared_target_or_lane_is_refused_and_docker_is_not_called(
    overrides: dict[str, str],
) -> None:
    docker = FakeDocker()
    result = recreate_standalone_compose(docker, POLICY, _request(**overrides))

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.REFUSED
    assert refusal(docker, POLICY, _request(**overrides)) is not None
    assert docker.calls == []


def test_a_host_without_the_env_file_variable_does_not_serve_the_target() -> None:
    docker = FakeDocker(env_file=None)

    assert serves(docker, POLICY, _request()) is False
    result = recreate_standalone_compose(docker, POLICY, _request())
    assert result.outcome is EnumStandaloneComposeRecreateOutcome.REFUSED
    assert docker.calls == []


def test_an_undeclared_target_is_answered_by_every_host() -> None:
    docker = FakeDocker(env_file=None)
    assert serves(docker, POLICY, _request(service="some-other-service")) is True


def test_the_handler_resolves_docker_and_policy_from_the_container() -> None:
    docker = FakeDocker()
    handler = HandlerStandaloneComposeRecreate(
        cast("ModelONEXContainer", FakeContainer(docker))
    )

    result = handler.handle(_request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.RECREATED
    assert len(docker.calls) == 1


def test_contract_declares_dev_only_and_one_target() -> None:
    assert POLICY.allowed_lanes == frozenset({"dev"})
    assert len(POLICY.targets) == 1
    assert CONTRACT.is_file()
