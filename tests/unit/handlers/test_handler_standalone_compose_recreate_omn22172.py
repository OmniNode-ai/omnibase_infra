# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The standalone compose recreate effect, against a fake docker (OMN-22172 AC1, AC2, AC4)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.enums.enum_standalone_compose_recreate_outcome import (
    EnumStandaloneComposeRecreateOutcome,
)
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.handler_standalone_compose_recreate import (
    HandlerStandaloneComposeDocker,
    HandlerStandaloneComposeRecreate,
    recreate_standalone_compose,
    refusal,
    serves,
)
from omnibase_infra.models.model_standalone_compose_docker_config import (
    ModelStandaloneComposeDockerConfig,
)
from omnibase_infra.models.model_standalone_compose_recreate_policy import (
    ModelStandaloneComposeRecreatePolicy,
)
from omnibase_infra.models.model_standalone_compose_recreate_request import (
    ModelStandaloneComposeRecreateRequest,
)
from omnibase_infra.nodes import node_standalone_compose_recreate_effect

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
        removes: tuple[str, ...] = (),
        census_fails_on_call: int | None = None,
    ) -> None:
        self.containers = {**OTHERS, TARGET.container_key: "id-floor-0"}
        self._env_file = env_file
        self._also = also_changes
        self._fail = fail
        self._recreates = recreates
        self._removes = removes
        self._census_fails_on_call = census_fails_on_call
        self.census_calls = 0
        self.calls: list[tuple[str, str, str]] = []

    def list_containers(self) -> dict[str, str]:
        self.census_calls += 1
        if self.census_calls == self._census_fails_on_call:
            raise InfraConnectionError("unreadable Docker container census")
        return dict(self.containers)

    def recreate_service(self, compose_file: str, env_file: str, service: str) -> None:
        self.calls.append((compose_file, env_file, service))
        if self._fail:
            raise InfraConnectionError("compose exited 1")
        if self._recreates:
            self.containers[TARGET.container_key] = "id-floor-1"
        for name in self._also:
            self.containers[name] = "id-changed"
        for name in self._removes:
            del self.containers[name]

    def resolve_env_file(self, variable: str) -> str | None:
        assert variable == TARGET.env_file_variable
        return self._env_file


class FakeContainer:
    def __init__(self, docker: FakeDocker) -> None:
        self._services: dict[type, object] = {
            HandlerStandaloneComposeDocker: docker,
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


def _contract_with(tmp_path: Path, **overrides: object) -> Path:
    """A copy of the node contract with the named top-level keys replaced."""
    contract = yaml.safe_load(CONTRACT.read_text())
    contract.update(overrides)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract))
    return path


def _declared_targets() -> list[dict[str, str]]:
    return list(yaml.safe_load(CONTRACT.read_text())["standalone_targets"])


def _second_target() -> dict[str, str]:
    return {
        "compose_file": "docker/docker-compose.second.yml",
        "service": "second-service",
        "container_key": "omninode-second-service",
        "env_file_variable": TARGET.env_file_variable,
    }


def test_contract_lanes_decide_which_lane_the_handler_admits(tmp_path: Path) -> None:
    prod = _request(runtime_lane="prod")
    assert refusal(FakeDocker(), POLICY, prod) is not None

    widened = ModelStandaloneComposeRecreatePolicy.from_contract(
        _contract_with(tmp_path, allowed_lanes=["dev", "prod"])
    )
    assert refusal(FakeDocker(), widened, prod) is None
    assert (
        recreate_standalone_compose(FakeDocker(), widened, prod).outcome
        is EnumStandaloneComposeRecreateOutcome.RECREATED
    )

    narrowed = ModelStandaloneComposeRecreatePolicy.from_contract(
        _contract_with(tmp_path, allowed_lanes=["canary"])
    )
    assert refusal(FakeDocker(), narrowed, _request()) is not None


def test_contract_targets_decide_which_service_the_handler_recreates(
    tmp_path: Path,
) -> None:
    second = _second_target()
    request = _request(compose_file=second["compose_file"], service=second["service"])
    assert refusal(FakeDocker(), POLICY, request) is not None

    widened = ModelStandaloneComposeRecreatePolicy.from_contract(
        _contract_with(tmp_path, standalone_targets=[*_declared_targets(), second])
    )
    assert refusal(FakeDocker(), widened, request) is None

    emptied = ModelStandaloneComposeRecreatePolicy.from_contract(
        _contract_with(tmp_path, standalone_targets=[])
    )
    assert refusal(FakeDocker(), emptied, _request()) is not None


def test_contract_container_key_is_the_one_container_the_handler_lets_move(
    tmp_path: Path,
) -> None:
    renamed = ModelStandaloneComposeRecreatePolicy.from_contract(
        _contract_with(
            tmp_path,
            standalone_targets=[
                {**_declared_targets()[0], "container_key": "omninode-runtime"}
            ],
        )
    )
    result = recreate_standalone_compose(FakeDocker(), renamed, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert TARGET.container_key in result.message


@pytest.mark.parametrize(
    "missing", ["docker_timeout_seconds", "allowed_lanes", "standalone_targets"]
)
def test_contract_missing_a_policy_key_fails_loudly(
    tmp_path: Path, missing: str
) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    del contract[missing]
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract))

    with pytest.raises(KeyError, match=missing):
        ModelStandaloneComposeRecreatePolicy.from_contract(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"docker_timeout_seconds": 0},
        {"docker_timeout_seconds": "soon"},
        {"standalone_targets": [{"compose_file": "docker/x.yml"}]},
        {"standalone_targets": [{**_second_target(), "service": ""}]},
        {"standalone_targets": [{**_second_target(), "unexpected": "field"}]},
    ],
)
def test_contract_with_a_malformed_policy_value_is_rejected(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        ModelStandaloneComposeRecreatePolicy.from_contract(
            _contract_with(tmp_path, **overrides)
        )


def test_a_census_failure_before_the_recreate_propagates_and_nothing_is_recreated() -> (
    None
):
    docker = FakeDocker(census_fails_on_call=1)

    with pytest.raises(InfraConnectionError, match="census"):
        recreate_standalone_compose(docker, POLICY, _request())
    assert docker.calls == []


def test_a_census_failure_after_the_recreate_propagates_and_is_not_a_recreated_result() -> (
    None
):
    docker = FakeDocker(census_fails_on_call=2)

    with pytest.raises(InfraConnectionError, match="census"):
        recreate_standalone_compose(docker, POLICY, _request())
    assert len(docker.calls) == 1


def test_a_census_failure_after_a_compose_failure_propagates() -> None:
    docker = FakeDocker(fail=True, census_fails_on_call=2)

    with pytest.raises(InfraConnectionError, match="census"):
        recreate_standalone_compose(docker, POLICY, _request())


def test_a_service_container_missing_after_the_recreate_is_a_scope_violation() -> None:
    docker = FakeDocker(recreates=False, removes=(TARGET.container_key,))
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert result.changed == (TARGET.container_key,)
    assert TARGET.container_key not in result.containers_after
    assert result.service_container_after == ""


def test_another_container_removed_by_the_recreate_is_a_scope_violation() -> None:
    docker = FakeDocker(removes=("omnibase-infra-postgres",))
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert "omnibase-infra-postgres" in result.message
    assert "omnibase-infra-postgres" not in result.containers_after


def test_a_stopped_foreign_container_appearing_is_a_scope_violation() -> None:
    docker = FakeDocker(also_changes=("stopped-foreign",))
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert "stopped-foreign" in result.message


def test_a_stopped_foreign_container_being_recreated_is_a_scope_violation() -> None:
    docker = FakeDocker(also_changes=("stopped-foreign",))
    docker.containers["stopped-foreign"] = "id-stopped-0"
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
    assert result.containers_before["stopped-foreign"] == "id-stopped-0"
    assert result.containers_after["stopped-foreign"] == "id-changed"


def test_a_stopped_foreign_container_left_alone_does_not_block_the_recreate() -> None:
    docker = FakeDocker()
    docker.containers["stopped-foreign"] = "id-stopped-0"
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.RECREATED
    assert result.containers_after["stopped-foreign"] == "id-stopped-0"


class FakeConfigContainer:
    def __init__(self, config: ModelStandaloneComposeDockerConfig) -> None:
        self._config = config

    def get_service(self, interface: type) -> object:
        assert interface is ModelStandaloneComposeDockerConfig
        return self._config


def _real_docker(
    tmp_path: Path, script: str, *, timeout_seconds: float
) -> HandlerStandaloneComposeDocker:
    process = tmp_path / "docker_process.py"
    process.write_text(script)
    config = ModelStandaloneComposeDockerConfig(
        repo_root=tmp_path,
        command=(sys.executable, str(process)),
        environment={TARGET.env_file_variable: "/host/floor.env"},
        timeout_seconds=timeout_seconds,
    )
    return HandlerStandaloneComposeDocker(
        cast("ModelONEXContainer", FakeConfigContainer(config))
    )


_SLOW_COMPOSE = f"""\
import json, sys, time
if sys.argv[1] == "compose":
    time.sleep(30)
print(json.dumps({{"Names": "{TARGET.container_key}", "ID": "{"a" * 64}"}}))
"""


def test_a_timeout_during_compose_up_is_a_typed_failure_with_nothing_changed(
    tmp_path: Path,
) -> None:
    docker = _real_docker(tmp_path, _SLOW_COMPOSE, timeout_seconds=1.0)

    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.FAILED
    assert result.changed == ()
    assert result.containers_before == {TARGET.container_key: "a" * 64}
    assert result.containers_after == result.containers_before


def test_an_empty_census_stdout_reads_as_a_host_with_no_containers(
    tmp_path: Path,
) -> None:
    docker = _real_docker(tmp_path, "", timeout_seconds=5)

    assert docker.list_containers() == {}
