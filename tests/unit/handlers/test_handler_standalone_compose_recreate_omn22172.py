# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The standalone compose recreate effect, against a fake docker (OMN-22172 AC1, AC2, AC4)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
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
from omnibase_infra.models.model_standalone_compose_recreate_target import (
    ModelStandaloneComposeRecreateTarget,
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
        carry_fails: bool = False,
        carry_changes: bool = True,
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
        self._carry_fails = carry_fails
        self._carry_changes = carry_changes
        self.events: list[str] = []
        self.carried: list[tuple[str, str]] = []

    def list_containers(self) -> dict[str, str]:
        self.census_calls += 1
        if self.census_calls == self._census_fails_on_call:
            raise InfraConnectionError("unreadable Docker container census")
        return dict(self.containers)

    def carry_file(self, git_ref: str, relative_path: str) -> bool:
        self.events.append("carry")
        self.carried.append((git_ref, relative_path))
        if self._carry_fails:
            raise InfraConnectionError("git show exited 128")
        return self._carry_changes

    def recreate_service(self, compose_file: str, env_file: str, service: str) -> None:
        self.events.append("recreate")
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


CARRIED = "docker/factory-floor-web/nginx.conf"


def _git(root: Path, *arguments: str) -> None:
    """Run git in ``root`` with an environment that cannot reach out of it (OMN-14891)."""
    env = scrub_git_location_env(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, env=env, capture_output=True
    )


def _source_clone(root: Path, content: bytes) -> Path:
    """A git clone whose tag ``deploy-ref`` holds ``CARRIED``; the working tree then moves on."""
    root.mkdir()
    _git(root, "init", "--quiet", "-b", "main")
    (root / CARRIED).parent.mkdir(parents=True)
    (root / CARRIED).write_bytes(content)
    _git(root, "add", CARRIED)
    _git(
        root,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "carried file",
    )
    _git(root, "tag", "deploy-ref")
    (root / CARRIED).write_bytes(b"working tree edit that is not at the ref\n")
    return root


def _real_docker(
    tmp_path: Path, script: str, *, timeout_seconds: float
) -> HandlerStandaloneComposeDocker:
    process = tmp_path / "docker_process.py"
    process.write_text(script)
    (tmp_path / "host").mkdir()
    config = ModelStandaloneComposeDockerConfig(
        repo_root=tmp_path / "host",
        source_root=_source_clone(tmp_path / "source", b"server { }\n"),
        command=(sys.executable, str(process)),
        environment={
            **scrub_git_location_env(os.environ),
            TARGET.env_file_variable: "/host/floor.env",
        },
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

    result = recreate_standalone_compose(docker, POLICY, _request(git_ref="deploy-ref"))

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.FAILED
    assert result.changed == ()
    assert result.containers_before == {TARGET.container_key: "a" * 64}
    assert result.containers_after == result.containers_before


def test_an_empty_census_stdout_reads_as_a_host_with_no_containers(
    tmp_path: Path,
) -> None:
    docker = _real_docker(tmp_path, "", timeout_seconds=5)

    assert docker.list_containers() == {}


_QUICK_COMPOSE = f"""\
import json, os, sys
print(json.dumps({{"Names": "{TARGET.container_key}", "ID": "{"a" * 64}"}}))
"""


def test_the_contract_carries_the_nginx_conf_the_compose_file_mounts() -> None:
    repo = Path(__file__).resolve().parents[3]
    compose_path = repo / TARGET.compose_file
    compose = yaml.safe_load(compose_path.read_text())
    mounted = [
        (compose_path.parent / volume["source"]).resolve().relative_to(repo).as_posix()
        for volume in compose["services"][TARGET.service]["volumes"]
        if isinstance(volume, dict) and volume.get("target") == "/etc/nginx/nginx.conf"
    ]

    assert mounted == [CARRIED]
    assert TARGET.carried_files == (CARRIED,)
    assert (repo / CARRIED).is_file()


def test_the_carried_file_reaches_the_host_before_compose_recreates_the_service() -> (
    None
):
    docker = FakeDocker()
    result = recreate_standalone_compose(docker, POLICY, _request(git_ref="deploy-ref"))

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.RECREATED
    assert docker.carried == [("deploy-ref", CARRIED)]
    assert docker.events == ["carry", "recreate"]
    assert result.carried_files == (CARRIED,)


def test_a_carried_file_already_on_the_host_is_not_reported_as_carried() -> None:
    docker = FakeDocker(carry_changes=False)
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.RECREATED
    assert docker.events == ["carry", "recreate"]
    assert result.carried_files == ()


def test_a_carry_failure_is_a_typed_failure_and_nothing_is_recreated() -> None:
    docker = FakeDocker(carry_fails=True)
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.FAILED
    assert CARRIED in result.message
    assert docker.events == ["carry"]
    assert docker.census_calls == 0
    assert docker.calls == []


def test_a_refused_request_carries_nothing() -> None:
    docker = FakeDocker(env_file=None)
    result = recreate_standalone_compose(docker, POLICY, _request())

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.REFUSED
    assert docker.carried == []


def test_the_carried_file_is_written_from_the_ref_not_from_the_working_tree(
    tmp_path: Path,
) -> None:
    docker = _real_docker(tmp_path, _QUICK_COMPOSE, timeout_seconds=10)
    stale = tmp_path / "host" / CARRIED
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale host copy\n")

    assert docker.carry_file("deploy-ref", CARRIED) is True

    assert stale.read_bytes() == b"server { }\n"
    assert stale.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in stale.parent.iterdir()] == ["nginx.conf"]
    assert docker.carry_file("deploy-ref", CARRIED) is False


def test_carrying_creates_a_file_the_host_does_not_have_yet(tmp_path: Path) -> None:
    docker = _real_docker(tmp_path, _QUICK_COMPOSE, timeout_seconds=10)

    assert docker.carry_file("deploy-ref", CARRIED) is True

    assert (tmp_path / "host" / CARRIED).read_bytes() == b"server { }\n"


def test_a_full_recreate_leaves_the_host_with_the_refs_nginx_conf(
    tmp_path: Path,
) -> None:
    docker = _real_docker(tmp_path, _QUICK_COMPOSE, timeout_seconds=10)
    host_file = tmp_path / "host" / CARRIED
    host_file.parent.mkdir(parents=True)
    host_file.write_bytes(b"stale host copy\n")

    result = recreate_standalone_compose(docker, POLICY, _request(git_ref="deploy-ref"))

    assert host_file.read_bytes() == b"server { }\n"
    assert result.carried_files == (CARRIED,)


def test_an_unknown_ref_fails_and_leaves_the_host_copy_untouched(
    tmp_path: Path,
) -> None:
    docker = _real_docker(tmp_path, _QUICK_COMPOSE, timeout_seconds=10)
    host_file = tmp_path / "host" / CARRIED
    host_file.parent.mkdir(parents=True)
    host_file.write_bytes(b"stale host copy\n")

    result = recreate_standalone_compose(
        docker, POLICY, _request(git_ref="no-such-ref")
    )

    assert result.outcome is EnumStandaloneComposeRecreateOutcome.FAILED
    assert host_file.read_bytes() == b"stale host copy\n"
    assert result.containers_before == {}


@pytest.mark.parametrize("git_ref", ["--output=/tmp/x", "-n"])
def test_a_ref_that_looks_like_an_option_is_refused_before_git_runs(
    tmp_path: Path, git_ref: str
) -> None:
    docker = _real_docker(tmp_path, _QUICK_COMPOSE, timeout_seconds=10)

    with pytest.raises(InfraConnectionError):
        docker.carry_file(git_ref, CARRIED)


@pytest.mark.parametrize(
    "path",
    ["", "/etc/nginx/nginx.conf", "../outside", "docker/../../outside", "-flag", "a:b"],
)
def test_a_target_may_only_carry_plain_repo_relative_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ModelStandaloneComposeRecreateTarget(
            compose_file=TARGET.compose_file,
            service=TARGET.service,
            container_key=TARGET.container_key,
            env_file_variable=TARGET.env_file_variable,
            carried_files=(path,),
        )
