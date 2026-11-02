# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler-to-process readback for standalone recreation (OMN-22172).

The Docker subprocess is a stateful simulator; no Docker daemon or lane is touched.
The real Docker handler, contract, service registry and recreate handler execute unchanged.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums.enum_standalone_compose_recreate_outcome import (
    EnumStandaloneComposeRecreateOutcome,
)
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.handler_standalone_compose_recreate import (
    HandlerStandaloneComposeDocker,
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

pytestmark = pytest.mark.integration


async def _docker(
    repo_root: Path,
    *,
    command: tuple[str, ...],
    environment: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> HandlerStandaloneComposeDocker:
    contract_path = (
        Path(node_standalone_compose_recreate_effect.__file__).parent / "contract.yaml"
    )
    policy = ModelStandaloneComposeRecreatePolicy.from_contract(contract_path)
    config = ModelStandaloneComposeDockerConfig(
        repo_root=repo_root,
        command=command,
        environment=dict(os.environ if environment is None else environment),
        timeout_seconds=policy.docker_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds,
    )
    container = ModelONEXContainer()
    assert container.service_registry is not None
    await container.service_registry.register_instance(
        ModelStandaloneComposeDockerConfig, config
    )
    return HandlerStandaloneComposeDocker(container)


_DOCKER_PROCESS = """\
import json
import os
import sys
from pathlib import Path
state_path = Path(os.environ["SIMULATOR_STATE"])
state = json.loads(state_path.read_text())
args = sys.argv[1:]
state["calls"].append(args)
if args[0] == "ps":
    assert args == ["ps", "--all", "--no-trunc", "--format", "{{json .}}"]
    for name, identifier in state["containers"].items():
        print(json.dumps({"Names": name, "ID": identifier}))
elif args[0] == "compose":
    assert args == state["expected_recreate"]
    state["containers"][state["target"]] = "b" * 64
    if state["move_foreign"]:
        state["containers"]["unrelated-stopped"] = "d" * 64
else:
    raise AssertionError(args)
state_path.write_text(json.dumps(state))
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("move_foreign", [False, True])
async def test_contract_handler_process_and_full_host_readback(
    tmp_path: Path, move_foreign: bool
) -> None:
    contract_path = (
        Path(node_standalone_compose_recreate_effect.__file__).parent / "contract.yaml"
    )
    policy = ModelStandaloneComposeRecreatePolicy.from_contract(contract_path)
    target = policy.targets[0]
    state_path = tmp_path / "docker-state.json"
    expected_recreate = [
        "compose",
        "--file",
        target.compose_file,
        "--env-file",
        "/host/floor.env",
        "up",
        "--detach",
        "--no-deps",
        "--force-recreate",
        "--",
        target.service,
    ]
    state_path.write_text(
        json.dumps(
            {
                "calls": [],
                "containers": {
                    target.container_key: "a" * 64,
                    "unrelated-stopped": "c" * 64,
                },
                "target": target.container_key,
                "expected_recreate": expected_recreate,
                "move_foreign": move_foreign,
            }
        )
    )
    process = tmp_path / "docker_process.py"
    process.write_text(_DOCKER_PROCESS)
    adapter = await _docker(
        tmp_path,
        environment={
            **os.environ,
            target.env_file_variable: "/host/floor.env",
            "SIMULATOR_STATE": str(state_path),
        },
        command=(sys.executable, str(process)),
    )
    container = ModelONEXContainer()
    assert container.service_registry is not None
    await container.service_registry.register_instance(
        HandlerStandaloneComposeDocker, adapter
    )
    await container.service_registry.register_instance(
        ModelStandaloneComposeRecreatePolicy, policy
    )
    contract = yaml.safe_load(contract_path.read_text())
    route = contract["handler_routing"]["handlers"][0]["handler"]
    handler_type = getattr(importlib.import_module(route["module"]), route["name"])
    handler = handler_type(container)
    request = ModelStandaloneComposeRecreateRequest(
        correlation_id=uuid4(),
        requested_by="integration-test",
        runtime_lane="dev",
        compose_file=target.compose_file,
        service=target.service,
        git_ref="origin/dev",
    )

    # Admission rejects the request before any subprocess, even a read-only ps.
    refused = handler.handle(request.model_copy(update={"runtime_lane": "prod"}))
    assert refused.outcome is EnumStandaloneComposeRecreateOutcome.REFUSED
    assert json.loads(state_path.read_text())["calls"] == []

    result = handler.handle(request)
    state = json.loads(state_path.read_text())
    assert [call[0] for call in state["calls"]] == ["ps", "compose", "ps"]
    assert state["calls"][1] == expected_recreate
    assert result.correlation_id == request.correlation_id
    assert result.containers_before[target.container_key] == "a" * 64
    assert result.containers_after[target.container_key] == "b" * 64
    if move_foreign:
        assert result.outcome is EnumStandaloneComposeRecreateOutcome.SCOPE_VIOLATION
        assert "unrelated-stopped" in result.changed
    else:
        assert result.outcome is EnumStandaloneComposeRecreateOutcome.RECREATED
        assert result.changed == (target.container_key,)
        assert (
            result.containers_before["unrelated-stopped"]
            == result.containers_after["unrelated-stopped"]
        )


@pytest.mark.parametrize(
    "output",
    [
        "not-json",
        "[]",
        '{"Names": "target"}',
        '{"Names": "target", "ID": ""}',
        '{"Names": "target", "ID": "a"}\n{"Names": "target", "ID": "b"}',
    ],
)
@pytest.mark.asyncio
async def test_adapter_refuses_an_unreadable_or_ambiguous_host_census(
    tmp_path: Path, output: str
) -> None:
    adapter = await _docker(
        tmp_path, command=(sys.executable, "-c", f"print({output!r})")
    )
    with pytest.raises(InfraConnectionError, match="unreadable Docker"):
        adapter.list_containers()


@pytest.mark.asyncio
async def test_compose_process_failure_is_a_typed_failure_with_after_readback(
    tmp_path: Path,
) -> None:
    from omnibase_infra.handlers.handler_standalone_compose_recreate import (
        recreate_standalone_compose,
    )

    contract_path = (
        Path(node_standalone_compose_recreate_effect.__file__).parent / "contract.yaml"
    )
    policy = ModelStandaloneComposeRecreatePolicy.from_contract(contract_path)
    target = policy.targets[0]
    process = tmp_path / "docker_failure.py"
    calls_path = tmp_path / "calls.jsonl"
    process.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"with Path({str(calls_path)!r}).open('a') as calls:\n"
        "    calls.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'compose':\n"
        "    print('compose refused', file=sys.stderr)\n"
        "    sys.exit(7)\n"
        f"print({json.dumps({'Names': target.container_key, 'ID': 'a' * 64})!r})\n"
    )
    adapter = await _docker(
        tmp_path,
        command=(sys.executable, str(process)),
        environment={**os.environ, target.env_file_variable: "/host/floor.env"},
    )
    request = ModelStandaloneComposeRecreateRequest(
        correlation_id=uuid4(),
        requested_by="integration-test",
        runtime_lane="dev",
        compose_file=target.compose_file,
        service=target.service,
        git_ref="origin/dev",
    )
    result = recreate_standalone_compose(adapter, policy, request)
    assert result.outcome is EnumStandaloneComposeRecreateOutcome.FAILED
    assert "exited 7" in result.message
    assert result.containers_after == result.containers_before
    assert [json.loads(line)[0] for line in calls_path.read_text().splitlines()] == [
        "ps",
        "compose",
        "ps",
    ]


@pytest.mark.asyncio
async def test_adapter_bounds_the_docker_process_lifetime(tmp_path: Path) -> None:
    adapter = await _docker(
        tmp_path,
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
        timeout_seconds=0.1,
    )
    with pytest.raises(InfraConnectionError):
        adapter.list_containers()
