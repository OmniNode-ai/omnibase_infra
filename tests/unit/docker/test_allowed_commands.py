# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command plants go through T1's stack-shape gate, including known service names."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.unit.validation.test_checker_stack_shape import (
    _DATA,
    _INFRA,
    _OVERLAY,
    _edit,
    _judge,
    _stack,
    _text,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("field", ["command", "entrypoint"])
@pytest.mark.parametrize(
    "command", ["python -m planted.sidecar", ["python", "-m", "planted.sidecar"]]
)
def test_allowed_commands_refuses_a_known_runtime_service(
    tmp_path: Path, field: str, command: str | list[str]
) -> None:
    root = _stack(tmp_path)
    path = root / _OVERLAY
    compose = yaml.safe_load(path.read_text())
    compose["services"]["omninode-runtime"][field] = command
    path.write_text(yaml.safe_dump(compose))
    refused = _text(_judge(root))
    assert "omninode-runtime" in refused
    assert "python" in refused and "planted.sidecar" in refused


def test_allowed_commands_admits_the_declared_runtime_command(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    path = root / _OVERLAY
    compose = yaml.safe_load(path.read_text())
    compose["services"]["omninode-runtime"]["command"] = ["onex-runtime"]
    path.write_text(yaml.safe_dump(compose))
    assert _judge(root) == ()


def test_allowed_commands_cannot_hide_behind_the_service_baseline(
    tmp_path: Path,
) -> None:
    root = _stack(tmp_path)
    path = root / _INFRA
    compose = yaml.safe_load(path.read_text())
    compose["services"]["legacy-consumer"]["command"] = [
        "python",
        "-m",
        "planted.sidecar",
    ]
    path.write_text(yaml.safe_dump(compose))
    refused = _text(_judge(root))
    assert "legacy-consumer" in refused and "planted.sidecar" in refused


def test_allowed_commands_does_not_admit_a_shell_wrapper(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    path = root / _OVERLAY
    compose = yaml.safe_load(path.read_text())
    compose["services"]["omninode-runtime"]["command"] = [
        "sh",
        "-c",
        "onex-runtime; python -m planted.sidecar",
    ]
    path.write_text(yaml.safe_dump(compose))
    assert "planted.sidecar" in _text(_judge(root))


def test_allowed_commands_policy_is_required(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    data = yaml.safe_load((root / _DATA).read_text())
    data.pop("allowed_commands", None)
    (root / _DATA).write_text(yaml.safe_dump(data))
    assert "allowed_commands" in _text(_judge(root))


@pytest.mark.parametrize("hashed_scope", [False, True])
def test_allowed_commands_dependency_permission_is_service_scoped(
    tmp_path: Path,
    hashed_scope: bool,
) -> None:
    import hashlib

    root = _stack(tmp_path)
    data = yaml.safe_load((root / _DATA).read_text())
    data["allowed_commands"].append(
        {
            "argv": ["postgres"],
            "match": "executable",
            "field": "command",
            "services": [
                "sha256:" + hashlib.sha256(b"postgres").hexdigest()
                if hashed_scope
                else "postgres"
            ],
            "reason": "Database dependency",
        }
    )
    (root / _DATA).write_text(yaml.safe_dump(data))
    compose = yaml.safe_load((root / _OVERLAY).read_text())
    compose["services"]["omninode-runtime"]["command"] = ["postgres"]
    (root / _OVERLAY).write_text(yaml.safe_dump(compose))
    assert "omninode-runtime" in _text(_judge(root))


def _command_baseline_fixture(tmp_path: Path) -> Path:
    root = _stack(tmp_path)
    data = yaml.safe_load((root / _DATA).read_text())
    data["allowed_commands"] = [
        r for r in data["allowed_commands"] if r["services"] != ["legacy-consumer"]
    ]
    data["command_baseline"] = [
        {
            "service": "legacy-consumer",
            "field": "command",
            "argv": ["python", "-m", "example.legacy_consumer"],
            "files": 1,
        }
    ]
    (root / _DATA).write_text(yaml.safe_dump(data))
    return root


def test_allowed_commands_baseline_admits_only_the_recorded_command(
    tmp_path: Path,
) -> None:
    root = _command_baseline_fixture(tmp_path)
    assert _judge(root) == ()
    _edit(root / _INFRA, "example.legacy_consumer", "example.new_consumer")
    refused = _text(_judge(root))
    assert "legacy-consumer" in refused and "example.new_consumer" in refused


def test_allowed_commands_baseline_cannot_grow_or_change_against_base(
    tmp_path: Path,
) -> None:
    from tests.unit.validation.test_checker_stack_shape import _commit_all

    root = _command_baseline_fixture(tmp_path)
    _commit_all(root)
    _edit(root / _INFRA, "example.legacy_consumer", "example.new_consumer")
    _edit(root / _DATA, "example.legacy_consumer", "example.new_consumer")
    refused = _text(_judge(root, base_ref="HEAD"))
    assert "command baseline for legacy-consumer" in refused
    assert "example.new_consumer" in refused
    assert "not on the base revision" in refused


def test_allowed_commands_baseline_count_cannot_grow_against_base(
    tmp_path: Path,
) -> None:
    from tests.unit.validation.test_checker_stack_shape import _commit_all

    root = _command_baseline_fixture(tmp_path)
    _commit_all(root)
    data = yaml.safe_load((root / _DATA).read_text())
    data["command_baseline"][0]["files"] = 2
    data["baseline"][0]["files"] = 2
    (root / _DATA).write_text(yaml.safe_dump(data))
    compose = yaml.safe_load((root / _OVERLAY).read_text())
    compose["services"]["legacy-consumer"] = {
        "command": ["python", "-m", "example.legacy_consumer"],
        "image": "example/runtime:tag",
    }
    (root / _OVERLAY).write_text(yaml.safe_dump(compose))
    assert "not on the base revision at this count" in _text(
        _judge(root, base_ref="HEAD")
    )


def test_allowed_commands_baseline_can_shrink_with_its_service(tmp_path: Path) -> None:
    from tests.unit.validation.test_checker_stack_shape import _commit_all

    root = _command_baseline_fixture(tmp_path)
    _commit_all(root)
    data = yaml.safe_load((root / _DATA).read_text())
    data["command_baseline"] = []
    data["baseline"] = [
        r for r in data["baseline"] if r["service"] != "legacy-consumer"
    ]
    (root / _DATA).write_text(yaml.safe_dump(data))
    compose = yaml.safe_load((root / _INFRA).read_text())
    del compose["services"]["legacy-consumer"]
    (root / _INFRA).write_text(yaml.safe_dump(compose))
    assert _judge(root, base_ref="HEAD") == ()


def test_allowed_commands_first_baseline_is_held_to_base_compose(
    tmp_path: Path,
) -> None:
    from tests.unit.validation.test_checker_stack_shape import _commit_all

    root = _command_baseline_fixture(tmp_path)
    current = (root / _DATA).read_text()
    data = yaml.safe_load(current)
    del data["command_baseline"]
    (root / _DATA).write_text(yaml.safe_dump(data))
    _commit_all(root)
    (root / _DATA).write_text(current)
    assert _judge(root, base_ref="HEAD") == ()
    _edit(root / _INFRA, "example.legacy_consumer", "example.planted")
    _edit(root / _DATA, "example.legacy_consumer", "example.planted")
    assert "not on the base revision" in _text(_judge(root, base_ref="HEAD"))


@pytest.mark.parametrize("files", [True, "1", 0, -1, 1.5, None])
def test_allowed_commands_rejects_untyped_baseline_counts(
    tmp_path: Path, files: object
) -> None:
    root = _command_baseline_fixture(tmp_path)
    data = yaml.safe_load((root / _DATA).read_text())
    data["command_baseline"][0]["files"] = files
    (root / _DATA).write_text(yaml.safe_dump(data))
    assert "positive files count" in _text(_judge(root))


@pytest.mark.parametrize("services", ["omninode-runtime", [1], [None], None])
def test_allowed_commands_rejects_untyped_service_scopes(
    tmp_path: Path, services: object
) -> None:
    root = _stack(tmp_path)
    data = yaml.safe_load((root / _DATA).read_text())
    data["allowed_commands"][0]["services"] = services
    (root / _DATA).write_text(yaml.safe_dump(data))
    assert "allowed_commands.services must be a string list" in _text(_judge(root))


def test_allowed_commands_checks_the_forwarder_outside_the_runtime_stack(
    tmp_path: Path,
) -> None:
    root = _stack(tmp_path)
    path = root / "docker/docker-compose.unrelated.yml"
    path.write_text(
        "services:\n  gateway-forwarder:\n    command: [onex-gateway-forwarder, --config, forwarder.yaml]\n"
    )
    data = yaml.safe_load((root / _DATA).read_text())
    data["command_only_compose_files"] = [path.relative_to(root).as_posix()]
    (root / _DATA).write_text(yaml.safe_dump(data))
    assert _judge(root) == ()
    _edit(
        path,
        "onex-gateway-forwarder, --config, forwarder.yaml",
        "python, -m, planted.forwarder",
    )
    refused = _text(_judge(root))
    assert "gateway-forwarder" in refused and "planted.forwarder" in refused


def test_allowed_commands_console_scripts_need_a_disposition(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "example"\n[project.scripts]\nrogue = "example.cli:main"\n'
    )
    assert "example" in _text(_judge(root))
    data = yaml.safe_load((root / _DATA).read_text())
    data["console_scripts"] = {
        "example": {
            "rogue": {
                "target": "example.cli:main",
                "status": "developer_only",
                "reason": "Developer diagnostic",
            }
        }
    }
    (root / _DATA).write_text(yaml.safe_dump(data))
    assert _judge(root) == ()
    _edit(
        root / "pyproject.toml",
        'rogue = "example.cli:main"',
        'rogue = "example.cli:main"\nsecond = "example.other:main"',
    )
    assert "example:second" in _text(_judge(root))


def test_allowed_commands_gates_cover_the_extension_and_manifest() -> None:
    import re

    from tests.unit.validation.test_checker_stack_shape import _REPO, _hook

    hook = _hook()
    assert "tests/unit/docker/test_allowed_commands.py" in str(hook["entry"])
    pattern = re.compile(str(hook["files"]))
    for path in (
        "pyproject.toml",
        "uv.lock",
        "tests/unit/docker/test_allowed_commands.py",
        "src/omnibase_infra/nodes/node_architecture_validator/handlers/handler_architecture_validation.py",
        "src/omnibase_infra/nodes/node_architecture_validator/contract.yaml",
    ):
        assert pattern.search(path), path
    workflow = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text())
    steps = "\n".join(
        str(s.get("run", "")) for s in workflow["jobs"]["stack-shape"]["steps"]
    )
    assert "pytest tests/unit/docker -q -k allowed_commands" in steps


@pytest.mark.parametrize("profile", ["inmemory", "redpanda_overlay"])
@pytest.mark.asyncio
async def test_allowed_commands_builds_the_node_and_dispatches_under_both_profiles(
    tmp_path: Path, profile: str
) -> None:
    """Both DI transports build the node; pure compute dispatch needs no broker I/O."""
    from uuid import uuid4

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
    from omnibase_core.services.service_local_handler_ownership_query import (
        ServiceLocalHandlerOwnershipQuery,
    )
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
        ModelKafkaEventBusConfig,
    )
    from omnibase_infra.nodes.node_architecture_validator.models.model_architecture_validation_request import (
        ModelArchitectureValidationRequest,
    )
    from omnibase_infra.nodes.node_architecture_validator.models.model_architecture_validation_result import (
        ModelArchitectureValidationResult,
    )
    from omnibase_infra.nodes.node_architecture_validator.models.model_stack_shape_snapshot import (
        ModelStackShapeSnapshot,
    )
    from omnibase_infra.nodes.node_architecture_validator.node_architecture_validator import (
        NodeArchitectureValidatorCompute,
    )
    from omnibase_infra.runtime.auto_wiring.discovery import (
        discover_contracts_from_paths,
    )
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _prepare_handler_wiring,
    )
    from tests.unit.validation.test_checker_stack_shape import _REPO

    root = _stack(tmp_path)
    contract_path = (
        _REPO / "src/omnibase_infra/nodes/node_architecture_validator/contract.yaml"
    )
    from omnibase_core.utils.util_safe_yaml_loader import (
        load_yaml_mapping_no_duplicates,
    )

    load_yaml_mapping_no_duplicates(
        contract_path.read_text(), source=str(contract_path)
    )
    manifest = discover_contracts_from_paths([contract_path])
    assert not manifest.errors, manifest.errors
    contract = manifest.contracts[0]
    assert contract.event_bus is not None
    assert contract.handler_routing is not None
    assert contract.event_bus.subscribe_topics == (
        "onex.cmd.platform.architecture-validate.v1",
    )
    container = ModelONEXContainer()
    node = NodeArchitectureValidatorCompute(container=container)
    assert isinstance(node, NodeArchitectureValidatorCompute)
    bus = (
        EventBusInmemory(environment="test")
        if profile == "inmemory"
        else EventBusKafka.from_config(
            ModelKafkaEventBusConfig(bootstrap_servers="redpanda:9092")
        )
    )
    prepared = _prepare_handler_wiring(
        contract=contract,
        entry=contract.handler_routing.handlers[0],
        dispatch_engine=None,
        resolver=ServiceHandlerResolver(),
        ownership_query=ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        ),
        event_bus=bus,
        container=container,
    )
    assert not prepared.is_skip
    assert prepared.dispatcher is not None
    correlation = str(uuid4())
    request = ModelArchitectureValidationRequest(
        correlation_id=correlation,
        stack_shape=ModelStackShapeSnapshot(
            policy_yaml=(root / _DATA).read_text(),
            compose_files={
                "planted.yml": "services:\n  omninode-runtime:\n    command: [python, -m, planted.sidecar]\n"
            },
        ),
    )
    result = await prepared.dispatcher(
        {
            "payload": {
                "payload": request.model_dump(mode="json"),
                "event_type": "ModelArchitectureValidationRequest",
                "correlation_id": correlation,
            },
            "__bindings": {},
            "__debug_trace": {
                "topic": contract.event_bus.subscribe_topics[0],
                "correlation_id": correlation,
            },
        }
    )
    events = [
        event
        for event in result.output_events
        if isinstance(event, ModelArchitectureValidationResult)
    ]
    assert len(events) == 1, result
    assert events[0].correlation_id == correlation
    assert any(
        "omninode-runtime" in v.message and "planted.sidecar" in v.message
        for v in events[0].violations
    )


def test_allowed_commands_console_scripts_cover_real_declarations() -> None:
    from tests.unit.validation.test_checker_stack_shape import _REPO

    refused = _judge(_REPO, base_ref="HEAD")
    assert refused == (), _text(refused)


def test_allowed_commands_generic_node_client_publishes_the_request(
    tmp_path: Path,
) -> None:
    import json
    import subprocess

    from tests.unit.validation.test_checker_stack_shape import _REPO

    root = _stack(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "stack_shape": {
                    "policy_yaml": (root / _DATA).read_text(),
                    "compose_files": {
                        "planted.yml": "services:\n  omninode-runtime:\n    command: [python, -m, planted.sidecar]\n"
                    },
                }
            }
        )
    )
    state = tmp_path / "runtime-state"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "onex",
            "node",
            "node_architecture_validator",
            "--input",
            str(request),
            "--backend",
            "event_bus=inmemory",
            "--state-root",
            str(state),
        ],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((state / "workflow_result.json").read_text())
    violations = result["terminal_payload"]["violations"]
    assert any(
        "omninode-runtime" in v["message"] and "planted.sidecar" in v["message"]
        for v in violations
    )
