# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.models import (
    ModelLocalRuntimeIngressConfig,
    ModelLocalRuntimeIngressRequest,
    ModelLocalRuntimeIngressResponse,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    RuntimeLocalIngressServer,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.unit

_SESSION_ORCHESTRATOR_CONTRACT_PATH = (
    "/var/lib/omninode/node_session_orchestrator/contract.yaml"
)


def _session_orchestrator_route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path=_SESSION_ORCHESTRATOR_CONTRACT_PATH,
        package_name="omnimarket",
    )


@pytest.mark.asyncio
async def test_runtime_local_ingress_server_round_trip(tmp_path: Path) -> None:
    socket_path = Path(f"/tmp/runtime-local-ingress-{uuid4().hex}.sock")  # noqa: S108
    server = RuntimeLocalIngressServer(
        str(socket_path),
        AsyncMock(
            return_value=ModelLocalRuntimeIngressResponse(
                ok=True,
                command_name="test",
                node_alias="test",
                resolved_node_name="node_test",
                command_topic="onex.cmd.test.start.v1",
                dispatch_result=ModelDispatchBusTerminalResult(
                    status="completed",
                    payload={"ok": True},
                    completed_at=datetime.now(UTC),
                    correlation_id=uuid4(),
                ),
            )
        ),
    )

    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(b'{"node_alias":"test","payload":{}}\n')
        await writer.drain()
        response = await reader.readline()
        writer.close()
        await writer.wait_closed()

        decoded = json.loads(response.decode("utf-8"))
        assert decoded["ok"] is True
        assert decoded["node_alias"] == "test"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_runtime_local_ingress_server_refuses_non_socket_path(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "runtime-local-ingress.sock"
    socket_path.write_text("do not unlink", encoding="utf-8")
    server = RuntimeLocalIngressServer(
        str(socket_path),
        AsyncMock(),
    )

    with pytest.raises(FileExistsError, match="not an owned Unix socket"):
        await server.start()

    assert socket_path.read_text(encoding="utf-8") == "do not unlink"


def test_local_runtime_ingress_request_rejects_blank_name() -> None:
    with pytest.raises(
        ValidationError, match="command_name/node_alias must be a non-empty string"
    ):
        ModelLocalRuntimeIngressRequest(node_alias="   ")


def test_parse_active_runtime_packages_honors_env_override() -> None:
    resolved = parse_active_runtime_packages(
        ("omnibase_infra",),
        env={"ONEX_ACTIVE_RUNTIME_PACKAGES": "omnibase_infra, omnimarket"},
    )
    assert resolved == ("omnibase_infra", "omnimarket")


def test_local_runtime_ingress_config_accepts_yaml_list_package_names() -> None:
    config = ModelLocalRuntimeIngressConfig.model_validate(
        {
            "package_names": ["omnibase_infra", "omnimarket"],
            "enabled_profiles": ["main"],
        }
    )

    assert config.package_names == ("omnibase_infra", "omnimarket")
    assert config.enabled_profiles == ("main",)


@pytest.mark.asyncio
async def test_runtime_host_process_skips_local_ingress_for_disallowed_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PROFILE", "effects")
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process.discover_runtime_local_ingress_routes",
        lambda _packages: pytest.fail("effects profile must not discover routes"),
    )

    process = RuntimeHostProcess(
        config=make_runtime_config(
            local_ingress={
                "enabled": True,
                "enabled_profiles": ["main"],
            }
        ),
        dispatch_engine=AsyncMock(),
    )

    await process._start_local_ingress()

    assert process._local_ingress_routes == {}
    assert process._local_ingress_active_packages == ()


def test_discover_runtime_local_ingress_routes_registers_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "fakepkg"
    (package_root / "nodes" / "node_demo").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nodes" / "node_demo" / "contract.yaml").write_text(
        """
name: demo
event_bus:
  subscribe_topics:
    - onex.cmd.demo.start.v1
terminal_event: onex.evt.demo.completed.v1
terminal_events:
  success: onex.evt.demo.completed.v1
  failure: onex.evt.demo.failed.v1
handler_routing:
  handlers:
    - operation: demo.run
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    routes = discover_runtime_local_ingress_routes(("fakepkg",))

    assert routes["demo"].command_topic == "onex.cmd.demo.start.v1"
    assert routes["node_demo"].contract_name == "demo"
    assert routes["demo.run"].contract_name == "demo"
    assert "node_demo.demo.run" not in routes
    assert routes["demo"].terminal_event == "onex.evt.demo.completed.v1"
    assert routes["demo"].terminal_events == (
        "onex.evt.demo.completed.v1",
        "onex.evt.demo.failed.v1",
    )


def test_discover_runtime_local_ingress_routes_skips_malformed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "fakepkg"
    (package_root / "nodes" / "node_bad").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nodes" / "node_bad" / "contract.yaml").write_text(
        "name: [unterminated",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    assert discover_runtime_local_ingress_routes(("fakepkg",)) == {}


def test_discover_runtime_local_ingress_routes_omits_ambiguous_public_base_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_one = tmp_path / "pkg1"
    package_two = tmp_path / "pkg2"
    for root in (package_one, package_two):
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "nodes" / "node_same").mkdir(parents=True)
    (package_one / "nodes" / "node_same" / "contract.yaml").write_text(
        "name: same\nevent_bus:\n  subscribe_topics:\n    - onex.cmd.alpha.start.v1\n",
        encoding="utf-8",
    )
    (package_two / "nodes" / "node_same" / "contract.yaml").write_text(
        "name: same\nevent_bus:\n  subscribe_topics:\n    - onex.cmd.beta.start.v1\n",
        encoding="utf-8",
    )

    def _import_module(name: str) -> object:
        root = package_one if name == "pkg1" else package_two
        return SimpleNamespace(__file__=str(root / "__init__.py"))

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        _import_module,
    )

    routes = discover_runtime_local_ingress_routes(("pkg1", "pkg2"))

    assert "same" not in routes
    assert "node_same" not in routes
    assert routes["pkg1.same"].command_topic == "onex.cmd.alpha.start.v1"
    assert routes["pkg1.node_same"].contract_name == "same"
    assert routes["pkg2.same"].command_topic == "onex.cmd.beta.start.v1"
    assert routes["pkg2.node_same"].contract_name == "same"


def test_discover_runtime_local_ingress_routes_allows_equivalent_duplicate_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_one = tmp_path / "pkg1"
    package_two = tmp_path / "pkg2"
    for root in (package_one, package_two):
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "nodes" / "node_same").mkdir(parents=True)
        (root / "nodes" / "node_same" / "contract.yaml").write_text(
            """
name: same
event_bus:
  subscribe_topics:
    - onex.cmd.same.start.v1
terminal_event: onex.evt.same.completed.v1
handler_routing:
  handlers:
    - operation: same.run
""".strip(),
            encoding="utf-8",
        )

    def _import_module(name: str) -> object:
        root = package_one if name == "pkg1" else package_two
        return SimpleNamespace(__file__=str(root / "__init__.py"))

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        _import_module,
    )

    routes = discover_runtime_local_ingress_routes(("pkg1", "pkg2"))

    assert routes["same"].contract_path == str(
        package_one / "nodes" / "node_same" / "contract.yaml"
    )
    assert routes["node_same"].command_topic == "onex.cmd.same.start.v1"
    assert routes["same.run"].terminal_event == "onex.evt.same.completed.v1"


def test_discover_runtime_local_ingress_routes_omits_ambiguous_public_operation_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_one = tmp_path / "pkg1"
    package_two = tmp_path / "pkg2"
    for root, node_name, contract_name, topic in (
        (package_one, "node_alpha", "alpha", "onex.cmd.alpha.start.v1"),
        (package_two, "node_beta", "beta", "onex.cmd.beta.start.v1"),
    ):
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("", encoding="utf-8")
        node_dir = root / "nodes" / node_name
        node_dir.mkdir(parents=True)
        (node_dir / "contract.yaml").write_text(
            f"""
name: {contract_name}
event_bus:
  subscribe_topics:
    - {topic}
handler_routing:
  handlers:
    - operation: shared.run
""".strip(),
            encoding="utf-8",
        )

    def _import_module(name: str) -> object:
        root = package_one if name == "pkg1" else package_two
        return SimpleNamespace(__file__=str(root / "__init__.py"))

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        _import_module,
    )

    routes = discover_runtime_local_ingress_routes(("pkg1", "pkg2"))

    assert "shared.run" not in routes
    assert routes["pkg1.alpha.shared.run"].command_topic == "onex.cmd.alpha.start.v1"
    assert routes["pkg1.node_alpha.shared.run"].contract_name == "alpha"
    assert routes["pkg2.beta.shared.run"].command_topic == "onex.cmd.beta.start.v1"
    assert routes["pkg2.node_beta.shared.run"].contract_name == "beta"


def test_discover_runtime_local_ingress_routes_omits_ambiguous_raw_operation_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    for node_name, contract_name, topic in (
        ("node_alpha", "alpha", "onex.cmd.alpha.start.v1"),
        ("node_beta", "beta", "onex.cmd.beta.start.v1"),
    ):
        node_dir = package_root / "nodes" / node_name
        node_dir.mkdir(parents=True)
        (node_dir / "contract.yaml").write_text(
            f"""
name: {contract_name}
event_bus:
  subscribe_topics:
    - {topic}
handler_routing:
  handlers:
    - operation: run
""".strip(),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    routes = discover_runtime_local_ingress_routes(("pkg",))

    assert "run" not in routes
    assert routes["alpha.run"].command_topic == "onex.cmd.alpha.start.v1"
    assert routes["node_alpha.run"].contract_name == "alpha"
    assert routes["pkg.alpha.run"].contract_name == "alpha"
    assert routes["pkg.node_alpha.run"].contract_name == "alpha"
    assert routes["beta.run"].command_topic == "onex.cmd.beta.start.v1"
    assert routes["node_beta.run"].contract_name == "beta"
    assert routes["pkg.beta.run"].contract_name == "beta"
    assert routes["pkg.node_beta.run"].contract_name == "beta"


def test_discover_runtime_local_ingress_routes_omits_cross_repo_orchestrator_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infra_root = tmp_path / "omnibase_infra"
    market_root = tmp_path / "omnimarket"
    for root in (infra_root, market_root):
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "nodes" / "node_delegation_orchestrator").mkdir(parents=True)

    (
        infra_root / "nodes" / "node_delegation_orchestrator" / "contract.yaml"
    ).write_text(
        """
name: node_delegation_orchestrator
event_bus:
  subscribe_topics:
    - onex.cmd.omnibase-infra.delegation-request.v1
terminal_event: onex.evt.omnibase-infra.delegation-completed.v1
terminal_events:
  success: onex.evt.omnibase-infra.delegation-completed.v1
  failure: onex.evt.omnibase-infra.delegation-failed.v1
handler_routing:
  handlers:
    - operation: delegation.orchestrate
""".strip(),
        encoding="utf-8",
    )
    (
        market_root / "nodes" / "node_delegation_orchestrator" / "contract.yaml"
    ).write_text(
        """
name: node_delegation_orchestrator
event_bus:
  subscribe_topics:
    - onex.cmd.omnibase-infra.delegation-request.v1
terminal_event: onex.evt.omnibase-infra.delegation-completed.v1
handler_routing:
  handlers:
    - operation: delegation.orchestrate
""".strip(),
        encoding="utf-8",
    )

    def _import_module(name: str) -> object:
        root = infra_root if name == "omnibase_infra" else market_root
        return SimpleNamespace(__file__=str(root / "__init__.py"))

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        _import_module,
    )

    routes = discover_runtime_local_ingress_routes(("omnibase_infra", "omnimarket"))

    assert "node_delegation_orchestrator" not in routes
    assert "delegation.orchestrate" not in routes
    infra_alias = "omnibase_infra.node_delegation_orchestrator.delegation.orchestrate"
    market_alias = "omnimarket.node_delegation_orchestrator.delegation.orchestrate"
    assert routes[infra_alias].terminal_events == (
        "onex.evt.omnibase-infra.delegation-completed.v1",
        "onex.evt.omnibase-infra.delegation-failed.v1",
    )
    assert routes[market_alias].terminal_events == (
        "onex.evt.omnibase-infra.delegation-completed.v1",
    )


def test_discover_runtime_local_ingress_routes_registers_unqualified_operation_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "fakepkg"
    (package_root / "nodes" / "node_demo").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nodes" / "node_demo" / "contract.yaml").write_text(
        """
name: demo
event_bus:
  subscribe_topics:
    - onex.cmd.demo.start.v1
terminal_event: onex.evt.demo.completed.v1
handler_routing:
  handlers:
    - operation: run
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    routes = discover_runtime_local_ingress_routes(("fakepkg",))

    assert routes["run"].command_topic == "onex.cmd.demo.start.v1"
    assert routes["demo.run"].contract_name == "demo"
    assert routes["node_demo.run"].contract_name == "demo"
    assert routes["fakepkg.demo.run"].contract_name == "demo"
    assert routes["fakepkg.node_demo.run"].contract_name == "demo"


def test_discover_runtime_local_ingress_routes_uses_handler_event_type_for_operation_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "fakepkg"
    node_dir = package_root / "nodes" / "node_demo"
    node_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (node_dir / "contract.yaml").write_text(
        """
name: demo
input_model:
  name: ModelDemoRequest
  module: fakepkg.models.model_demo_request
event_bus:
  subscribe_topics:
    - onex.cmd.demo.start.v1
handler_routing:
  handlers:
    - operation: alpha
      event_type: demo.alpha-command
      input_model:
        name: ModelAlphaRequest
        module: fakepkg.models.model_alpha_request
    - operation: beta
      event_type: demo.beta-command
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    routes = discover_runtime_local_ingress_routes(("fakepkg",))

    assert routes["alpha"].event_type == "demo.alpha-command"
    assert routes["demo.alpha"].event_type == "demo.alpha-command"
    assert routes["fakepkg.demo.alpha"].event_type == "demo.alpha-command"
    assert routes["alpha"].input_model_module == "fakepkg.models.model_alpha_request"
    assert routes["alpha"].input_model_name == "ModelAlphaRequest"
    assert routes["beta"].event_type == "demo.beta-command"
    assert routes["demo.beta"].event_type == "demo.beta-command"
    assert routes["fakepkg.demo.beta"].event_type == "demo.beta-command"
    assert routes["beta"].input_model_module == "fakepkg.models.model_demo_request"
    assert routes["beta"].input_model_name == "ModelDemoRequest"


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_request() -> None:
    correlation_id = uuid4()
    dispatch_result = ModelDispatchBusTerminalResult(
        status="completed",
        payload={"status": "complete"},
        completed_at=datetime.now(UTC),
        correlation_id=correlation_id,
    )
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(
            return_value=(_session_orchestrator_route(), dispatch_result)
        )
    )

    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    process._pattern_b_broker = cast("object", broker)

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="session_orchestrator",
            payload={"dry_run": True},
            correlation_id=correlation_id,
        )
    )

    assert response.ok is True
    assert response.command_name == "session_orchestrator"
    assert response.command_topic == "onex.cmd.omnimarket.session-orchestrator-start.v1"
    assert response.output_payloads == [{"status": "complete"}]
    broker.dispatch_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_preserves_request_timeout() -> (
    None
):
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(
            return_value=(
                _session_orchestrator_route(),
                ModelDispatchBusTerminalResult(
                    status="completed",
                    payload={"status": "complete"},
                    completed_at=datetime.now(UTC),
                    correlation_id=uuid4(),
                ),
            )
        )
    )
    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    process._pattern_b_broker = cast("object", broker)

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="session_orchestrator",
            payload={"dry_run": True},
            timeout_ms=600_000,
        )
    )

    assert response.ok is True
    command = broker.dispatch_request.await_args.args[0]
    assert command.timeout_seconds == 600


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_rejects_invalid_payload_before_broker() -> (
    None
):
    route = ModelRuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path=_SESSION_ORCHESTRATOR_CONTRACT_PATH,
        package_name="omnimarket",
        input_model_module=(
            "omnibase_infra.runtime.models.model_local_runtime_ingress_request"
        ),
        input_model_name="ModelLocalRuntimeIngressRequest",
    )
    broker = SimpleNamespace(dispatch_request=AsyncMock())
    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = {"session_orchestrator": route}
    process._pattern_b_broker = cast("object", broker)

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="session_orchestrator",
            payload={"payload": {"dry_run": True}},
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "validation_error"
    broker.dispatch_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_publishes_validated_payload() -> (
    None
):
    route = ModelRuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path=_SESSION_ORCHESTRATOR_CONTRACT_PATH,
        package_name="omnimarket",
        input_model_module=(
            "omnibase_infra.runtime.models.model_local_runtime_ingress_request"
        ),
        input_model_name="ModelLocalRuntimeIngressRequest",
    )
    dispatch_result = ModelDispatchBusTerminalResult(
        status="completed",
        payload={"status": "complete"},
        completed_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(return_value=(route, dispatch_result))
    )
    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = {"session_orchestrator": route}
    process._pattern_b_broker = cast("object", broker)

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="session_orchestrator",
            payload={
                "command_name": "inner-command",
                "payload": {"dry_run": True},
            },
        )
    )

    assert response.ok is True
    command = broker.dispatch_request.await_args.args[0]
    assert command.payload == {
        "command_name": "inner-command",
        "payload": {"dry_run": True},
        "timeout_ms": 300_000,
    }


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_sanitizes_broker_error() -> (
    None
):
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(
            return_value=(
                _session_orchestrator_route(),
                ModelDispatchBusTerminalResult(
                    status="failed",
                    error_message="failed to connect to postgres://user:pass@db:5432/app",
                    completed_at=datetime.now(UTC),
                    correlation_id=uuid4(),
                ),
            )
        )
    )
    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    process._pattern_b_broker = cast("object", broker)

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="session_orchestrator",
            payload={"dry_run": True},
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert (
        response.error.message
        == "RuntimeError: [REDACTED - potentially sensitive data]"
    )


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_uses_handler_semaphore() -> (
    None
):
    dispatch_result = ModelDispatchBusTerminalResult(
        status="completed",
        payload={"status": "complete"},
        completed_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(
            return_value=(_session_orchestrator_route(), dispatch_result)
        )
    )

    process = RuntimeHostProcess(
        config=make_runtime_config(),
        dispatch_engine=AsyncMock(),
    )
    process._is_running = True
    process._handler_semaphore = asyncio.Semaphore(1)
    await process._handler_semaphore.acquire()
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    process._pattern_b_broker = cast("object", broker)

    task = asyncio.create_task(
        process._dispatch_local_ingress_request(
            ModelLocalRuntimeIngressRequest(
                command_name="session_orchestrator",
                payload={"dry_run": True},
                correlation_id=uuid4(),
            )
        )
    )
    await asyncio.sleep(0)
    broker.dispatch_request.assert_not_awaited()

    process._handler_semaphore.release()
    response = await task

    assert response.ok is True
    broker.dispatch_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_request_times_out() -> None:
    async def _sleepy_dispatch(
        *_args: object, **_kwargs: object
    ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
        await asyncio.sleep(0.05)
        return _session_orchestrator_route(), ModelDispatchBusTerminalResult(
            status="completed",
            payload={"status": "complete"},
            completed_at=datetime.now(UTC),
            correlation_id=uuid4(),
        )

    broker = SimpleNamespace(dispatch_request=AsyncMock(side_effect=_sleepy_dispatch))
    process = RuntimeHostProcess(
        config=make_runtime_config(local_ingress={"enabled": True}),
        dispatch_engine=AsyncMock(),
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    process._pattern_b_broker = cast("object", broker)

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="session_orchestrator",
            payload={"dry_run": True},
            timeout_ms=1,
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "dispatch_timeout"


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_warns_on_node_alias_compatibility(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(
            return_value=(
                _session_orchestrator_route(),
                ModelDispatchBusTerminalResult(
                    status="completed",
                    payload={"status": "complete"},
                    completed_at=datetime.now(UTC),
                    correlation_id=uuid4(),
                ),
            )
        )
    )
    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    process._pattern_b_broker = cast("object", broker)

    with caplog.at_level("WARNING"):
        response = await process._dispatch_local_ingress_request(
            ModelLocalRuntimeIngressRequest(
                node_alias="session_orchestrator",
                payload={"dry_run": True},
            )
        )

    assert response.ok is True
    assert "deprecated node_alias compatibility path" in caplog.text


@pytest.mark.asyncio
async def test_runtime_health_includes_local_ingress_details() -> None:
    process = RuntimeHostProcess(
        config=make_runtime_config(local_ingress={"enabled": True}),
        dispatch_engine=AsyncMock(),
    )
    seed_mock_handlers(process)
    process._is_running = True
    process._local_ingress_active_packages = ("omnibase_infra", "omnimarket")
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }

    health = await process.health_check()

    assert "local_ingress" in health
    local_ingress = cast("dict[str, object]", health["local_ingress"])
    assert local_ingress["enabled"] is True
    assert local_ingress["route_count"] == 1
    assert "components" in health
    components = cast("list[dict[str, object]]", health["components"])
    assert any(component["name"] == "local_ingress" for component in components)


# --------------------------------------------------------------------------- #
# OMN-15468: runtime_dispatch.terminal_events must reach the ingress route.
#
# 51 contracts declare their FAILURE terminal only under
# ``runtime_dispatch.terminal_events`` (the address the dashboard and other
# external clients dispatch through) -- measured over the raw corpus of 384
# ``src/omnimarket/nodes/*/contract.yaml`` at
# omnimarket@aea0c33dd89fb82fdca33aac7149992a21c46d43 (origin/dev, 2026-07-30),
# no discovery filter applied. Route discovery read only the TOP-LEVEL
# ``terminal_event`` / ``terminal_events`` keys, so the Pattern B broker -- which
# is already built to race every declared terminal topic (OMN-13118/13128) --
# was only ever handed the success topic. A node that correctly published its
# failure terminal therefore produced either a bogus ``completed`` (when the
# def-B wiring republished the returned model onto the success topic) or a
# spurious ``timeout``. Live reproduction on the .201 dev lane: correlation
# 4a5e0730-0000-4000-8000-000000000002 returned ok=true / status=completed with
# contract_passed=false in the very payload it carried.
# --------------------------------------------------------------------------- #

_GENERATION_CONSUMER_SHAPED_CONTRACT = """
name: gen_demo
event_bus:
  subscribe_topics:
    - onex.cmd.demo.generation-requested.v1
  publish_topics:
    - onex.evt.demo.generation-completed.v1
    - onex.evt.demo.generation-failed.v1
terminal_event: onex.evt.demo.generation-completed.v1
runtime_dispatch:
  command_topic: onex.cmd.demo.generation-requested.v1
  terminal_events:
    success: onex.evt.demo.generation-completed.v1
    failure: onex.evt.demo.generation-failed.v1
  default_timeout_ms: 120000
handler_routing:
  handlers:
    - operation: gen_demo.run
""".strip()

# 30 of those 51 declare NO top-level ``terminal_event`` or ``terminal_events``
# at all, so discovery gave them an EMPTY terminal-topic tuple and the broker
# rejected the command outright ("Route '<name>' does not declare terminal
# events"). 17 of the 30 also clear the route-discovery filter (an ``event_bus``
# mapping whose ``subscribe_topics`` yield a ``.cmd.`` command topic), i.e. 17
# were live undispatchable /skill routes and 13 were latent declarations.
# Same corpus/sha/date as the block above; re-derive, do not copy forward --
# the pre-correction figure here was an unsourced "24 of the 51" that
# reproduces under no framing (see _extract_terminal_events PROVENANCE note).
_PLURAL_ONLY_CONTRACT = """
name: plural_only_demo
event_bus:
  subscribe_topics:
    - onex.cmd.demo.plural-only-requested.v1
  publish_topics:
    - onex.evt.demo.plural-only-completed.v1
    - onex.evt.demo.plural-only-failed.v1
runtime_dispatch:
  command_topic: onex.cmd.demo.plural-only-requested.v1
  terminal_events:
    failure: onex.evt.demo.plural-only-failed.v1
    success: onex.evt.demo.plural-only-completed.v1
handler_routing:
  handlers:
    - operation: plural_only_demo.run
""".strip()


def _write_single_node_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    node_dir: str,
    contract_body: str,
) -> None:
    package_root = tmp_path / "fakepkg"
    (package_root / "nodes" / node_dir).mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nodes" / node_dir / "contract.yaml").write_text(
        contract_body,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )


def test_discover_routes_reads_runtime_dispatch_failure_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure terminal declared ONLY under runtime_dispatch must reach the route."""
    _write_single_node_package(
        tmp_path,
        monkeypatch,
        node_dir="node_gen_demo",
        contract_body=_GENERATION_CONSUMER_SHAPED_CONTRACT,
    )

    route = discover_runtime_local_ingress_routes(("fakepkg",))["gen_demo"]

    assert route.terminal_event == "onex.evt.demo.generation-completed.v1"
    assert route.terminal_events == (
        "onex.evt.demo.generation-completed.v1",
        "onex.evt.demo.generation-failed.v1",
    )


def test_discover_routes_orders_runtime_dispatch_success_terminal_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no top-level terminal_event, success must still be terminal_events[0].

    ``_status_for_terminal_topic`` falls back to ``terminal_topics[0]`` as the
    success topic when ``terminal_event`` is None, so the success entry must be
    hoisted explicitly rather than left to YAML mapping order.
    """
    _write_single_node_package(
        tmp_path,
        monkeypatch,
        node_dir="node_plural_only_demo",
        contract_body=_PLURAL_ONLY_CONTRACT,
    )

    route = discover_runtime_local_ingress_routes(("fakepkg",))["plural_only_demo"]

    assert route.terminal_event is None
    # ``failure`` is declared BEFORE ``success`` in the YAML above on purpose.
    assert route.terminal_events == (
        "onex.evt.demo.plural-only-completed.v1",
        "onex.evt.demo.plural-only-failed.v1",
    )
