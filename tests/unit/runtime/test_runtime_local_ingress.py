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

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.models import (
    ModelLocalRuntimeIngressConfig,
    ModelLocalRuntimeIngressRequest,
    ModelLocalRuntimeIngressResponse,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    RuntimeLocalIngressRoute,
    RuntimeLocalIngressServer,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.unit


def _session_orchestrator_route() -> RuntimeLocalIngressRoute:
    return RuntimeLocalIngressRoute(
        node_name="node_session_orchestrator",
        contract_name="session_orchestrator",
        command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        event_type="omnimarket.session-orchestrator-start",
        terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        contract_path="/tmp/node_session_orchestrator/contract.yaml",  # noqa: S108
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
                node_alias="test",
                resolved_node_name="node_test",
                dispatch_result=ModelDispatchResult(
                    status=EnumDispatchStatus.SUCCESS,
                    topic="onex.cmd.test.start.v1",
                    started_at=datetime.now(UTC),
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


def test_local_runtime_ingress_request_rejects_blank_node_name() -> None:
    with pytest.raises(ValidationError, match="node_alias must be a non-empty string"):
        ModelLocalRuntimeIngressRequest(node_alias="   ")


def test_parse_active_runtime_packages_honors_env_override() -> None:
    resolved = parse_active_runtime_packages(
        ("omnibase_infra",),
        env={"ONEX_ACTIVE_RUNTIME_PACKAGES": "omnibase_infra, omnimarket"},
    )
    assert resolved == ("omnibase_infra", "omnimarket")


def test_local_runtime_ingress_config_accepts_yaml_list_package_names() -> None:
    config = ModelLocalRuntimeIngressConfig.model_validate(
        {"package_names": ["omnibase_infra", "omnimarket"]}
    )

    assert config.package_names == ("omnibase_infra", "omnimarket")


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
    assert routes["demo"].terminal_event == "onex.evt.demo.completed.v1"


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


def test_discover_runtime_local_ingress_routes_rejects_duplicate_alias(
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

    with pytest.raises(ValueError, match="Duplicate local ingress route alias 'same'"):
        discover_runtime_local_ingress_routes(("pkg1", "pkg2"))


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_request() -> None:
    dispatch_result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )
    dispatch_engine = AsyncMock()
    dispatch_engine.dispatch = AsyncMock(return_value=dispatch_result)

    process = RuntimeHostProcess(
        config=make_runtime_config(),
        dispatch_engine=dispatch_engine,
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }
    applier = SimpleNamespace(apply=AsyncMock())
    process._local_ingress_dispatch_result_applier = cast(
        "DispatchResultApplier", applier
    )

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            node_alias="session_orchestrator",
            payload={"dry_run": True},
            correlation_id=uuid4(),
        )
    )

    assert response.ok is True
    assert response.topic == "onex.cmd.omnimarket.session-orchestrator-start.v1"
    dispatch_engine.dispatch.assert_awaited()
    applier.apply.assert_awaited()


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_uses_handler_semaphore() -> (
    None
):
    dispatch_result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )
    dispatch_engine = AsyncMock()
    dispatch_engine.dispatch = AsyncMock(return_value=dispatch_result)

    process = RuntimeHostProcess(
        config=make_runtime_config(),
        dispatch_engine=dispatch_engine,
    )
    process._is_running = True
    process._handler_semaphore = asyncio.Semaphore(1)
    await process._handler_semaphore.acquire()
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }

    task = asyncio.create_task(
        process._dispatch_local_ingress_request(
            ModelLocalRuntimeIngressRequest(
                node_alias="session_orchestrator",
                payload={"dry_run": True},
                correlation_id=uuid4(),
            )
        )
    )
    await asyncio.sleep(0)
    dispatch_engine.dispatch.assert_not_awaited()

    process._handler_semaphore.release()
    response = await task

    assert response.ok is True
    dispatch_engine.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_host_process_dispatch_local_ingress_request_times_out() -> None:
    async def _sleepy_dispatch(
        *_args: object, **_kwargs: object
    ) -> ModelDispatchResult:
        await asyncio.sleep(0.05)
        return ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            correlation_id=uuid4(),
        )

    dispatch_engine = AsyncMock()
    dispatch_engine.dispatch = AsyncMock(side_effect=_sleepy_dispatch)
    process = RuntimeHostProcess(
        config=make_runtime_config(local_ingress={"enabled": True}),
        dispatch_engine=dispatch_engine,
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": _session_orchestrator_route()
    }

    response = await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            node_alias="session_orchestrator",
            payload={"dry_run": True},
            timeout_ms=1,
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "dispatch_timeout"


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
