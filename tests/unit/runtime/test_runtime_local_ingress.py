# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.runtime_local_ingress import (
    RuntimeLocalIngressServer,
    discover_runtime_local_ingress_routes,
    parse_active_runtime_packages,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config


@pytest.mark.asyncio
async def test_runtime_local_ingress_server_round_trip(tmp_path: Path) -> None:
    socket_path = Path(f"/tmp/runtime-local-ingress-{uuid4().hex}.sock")  # noqa: S108
    server = RuntimeLocalIngressServer(
        str(socket_path),
        AsyncMock(return_value={"ok": True, "status": "done"}),
    )

    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(b'{"node_name":"test","payload":{}}\n')
        await writer.drain()
        response = await reader.readline()
        writer.close()
        await writer.wait_closed()

        assert json.loads(response.decode("utf-8")) == {"ok": True, "status": "done"}
    finally:
        await server.stop()


def test_parse_active_runtime_packages_honors_env_override() -> None:
    resolved = parse_active_runtime_packages(
        ("omnibase_infra",),
        env={"ONEX_ACTIVE_RUNTIME_PACKAGES": "omnibase_infra, omnimarket"},
    )
    assert resolved == ("omnibase_infra", "omnimarket")


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
        "session_orchestrator": SimpleNamespace(
            node_name="node_session_orchestrator",
            contract_name="session_orchestrator",
            command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
            event_type="omnimarket.session-orchestrator-start",
            terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
        )
    }
    process._local_ingress_dispatch_result_applier = SimpleNamespace(apply=AsyncMock())

    response = await process._dispatch_local_ingress_request(
        {
            "node_name": "session_orchestrator",
            "payload": {"dry_run": True},
            "correlation_id": str(uuid4()),
        }
    )

    assert response["ok"] is True
    assert response["topic"] == "onex.cmd.omnimarket.session-orchestrator-start.v1"
    dispatch_engine.dispatch.assert_awaited()
    process._local_ingress_dispatch_result_applier.apply.assert_awaited()
