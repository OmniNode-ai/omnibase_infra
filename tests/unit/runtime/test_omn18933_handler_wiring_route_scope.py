# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol
from unittest.mock import patch
from uuid import uuid4

import omnimarket
import pytest
import yaml

from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
from omnibase_infra.runtime.bounded_delegation_routes import (
    resolve_bounded_delegation_route,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit


class _AddressedBus:
    environment = "dogfood"
    bootstrap_servers = "192.168.86.105:47092"

    async def publish(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("a declaration mismatch must stop before publish")

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("a declaration mismatch must stop before subscribe")


@pytest.mark.asyncio
async def test_handler_wiring_preserves_bus_identity_into_route_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_file = (
        Path(omnimarket.__file__).resolve().parents[2] / "config" / "ci_bus_lanes.yaml"
    )
    raw = yaml.safe_load(lane_file.read_text(encoding="utf-8"))
    raw["lanes"]["dogfood"]["delegation_routes"][0]["consumer"] = (
        "omnimarket.nodes.wrong"
    )
    workspace = tmp_path / "workspace"
    overlay = workspace / "omnimarket" / "config" / "ci_bus_lanes.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    contract = ModelDiscoveredContract(
        name="node_local",
        node_type="EFFECT_GENERIC",
        contract_version={"major": 1, "minor": 0, "patch": 0},
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_local",
        package_name="test_pkg",
        event_bus={
            "subscribe_topics": ["onex.evt.platform.local-input.v1"],
            "publish_topics": [],
        },
        handler_routing={
            "routing_strategy": "payload_type_match",
            "handlers": [
                {
                    "handler": {
                        "name": "HandlerWithDispatchPort",
                        "module": "fake.module",
                    },
                    "event_model": None,
                    "operation": None,
                }
            ],
        },
    )
    entry = contract.handler_routing.handlers[0]
    bus = _AddressedBus()

    class ProtocolDelegationDispatchPort(Protocol):
        def dispatch(self, **kwargs: object) -> Awaitable[dict[str, object]]: ...

    class HandlerWithDispatchPort:
        last_port: object | None = None

        def __init__(
            self,
            event_bus: object,
            dispatch_port: ProtocolDelegationDispatchPort | None = None,
        ) -> None:
            self.event_bus = event_bus
            self.dispatch_port = dispatch_port
            HandlerWithDispatchPort.last_port = dispatch_port

        def handle(self, envelope: object) -> None:
            return None

    ownership = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerWithDispatchPort,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=ServiceHandlerResolver(),
            ownership_query=ownership,
            event_bus=bus,
            container=None,
        )

    assert prepared.is_skip is False
    port = HandlerWithDispatchPort.last_port
    assert isinstance(port, RuntimeDelegationDispatchPort)
    assert port._event_bus is bus

    route = ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator/contract.yaml",
        package_name="omnimarket",
    )
    with (
        patch(
            "omnibase_infra.runtime.service_delegation_dispatch_port.discover_runtime_local_ingress_routes",
            return_value={
                "omnimarket.node_delegation_orchestrator.delegation.orchestrate": route
            },
        ),
        patch(
            "omnibase_infra.runtime.service_delegation_dispatch_port.resolve_bounded_delegation_route",
            side_effect=lambda **kwargs: resolve_bounded_delegation_route(
                **kwargs,
                overlay_path_for_test=overlay,
            ),
        ),
        patch(
            "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
            side_effect=AssertionError(
                "route guard must run before broker construction"
            ),
        ),
    ):
        with pytest.raises(InfraUnavailableError, match="consumer mismatch"):
            await port.dispatch(
                prompt="route-guard",
                task_type="document",
                correlation_id=uuid4(),
                max_tokens=64,
                source_file_path=None,
                source_session_id=None,
                wait=True,
                execution_timeout_seconds=10,
                terminal_delivery_margin_seconds=2,
            )
