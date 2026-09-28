# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The graph command is contract-routed and emits only its signed terminal."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_request import (
    ModelExecutionGraphRequest,
)
from omnibase_core.models.resolver.model_handler_resolver_context import (
    ModelHandlerResolverContext,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_infra.nodes.node_execution_graph_read_effect.handlers.handler_execution_graph_read import (
    HandlerExecutionGraphRead,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
    load_event_bus_subcontract,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandExecutor,
)
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)
from omnibase_infra.runtime.models.model_execution_graph_read_ingress_config import (
    ModelExecutionGraphReadIngressConfig,
)
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_execution_graph_read_effect/contract.yaml"
)
_COMMAND = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
_TERMINAL = "onex.evt.omnibase-infra.delegation-execution-graph-read-terminal.v1"


@pytest.mark.unit
def test_graph_contract_declares_exact_signed_command_and_terminal() -> None:
    subcontract = load_event_bus_subcontract(_CONTRACT)
    assert subcontract.subscribe_topics == [_COMMAND]
    assert subcontract.signed_ingress_topics == [_COMMAND]
    assert subcontract.publish_topics == [_TERMINAL]

    discovered = discover_contracts_from_paths([_CONTRACT])
    assert len(discovered.contracts) == 1
    contract = discovered.contracts[0]
    assert contract.name == "node_execution_graph_read_effect"
    assert contract.runtime_profiles == ("effects",)
    assert contract.handler_routing is not None
    assert (
        contract.handler_routing.handlers[0].handler.name == "HandlerExecutionGraphRead"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_signed_command_subscription_is_wired_when_verifier_exists() -> (
    None
):
    bus = AsyncMock(spec=ProtocolEventBusSubscriber)
    scope = TrustedGatewaySignerScope("gateway", "test", "graph-read")
    wiring = EventBusSubcontractWiring(
        event_bus=bus,
        dispatch_engine=AsyncMock(),
        environment="test",
        node_name="graph-read",
        service="omnibase-infra",
        version="v1",
        execution_graph_read_ingress=ModelExecutionGraphReadIngressConfig(
            command_topic=_COMMAND,
            gateway_policy=TrustedExecutionGraphGatewayPolicy(frozenset({scope})),
        ),
        execution_graph_read_key_provider=InMemoryKeyProvider(),
    )
    await wiring.wire_subscriptions(load_event_bus_subcontract(_CONTRACT), "graph-read")
    bus.subscribe.assert_awaited_once()
    assert bus.subscribe.await_args.kwargs["topic"] == _COMMAND


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_handler_requires_composed_executor() -> None:
    with pytest.raises(TypeError, match="executor"):
        HandlerExecutionGraphRead(cast("ModelONEXContainer", object()))  # type: ignore[call-arg]


@pytest.mark.unit
def test_discovery_cannot_activate_handler_without_composed_executor() -> None:
    context = ModelHandlerResolverContext(
        handler_cls=HandlerExecutionGraphRead,
        handler_module=HandlerExecutionGraphRead.__module__,
        handler_name="HandlerExecutionGraphRead",
        contract_name="node_execution_graph_read_effect",
        node_name="node_execution_graph_read_effect",
        container=ModelONEXContainer(),
    )
    with pytest.raises(TypeError, match="executor"):
        ServiceHandlerResolver().resolve(context)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_handler_returns_no_unsigned_terminal_event() -> None:
    executor = AsyncMock(spec=ExecutionGraphReadCommandExecutor)
    handler = HandlerExecutionGraphRead(
        cast("ModelONEXContainer", object()),
        cast("ExecutionGraphReadCommandExecutor", executor),
    )
    request = ModelExecutionGraphRequest(correlation_id=uuid4(), cursor_mode="latest")
    envelope = ModelEventEnvelope[dict[str, object]](
        correlation_id=request.correlation_id,
        payload=request.model_dump(mode="json"),
    )
    with bind_dispatch_envelope(envelope):
        output = await handler.handle(request)
    executor.handle.assert_awaited_once_with(request)
    assert output.correlation_id == request.correlation_id
    assert output.input_envelope_id == envelope.envelope_id
    assert output.events == ()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_handler_refuses_missing_dispatch_identity_before_execution() -> (
    None
):
    executor = AsyncMock(spec=ExecutionGraphReadCommandExecutor)
    handler = HandlerExecutionGraphRead(
        cast("ModelONEXContainer", object()),
        cast("ExecutionGraphReadCommandExecutor", executor),
    )
    request = ModelExecutionGraphRequest(correlation_id=uuid4(), cursor_mode="latest")
    with pytest.raises(RuntimeError, match="typed dispatch envelope"):
        await handler.handle(request)
    executor.handle.assert_not_awaited()


@pytest.mark.unit
def test_graph_database_config_rejects_same_database_with_other_credentials() -> None:
    with pytest.raises(ValueError, match="distinct databases"):
        ModelExecutionGraphReadDatabases(
            analytics_dsn="postgresql://reader:a@127.0.0.1/omnibase_infra",
            ledger_dsn="postgresql://writer:b@127.0.0.1/omnibase_infra",
        )
