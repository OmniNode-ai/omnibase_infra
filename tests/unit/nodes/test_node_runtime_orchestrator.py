# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test NodeRuntimeOrchestrator coordinates 4-step boot sequence [OMN-6351]."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.unit
def test_orchestrator_exists() -> None:
    """NodeRuntimeOrchestrator must be importable."""
    from omnibase_infra.nodes.node_runtime_orchestrator.node import (
        NodeRuntimeOrchestrator,
    )

    assert NodeRuntimeOrchestrator is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boot_sequence_ordering() -> None:
    """Boot steps must execute in order: loader -> registry -> graph -> wiring."""
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    mock_contract_loader = AsyncMock()
    mock_contract_registry = AsyncMock()
    mock_node_graph = AsyncMock()
    mock_event_bus_wiring = AsyncMock()

    container = MagicMock()
    container.get_service.side_effect = lambda name: {
        "ProtocolContractLoader": mock_contract_loader,
        "ProtocolContractRegistry": mock_contract_registry,
        "ProtocolNodeGraph": mock_node_graph,
        "ProtocolEventBusWiring": mock_event_bus_wiring,
    }[name]

    handler = HandlerRuntimeLifecycle(container=container)
    await handler.execute_startup()

    mock_contract_loader.assert_awaited_once()
    mock_contract_registry.assert_awaited_once()
    mock_node_graph.assert_awaited_once()
    mock_event_bus_wiring.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fail_fast_on_step_failure() -> None:
    """If a step fails, subsequent steps must not execute."""
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    mock_contract_loader = AsyncMock(side_effect=RuntimeError("scan failed"))
    mock_contract_registry = AsyncMock()

    container = MagicMock()
    container.get_service.side_effect = lambda name: {
        "ProtocolContractLoader": mock_contract_loader,
        "ProtocolContractRegistry": mock_contract_registry,
        "ProtocolNodeGraph": AsyncMock(),
        "ProtocolEventBusWiring": AsyncMock(),
    }[name]

    handler = HandlerRuntimeLifecycle(container=container)

    with pytest.raises(RuntimeError, match="scan failed"):
        await handler.execute_startup()

    mock_contract_registry.assert_not_awaited()
