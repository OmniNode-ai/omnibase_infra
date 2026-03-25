# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test NodeRuntimeOrchestrator coordinates 4-step boot sequence [OMN-6351]."""

from __future__ import annotations

from unittest.mock import AsyncMock

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

    call_order: list[str] = []
    mock_contract_loader = AsyncMock(side_effect=lambda: call_order.append("loader"))
    mock_contract_registry = AsyncMock(
        side_effect=lambda: call_order.append("registry")
    )
    mock_node_graph = AsyncMock(side_effect=lambda: call_order.append("graph"))
    mock_event_bus_wiring = AsyncMock(side_effect=lambda: call_order.append("wiring"))

    handler = HandlerRuntimeLifecycle(
        steps=(
            mock_contract_loader,
            mock_contract_registry,
            mock_node_graph,
            mock_event_bus_wiring,
        ),
    )
    await handler.execute_startup()

    mock_contract_loader.assert_awaited_once()
    mock_contract_registry.assert_awaited_once()
    mock_node_graph.assert_awaited_once()
    mock_event_bus_wiring.assert_awaited_once()
    # Verify actual ordering, not just that each ran
    assert call_order == ["loader", "registry", "graph", "wiring"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fail_fast_on_step_failure() -> None:
    """If a step fails, subsequent steps must not execute."""
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    mock_contract_loader = AsyncMock(side_effect=RuntimeError("scan failed"))
    mock_contract_registry = AsyncMock()
    mock_node_graph = AsyncMock()
    mock_event_bus_wiring = AsyncMock()

    handler = HandlerRuntimeLifecycle(
        steps=(
            mock_contract_loader,
            mock_contract_registry,
            mock_node_graph,
            mock_event_bus_wiring,
        ),
    )

    with pytest.raises(RuntimeError, match="scan failed"):
        await handler.execute_startup()

    # All subsequent steps must be untouched
    mock_contract_registry.assert_not_awaited()
    mock_node_graph.assert_not_awaited()
    mock_event_bus_wiring.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_steps_raises() -> None:
    """If no steps are provided, must raise ValueError."""
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    handler = HandlerRuntimeLifecycle()
    with pytest.raises(ValueError, match="No boot steps"):
        await handler.execute_startup()
