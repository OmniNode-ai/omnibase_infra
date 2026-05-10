# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for mandatory DI param enforcement (OMN-10278).

Verifies that handlers with removed = None defaults are constructable
when the DI container provides the required params, and raise TypeError
when params are absent — ensuring the resolver path is the only valid
construction path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_container() -> MagicMock:
    container = MagicMock()
    container.get_service.return_value = MagicMock()
    return container


def test_handler_ledger_projection_constructable_with_container() -> None:
    """HandlerLedgerProjection accepts a container without error."""
    from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
        HandlerLedgerProjection,
    )

    container = _make_container()
    handler = HandlerLedgerProjection(container=container)
    assert handler is not None


def test_handler_build_loop_projection_constructable_with_container() -> None:
    """HandlerBuildLoopProjection accepts a container without error."""
    from omnibase_infra.nodes.node_build_loop_projection_compute.handlers.handler_build_loop_projection import (
        HandlerBuildLoopProjection,
    )

    container = _make_container()
    handler = HandlerBuildLoopProjection(container=container)
    assert handler is not None


def test_handler_runtime_lifecycle_constructable_with_container() -> None:
    """HandlerRuntimeLifecycle accepts a container without error."""
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    container = _make_container()
    handler = HandlerRuntimeLifecycle(container=container)
    assert handler is not None


def test_handler_runtime_error_triage_constructable_with_event_bus() -> None:
    """HandlerRuntimeErrorTriage accepts event_bus without error."""
    from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
        HandlerRuntimeErrorTriage,
    )

    event_bus = MagicMock()
    handler = HandlerRuntimeErrorTriage(event_bus=event_bus)
    assert handler is not None


def test_handlers_reject_construction_without_required_params() -> None:
    """All migrated handlers must raise TypeError when called with no args."""
    from omnibase_infra.nodes.node_build_loop_projection_compute.handlers.handler_build_loop_projection import (
        HandlerBuildLoopProjection,
    )
    from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
        HandlerLedgerProjection,
    )
    from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
        HandlerRuntimeErrorTriage,
    )
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    for cls in [
        HandlerLedgerProjection,
        HandlerBuildLoopProjection,
        HandlerRuntimeLifecycle,
        HandlerRuntimeErrorTriage,
    ]:
        with pytest.raises(TypeError):
            cls()  # type: ignore[call-arg]
