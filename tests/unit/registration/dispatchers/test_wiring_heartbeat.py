# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for heartbeat dispatcher wiring.

Tests that the wiring module correctly includes the heartbeat
dispatcher and route constants.

Related:
    - OMN-1990: Wire heartbeat dispatcher
"""

from __future__ import annotations


class TestWiringHeartbeatConstants:
    """Tests for heartbeat wiring constants."""

    def test_route_id_constant_exists(self) -> None:
        """ROUTE_ID_NODE_HEARTBEAT is defined in wiring module."""
        from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
            ROUTE_ID_NODE_HEARTBEAT,
        )

        assert ROUTE_ID_NODE_HEARTBEAT == "route.registration.node-heartbeat"

    def test_route_id_in_all_exports(self) -> None:
        """ROUTE_ID_NODE_HEARTBEAT is in __all__."""
        from omnibase_infra.nodes.node_registration_orchestrator import wiring

        assert "ROUTE_ID_NODE_HEARTBEAT" in wiring.__all__

    def test_dispatcher_exported_from_dispatchers(self) -> None:
        """DispatcherNodeHeartbeat is exported from dispatchers package."""
        from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
            DispatcherNodeHeartbeat,
        )

        assert DispatcherNodeHeartbeat is not None

    def test_dispatcher_in_all_exports(self) -> None:
        """DispatcherNodeHeartbeat is in __all__."""
        from omnibase_infra.nodes.node_registration_orchestrator import dispatchers

        assert "DispatcherNodeHeartbeat" in dispatchers.__all__

    def test_handler_getter_exists(self) -> None:
        """get_handler_node_heartbeat_from_container is exported."""
        from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
            get_handler_node_heartbeat_from_container,
        )

        assert callable(get_handler_node_heartbeat_from_container)
