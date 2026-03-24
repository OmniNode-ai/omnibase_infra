# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test NodeEventBusWiringEffect wraps EventBusSubcontractWiring [OMN-6350]."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_node_event_bus_wiring_effect_exists() -> None:
    """NodeEventBusWiringEffect must be importable."""
    from omnibase_infra.nodes.node_event_bus_wiring_effect.node import (
        NodeEventBusWiringEffect,
    )

    assert NodeEventBusWiringEffect is not None


@pytest.mark.unit
def test_handler_event_bus_wiring_exists() -> None:
    """HandlerEventBusWiring must be importable and have handle method."""
    from omnibase_infra.nodes.node_event_bus_wiring_effect.handlers.handler_event_bus_wiring import (
        HandlerEventBusWiring,
    )

    handler = HandlerEventBusWiring(container=MagicMock())
    assert hasattr(handler, "handle")
