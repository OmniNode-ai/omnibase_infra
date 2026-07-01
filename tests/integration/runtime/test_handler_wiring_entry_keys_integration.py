# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration guard for operation-keyed handler auto-wiring IDs."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_dispatcher_id,
    _derive_handler_entry_key,
    _derive_route_id,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelHandlerRef,
    ModelHandlerRoutingEntry,
)


def _routing_entry(operation: str) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(
            name="HandlerShared",
            module="omnimarket.handlers.model_router",
        ),
        operation=operation,
    )


@pytest.mark.integration
def test_operation_keyed_wiring_keeps_shared_handler_routes_distinct() -> None:
    topic = "onex.cmd.omnimarket.model-route.v1"
    gemini_key = _derive_handler_entry_key(_routing_entry("inference.gemini-cli"))
    opencode_key = _derive_handler_entry_key(_routing_entry("inference.opencode-cli"))

    assert gemini_key != opencode_key
    assert _derive_dispatcher_id("node_model_router", gemini_key) != (
        _derive_dispatcher_id("node_model_router", opencode_key)
    )
    assert _derive_route_id("node_model_router", gemini_key, topic) != (
        _derive_route_id("node_model_router", opencode_key, topic)
    )
