# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler routing entry model for event-to-handler mapping.

This model represents a single routing entry that maps an event model
(identified by routing_key) to a handler (identified by handler_key).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelRoutingEntry(BaseModel):
    """Single entry mapping an event model to a handler.

    Attributes:
        routing_key: The event model name used for routing (e.g.,
            "ModelNodeIntrospectionEvent"). This matches against
            incoming event payloads.
        handler_key: The handler's adapter ID in ServiceHandlerRegistry
            (e.g., "handler-node-introspected"). This is the kebab-case
            identifier used to look up the handler.

    Example:
        ```python
        entry = ModelRoutingEntry(
            routing_key="ModelNodeIntrospectionEvent",
            handler_key="handler-node-introspected",
        )
        ```

    Note:
        The handler_key is NOT the class name - it's the kebab-case
        adapter ID. Use _convert_class_to_handler_key() to convert
        from class names like "HandlerNodeIntrospected" to handler
        keys like "handler-node-introspected".
    """

    routing_key: str = Field(
        ...,
        description="Event model name used for routing (e.g., 'ModelNodeIntrospectionEvent')",
    )
    handler_key: str = Field(
        ...,
        description="Handler adapter ID in kebab-case (e.g., 'handler-node-introspected')",
    )

    model_config = {"frozen": True}


__all__ = ["ModelRoutingEntry"]
