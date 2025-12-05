"""Event bus models for omnibase_infra.

This module provides concrete Pydantic model implementations that satisfy
the SPI protocols (ProtocolEventMessage, ProtocolEventHeaders) for use
with event bus implementations.

Exports:
    ModelEventHeaders: Headers for event bus messages implementing ProtocolEventHeaders
    ModelEventMessage: Event bus message implementing ProtocolEventMessage
"""

from __future__ import annotations

from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

__all__: list[str] = [
    "ModelEventHeaders",
    "ModelEventMessage",
]
