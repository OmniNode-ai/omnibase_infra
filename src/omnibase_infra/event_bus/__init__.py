# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event bus implementations for omnibase_infra.

This module provides event bus implementations for local testing and development.
The primary implementation is InMemoryEventBus for unit testing and local development
without requiring external message broker infrastructure.

Exports:
    InMemoryEventBus: In-memory event bus for local testing and development
    ModelEventHeaders: Event headers model for message metadata
    ModelEventMessage: Event message model wrapping topic, key, value, and headers
"""

from __future__ import annotations

from omnibase_infra.event_bus.inmemory_event_bus import (
    InMemoryEventBus,
    ModelEventHeaders,
    ModelEventMessage,
)

__all__: list[str] = [
    "InMemoryEventBus",
    "ModelEventHeaders",
    "ModelEventMessage",
]
