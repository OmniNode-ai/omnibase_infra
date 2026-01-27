# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Testing utilities for ONEX event bus.

This module provides test adapters and helpers for event bus testing:
- AdapterProtocolEventPublisherInmemory: Test adapter implementing ProtocolEventPublisher
- decode_inmemory_event: Helper function to decode event bus messages

These utilities enable consistent test patterns without per-handler adapter duplication.
"""

from __future__ import annotations

from omnibase_infra.event_bus.testing.adapter_protocol_event_publisher_inmemory import (
    AdapterProtocolEventPublisherInmemory,
    decode_inmemory_event,
)
from omnibase_infra.event_bus.testing.model_publisher_metrics import (
    ModelPublisherMetrics,
)

__all__: list[str] = [
    "AdapterProtocolEventPublisherInmemory",
    "ModelPublisherMetrics",
    "decode_inmemory_event",
]
