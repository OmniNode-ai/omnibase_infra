"""Event bus models for omnibase_infra.

This module provides concrete Pydantic model implementations that satisfy
the SPI protocols (ProtocolEventMessage, ProtocolEventHeaders) for use
with event bus implementations.

Exports:
    ModelEventHeaders: Headers for event bus messages implementing ProtocolEventHeaders
    ModelEventMessage: Event bus message implementing ProtocolEventMessage
    ModelKafkaEventBusConfig: Configuration model for EventBusKafka
    ModelDlqEvent: DLQ event model for callback payloads
    ModelDlqMetrics: Aggregate metrics for DLQ operations
"""

from __future__ import annotations

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.models.model_dlq_event import ModelDlqEvent
from omnibase_infra.event_bus.models.model_dlq_metrics import ModelDlqMetrics
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

__all__: list[str] = [
    "ModelDlqEvent",
    "ModelDlqMetrics",
    "ModelEventHeaders",
    "ModelEventMessage",
    "ModelKafkaEventBusConfig",
]
