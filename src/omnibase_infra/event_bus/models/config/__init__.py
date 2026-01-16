# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration models for event bus implementations.

This module provides Pydantic configuration models for event bus implementations,
supporting environment variable overrides and YAML-based configuration loading.

Exports:
    ModelKafkaEventBusConfig: Configuration model for EventBusKafka
"""

from __future__ import annotations

from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

__all__: list[str] = [
    "ModelKafkaEventBusConfig",
]
