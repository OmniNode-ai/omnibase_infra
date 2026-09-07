# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration models for event bus implementations.

Pydantic configuration models for event bus implementations,
supporting environment variable overrides and YAML-based configuration loading.

Exports:
    FETCH_BUDGET_ENV_VAR: Name of the deployment variable carrying the budget
    ModelKafkaConsumerFetchBudget: Aggregate consumer fetch-memory budget
    ModelKafkaEventBusConfig: Configuration model for EventBusKafka
"""

from __future__ import annotations

from omnibase_infra.event_bus.models.config.model_kafka_consumer_fetch_budget import (
    FETCH_BUDGET_ENV_VAR,
    ModelKafkaConsumerFetchBudget,
)
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

__all__: list[str] = [
    "FETCH_BUDGET_ENV_VAR",
    "ModelKafkaConsumerFetchBudget",
    "ModelKafkaEventBusConfig",
]
