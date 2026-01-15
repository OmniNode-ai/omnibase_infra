# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event bus implementations for omnibase_infra.

This module provides event bus implementations for the ONEX infrastructure.
Two implementations are supported:
- EventBusInmemory: For unit testing and local development without external dependencies
- EventBusKafka: For production use with Kafka/Redpanda (see event_bus_kafka.py)

Event bus selection is handled by kernel.py at bootstrap time based on:
- KAFKA_BOOTSTRAP_SERVERS environment variable (if set, uses EventBusKafka)
- config.event_bus.type field in runtime_config.yaml

Exports:
    EventBusInmemory: In-memory event bus for local testing and development
    ModelEventHeaders: Event headers model for message metadata
    ModelEventMessage: Event message model wrapping topic, key, value, and headers

Topic Constants:
    DLQ_TOPIC_VERSION: Current DLQ topic version
    DLQ_DOMAIN: DLQ domain identifier
    DLQ_INTENT_TOPIC_SUFFIX: Suffix for intent DLQ topics
    DLQ_EVENT_TOPIC_SUFFIX: Suffix for event DLQ topics
    DLQ_COMMAND_TOPIC_SUFFIX: Suffix for command DLQ topics
    DLQ_CATEGORY_SUFFIXES: Mapping of categories to DLQ suffixes
    DLQ_TOPIC_PATTERN: Regex pattern for DLQ topic validation
    build_dlq_topic: Build a DLQ topic from components
    parse_dlq_topic: Parse a DLQ topic into components
    is_dlq_topic: Check if a topic is a DLQ topic
    get_dlq_topic_for_original: Get DLQ topic for an original topic
"""

from __future__ import annotations

from omnibase_infra.event_bus.event_bus_inmemory import (
    EventBusInmemory,
    ModelEventHeaders,
    ModelEventMessage,
)
from omnibase_infra.event_bus.topic_constants import (
    DLQ_CATEGORY_SUFFIXES,
    DLQ_COMMAND_TOPIC_SUFFIX,
    DLQ_DOMAIN,
    DLQ_EVENT_TOPIC_SUFFIX,
    DLQ_INTENT_TOPIC_SUFFIX,
    DLQ_TOPIC_PATTERN,
    DLQ_TOPIC_VERSION,
    build_dlq_topic,
    get_dlq_topic_for_original,
    is_dlq_topic,
    parse_dlq_topic,
)

__all__: list[str] = [
    "DLQ_CATEGORY_SUFFIXES",
    "DLQ_COMMAND_TOPIC_SUFFIX",
    "DLQ_DOMAIN",
    "DLQ_EVENT_TOPIC_SUFFIX",
    "DLQ_INTENT_TOPIC_SUFFIX",
    "DLQ_TOPIC_PATTERN",
    # Topic Constants
    "DLQ_TOPIC_VERSION",
    # Event Bus
    "EventBusInmemory",
    "ModelEventHeaders",
    "ModelEventMessage",
    # Topic Functions
    "build_dlq_topic",
    "get_dlq_topic_for_original",
    "is_dlq_topic",
    "parse_dlq_topic",
]
