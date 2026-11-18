# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event bus implementations for omnibase_infra.

Event bus implementations for the ONEX infrastructure.
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
    derive_dlq_topic_for_event_type: Derive DLQ topic from event_type domain prefix
"""

import importlib
from typing import Any

# OMN-23562: exports resolve on first read, not at import. Importing any
# submodule of this package runs this file, and an eager import of
# ``service_topic_manager`` here loaded the generated topic enums (about 1.4 s)
# for CLI commands that never touch a topic. Modules are tried in this order, so
# the cheap ones answer first. Importers see these names as ``Any``.
_EXPORT_MODULES: tuple[str, ...] = (
    "omnibase_infra.event_bus.enum_topic_validation_status",
    "omnibase_infra.event_bus.model_topic_validation_result",
    "omnibase_infra.event_bus.topic_constants",
    "omnibase_infra.event_bus.event_bus_inmemory",
    "omnibase_infra.event_bus.models",
    "omnibase_infra.event_bus.service_topic_startup_validator",
    "omnibase_infra.event_bus.service_topic_manager",
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
    # Enums
    "EnumTopicValidationStatus",
    # Event Bus
    "EventBusInmemory",
    "ModelEventHeaders",
    "ModelEventMessage",
    # Topic Management
    "ModelTopicValidationResult",
    "TopicProvisioner",
    "TopicStartupValidator",
    # Topic Functions
    "build_dlq_topic",
    "derive_dlq_topic_for_event_type",
    "get_dlq_topic_for_original",
    "is_dlq_topic",
    "parse_dlq_topic",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        for module_name in _EXPORT_MODULES:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                value = getattr(module, name)
                globals()[name] = value
                return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
