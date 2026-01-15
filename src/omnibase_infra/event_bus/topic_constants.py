# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic naming constants and utilities for ONEX event bus.

This module defines topic naming conventions for the ONEX event-driven architecture,
including Dead Letter Queue (DLQ) topics for permanently failed messages.

Topic Naming Conventions:
    ONEX uses two primary topic naming formats:

    1. **ONEX Kafka Format**: `onex.<domain>.<type>`
       - Example: `onex.registration.events`
       - Used for core ONEX system topics

    2. **Environment-Aware Format**: `<env>.<domain>.<category>.<version>`
       - Example: `dev.user.events.v1`, `prod.order.commands.v1`
       - Used for application-level message routing

DLQ Topic Naming:
    Dead Letter Queue topics follow the Environment-Aware format with 'dlq' as the domain:

    - **DLQ Topic Format**: `<env>.dlq.<category>.<version>`
    - Example: `dev.dlq.intents.v1`, `prod.dlq.events.v1`

    This convention ensures:
    - DLQ topics are clearly identifiable by the 'dlq' domain
    - Category (intents, events, commands) is preserved for routing analysis
    - Environment separation for multi-environment deployments
    - Version control for DLQ message schema evolution

Usage:
    >>> from omnibase_infra.event_bus.topic_constants import (
    ...     build_dlq_topic,
    ...     DLQ_INTENT_TOPIC_SUFFIX,
    ... )
    >>>
    >>> # Build environment-specific DLQ topic
    >>> topic = build_dlq_topic("prod", "intents")
    >>> print(topic)
    prod.dlq.intents.v1
    >>>
    >>> # Using pre-defined suffix
    >>> topic = f"dev.{DLQ_INTENT_TOPIC_SUFFIX}"
    >>> print(topic)
    dev.dlq.intents.v1

See Also:
    - ModelKafkaEventBusConfig.dead_letter_topic: DLQ configuration
    - EventBusKafka._publish_to_dlq(): DLQ publishing implementation
    - topic_category_validator.py: Topic naming validation
"""

from __future__ import annotations

import re
from typing import Final

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# ==============================================================================
# DLQ Topic Version
# ==============================================================================
# Version suffix for DLQ topics. Increment when DLQ message schema changes.
# Current schema includes: original_topic, original_message, failure_reason,
# failure_timestamp, correlation_id, retry_count, error_type

DLQ_TOPIC_VERSION: Final[str] = "v1"

# ==============================================================================
# DLQ Topic Domain
# ==============================================================================
# The 'dlq' domain identifies Dead Letter Queue topics

DLQ_DOMAIN: Final[str] = "dlq"

# ==============================================================================
# DLQ Topic Suffixes (without environment prefix)
# ==============================================================================
# These suffixes can be combined with environment prefix to form full topic names.
# Format: dlq.<category>.<version>

DLQ_INTENT_TOPIC_SUFFIX: Final[str] = f"{DLQ_DOMAIN}.intents.{DLQ_TOPIC_VERSION}"
"""DLQ topic suffix for permanently failed intents: 'dlq.intents.v1'"""

DLQ_EVENT_TOPIC_SUFFIX: Final[str] = f"{DLQ_DOMAIN}.events.{DLQ_TOPIC_VERSION}"
"""DLQ topic suffix for permanently failed events: 'dlq.events.v1'"""

DLQ_COMMAND_TOPIC_SUFFIX: Final[str] = f"{DLQ_DOMAIN}.commands.{DLQ_TOPIC_VERSION}"
"""DLQ topic suffix for permanently failed commands: 'dlq.commands.v1'"""

# ==============================================================================
# Category-to-Suffix Mapping
# ==============================================================================

DLQ_CATEGORY_SUFFIXES: Final[dict[str, str]] = {
    "intent": DLQ_INTENT_TOPIC_SUFFIX,
    "intents": DLQ_INTENT_TOPIC_SUFFIX,
    "event": DLQ_EVENT_TOPIC_SUFFIX,
    "events": DLQ_EVENT_TOPIC_SUFFIX,
    "command": DLQ_COMMAND_TOPIC_SUFFIX,
    "commands": DLQ_COMMAND_TOPIC_SUFFIX,
}
"""Mapping from message category to DLQ topic suffix (singular and plural forms)."""

# ==============================================================================
# DLQ Topic Validation Pattern
# ==============================================================================
# Validates DLQ topics in Environment-Aware format: <env>.dlq.<category>.<version>
# - env: alphanumeric with underscores/hyphens (e.g., dev, staging, prod, test-1)
# - domain: must be 'dlq'
# - category: intents, events, or commands
# - version: v followed by digits (e.g., v1, v2)

DLQ_TOPIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<env>[\w-]+)\.dlq\.(?P<category>intents|events|commands)\.(?P<version>v\d+)$",
    re.IGNORECASE,
)
"""
Regex pattern for validating DLQ topic names.

Groups:
    - env: Environment identifier (e.g., 'dev', 'prod')
    - category: Message category (intents, events, commands)
    - version: Topic version (e.g., 'v1')

Example matches:
    - dev.dlq.intents.v1
    - prod.dlq.events.v1
    - staging.dlq.commands.v2
"""

# ==============================================================================
# Environment Validation Pattern
# ==============================================================================
# Validates environment identifier: alphanumeric with underscores/hyphens only.
# This is the same pattern used in DLQ_TOPIC_PATTERN for the env group.

ENV_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[\w-]+$")
"""
Regex pattern for validating environment identifiers.

Valid examples: 'dev', 'prod', 'staging', 'test-1', 'my_env'
Invalid examples: 'env.name', 'env name', 'env@name', ''
"""


def build_dlq_topic(
    environment: str,
    category: str,
    *,
    version: str | None = None,
) -> str:
    """Build a DLQ topic name from components.

    Constructs a Dead Letter Queue topic name following ONEX conventions
    in the Environment-Aware format: `<env>.dlq.<category>.<version>`.

    Args:
        environment: Environment identifier (e.g., 'dev', 'prod', 'staging').
            Must be alphanumeric with optional underscores or hyphens.
        category: Message category for the DLQ. Accepts both singular and
            plural forms: 'intent'/'intents', 'event'/'events',
            'command'/'commands'. Will be normalized to plural form.
        version: Optional topic version (e.g., 'v1', 'v2'). If not provided,
            defaults to DLQ_TOPIC_VERSION ('v1').

    Returns:
        Fully-qualified DLQ topic name.

    Raises:
        ValueError: If environment is empty/whitespace, has invalid format,
            or category is invalid.

    Example:
        >>> build_dlq_topic("dev", "intents")
        'dev.dlq.intents.v1'
        >>> build_dlq_topic("prod", "intent")  # Singular form accepted
        'prod.dlq.intents.v1'
        >>> build_dlq_topic("staging", "events", version="v2")
        'staging.dlq.events.v2'
        >>> build_dlq_topic("test-env", "commands")
        'test-env.dlq.commands.v1'
        >>> build_dlq_topic("my_env", "intents")  # Underscores allowed
        'my_env.dlq.intents.v1'

    Test cases for environment validation:
        - Valid: 'dev', 'prod', 'staging', 'test-1', 'my_env', 'env123'
        - Invalid: 'env.name' (dots), 'env name' (spaces), 'env@name' (special chars)
        - Invalid: '' (empty), '   ' (whitespace only)
    """
    # Validate environment
    env = environment.strip()
    if not env:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            "environment cannot be empty",
            context=context,
            parameter="environment",
        )

    if not ENV_PATTERN.match(env):
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            f"Invalid environment '{environment}'. "
            "Must be alphanumeric with optional underscores or hyphens (pattern: [\\w-]+).",
            context=context,
            parameter="environment",
            value=environment,
        )

    # Normalize category to lowercase and look up suffix
    cat_lower = category.lower().strip()
    if cat_lower not in DLQ_CATEGORY_SUFFIXES:
        valid_categories = ", ".join(sorted(set(DLQ_CATEGORY_SUFFIXES.keys())))
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            f"Invalid category '{category}'. Valid categories: {valid_categories}",
            context=context,
            parameter="category",
            value=category,
        )

    # Determine version to use
    topic_version = version if version else DLQ_TOPIC_VERSION

    # Normalize category to plural form for consistency
    normalized_category = _normalize_category(cat_lower)

    return f"{env}.{DLQ_DOMAIN}.{normalized_category}.{topic_version}"


def _normalize_category(category: str) -> str:
    """Normalize category to plural form.

    Args:
        category: Category in singular or plural form.

    Returns:
        Category in plural form (intents, events, commands).
    """
    category_map = {
        "intent": "intents",
        "intents": "intents",
        "event": "events",
        "events": "events",
        "command": "commands",
        "commands": "commands",
    }
    return category_map.get(category, category)


def parse_dlq_topic(topic: str) -> dict[str, str] | None:
    """Parse a DLQ topic name into its components.

    Extracts environment, category, and version from a DLQ topic name
    that follows the ONEX naming convention.

    Args:
        topic: The DLQ topic name to parse.

    Returns:
        A dictionary with keys 'environment', 'category', and 'version'
        if the topic matches the DLQ pattern, or None if it doesn't match.

    Example:
        >>> parse_dlq_topic("dev.dlq.intents.v1")
        {'environment': 'dev', 'category': 'intents', 'version': 'v1'}
        >>> parse_dlq_topic("prod.dlq.events.v2")
        {'environment': 'prod', 'category': 'events', 'version': 'v2'}
        >>> parse_dlq_topic("not.a.dlq.topic")
        None
    """
    match = DLQ_TOPIC_PATTERN.match(topic)
    if not match:
        return None

    return {
        "environment": match.group("env"),
        "category": match.group("category"),
        "version": match.group("version"),
    }


def is_dlq_topic(topic: str) -> bool:
    """Check if a topic name is a DLQ topic.

    Args:
        topic: The topic name to check.

    Returns:
        True if the topic matches the DLQ naming pattern, False otherwise.

    Example:
        >>> is_dlq_topic("dev.dlq.intents.v1")
        True
        >>> is_dlq_topic("dev.user.events.v1")
        False
    """
    return DLQ_TOPIC_PATTERN.match(topic) is not None


def get_dlq_topic_for_original(
    original_topic: str,
    environment: str | None = None,
) -> str | None:
    """Get the DLQ topic for an original message topic.

    Infers the appropriate DLQ topic based on the category of the original
    topic. If it follows ONEX naming conventions, the category is extracted
    automatically.

    Args:
        original_topic: The original topic where the message was consumed from.
        environment: Optional environment override. If not provided, attempts
            to extract from the original topic (Environment-Aware format only).

    Returns:
        The DLQ topic name, or None if the category cannot be determined.

    Example:
        >>> get_dlq_topic_for_original("dev.checkout.intents.v1")
        'dev.dlq.intents.v1'
        >>> get_dlq_topic_for_original("prod.order.events.v1")
        'prod.dlq.events.v1'
        >>> get_dlq_topic_for_original("onex.registration.commands")
        None  # ONEX format has no environment, must provide explicitly
        >>> get_dlq_topic_for_original("onex.registration.commands", environment="prod")
        'prod.dlq.commands.v1'
    """
    # Import here to avoid circular imports
    from omnibase_infra.enums import EnumMessageCategory

    # Try to infer category from topic
    category = EnumMessageCategory.from_topic(original_topic)
    if category is None:
        return None

    # Try to extract environment from topic if not provided
    if environment is None:
        # Environment-Aware format: <env>.<domain>.<category>.<version>
        parts = original_topic.split(".")
        if len(parts) >= 2 and parts[0].lower() not in ("onex",):
            environment = parts[0]
        else:
            # Cannot determine environment from ONEX format topics
            return None

    return build_dlq_topic(environment, category.topic_suffix)


__all__ = [
    "DLQ_CATEGORY_SUFFIXES",
    "DLQ_COMMAND_TOPIC_SUFFIX",
    "DLQ_DOMAIN",
    "DLQ_EVENT_TOPIC_SUFFIX",
    "DLQ_INTENT_TOPIC_SUFFIX",
    "DLQ_TOPIC_PATTERN",
    # Constants
    "DLQ_TOPIC_VERSION",
    "ENV_PATTERN",
    # Functions
    "build_dlq_topic",
    "get_dlq_topic_for_original",
    "is_dlq_topic",
    "parse_dlq_topic",
]
