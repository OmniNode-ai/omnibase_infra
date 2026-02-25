# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka environment enumeration for type-safe environment handling.

This module defines the environment identifiers used for Kafka topic prefixes
and message routing. These short-form environment names are distinct from
EnumEnvironment (which uses full names like "development", "production") and
are specific to Kafka topic naming conventions.

Usage:
    EnumKafkaEnvironment is used in ModelKafkaEventBusConfig.environment to
    provide type safety and IDE autocomplete for Kafka environment identifiers.
    Invalid values fail at config parse time rather than silently producing
    malformed topic names.

See Also:
    - ModelKafkaEventBusConfig: Uses this enum for environment field
    - EnumEnvironment: Separate enum for full deployment environment names
    - OMN-1871: Original ticket adding this enum
"""

from __future__ import annotations

from enum import StrEnum


class EnumKafkaEnvironment(StrEnum):
    """Environment identifiers for Kafka topic prefixes and message routing.

    These short-form identifiers are used as prefixes in Kafka topic naming
    conventions (e.g., "dev.dlq.intents.v1", "prod.dlq.events.v1").

    Note: These are distinct from EnumEnvironment which uses full names
    (DEVELOPMENT, STAGING, PRODUCTION, CI) for security policy enforcement.
    EnumKafkaEnvironment uses the short Kafka-convention names.

    Attributes:
        DEV: Development environment (topic prefix "dev")
        STAGING: Staging/pre-production environment (topic prefix "staging")
        PROD: Production environment (topic prefix "prod")
        LOCAL: Local development environment (topic prefix "local")

    Example:
        >>> env = EnumKafkaEnvironment.PROD
        >>> env.value
        'prod'
        >>> f"{env.value}.dlq.intents.v1"
        'prod.dlq.intents.v1'
        >>> # Pydantic coerces string values at config parse time
        >>> config = ModelKafkaEventBusConfig(environment="prod")
        >>> config.environment == EnumKafkaEnvironment.PROD
        True
        >>> # Invalid values are rejected at parse time
        >>> ModelKafkaEventBusConfig(environment="invalid")  # raises ValidationError
    """

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    LOCAL = "local"


__all__: list[str] = ["EnumKafkaEnvironment"]
