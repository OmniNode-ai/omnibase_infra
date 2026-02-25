# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka topic name validation utility.

This module provides a standalone topic name validation function extracted
from ``KafkaEventBus._validate_topic_name()``. Having the logic as a
top-level utility makes it reusable by event publishers, contract validators,
configuration validators, and CLI tooling without requiring a ``KafkaEventBus``
instance.

Validation rules (per the Kafka documentation):
    - Non-empty string
    - Maximum 255 characters
    - Not the reserved names ``"."`` or ``".."``
    - Contains only: ``a-z``, ``A-Z``, ``0-9``, ``.`` (period), ``_`` (underscore),
      ``-`` (hyphen)

Reference:
    https://kafka.apache.org/documentation/#topicconfigs

Example:
    >>> from omnibase_infra.utils.util_topic_validation import validate_topic_name
    >>> validate_topic_name("onex.registration.events")      # passes silently
    >>> validate_topic_name("")                              # raises ProtocolConfigurationError
    >>> validate_topic_name("bad topic!")                    # raises ProtocolConfigurationError
"""

from __future__ import annotations

import re
from uuid import UUID

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# Characters allowed in a Kafka topic name.
_VALID_TOPIC_RE = re.compile(r"^[a-zA-Z0-9._-]+$")

# Maximum topic name length enforced by the Kafka broker.
_MAX_TOPIC_LENGTH = 255


def validate_topic_name(
    topic: str,
    correlation_id: UUID | None = None,
) -> None:
    """Validate a Kafka topic name against ONEX naming rules.

    Validates:
    - Non-empty string
    - At most 255 characters
    - Not the reserved names ``"."`` or ``".."``
    - Only alphanumeric characters, periods (```.```), underscores (```_```),
      and hyphens (```-```) are present

    Args:
        topic: The Kafka topic name to validate.
        correlation_id: Optional correlation ID for error context. When
            ``None``, a new correlation ID is generated automatically.

    Raises:
        ProtocolConfigurationError: If the topic name violates any of the
            rules listed above. The error includes a human-readable message,
            the offending value, and a correlation context for tracing.

    Example:
        >>> validate_topic_name("onex.registration.events")   # passes
        >>> validate_topic_name("")                           # raises
        Traceback (most recent call last):
            ...
        omnibase_infra.errors.ProtocolConfigurationError: Topic name cannot be empty
    """
    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.KAFKA,
        operation="validate_topic",
    )

    if not topic:
        raise ProtocolConfigurationError(
            "Topic name cannot be empty",
            context=context,
            parameter="topic",
            value=topic,
        )

    if len(topic) > _MAX_TOPIC_LENGTH:
        raise ProtocolConfigurationError(
            f"Topic name '{topic}' exceeds maximum length of {_MAX_TOPIC_LENGTH} characters",
            context=context,
            parameter="topic",
            value=topic,
        )

    if topic in (".", ".."):
        raise ProtocolConfigurationError(
            f"Topic name '{topic}' is reserved and cannot be used",
            context=context,
            parameter="topic",
            value=topic,
        )

    if not _VALID_TOPIC_RE.match(topic):
        raise ProtocolConfigurationError(
            f"Topic name '{topic}' contains invalid characters. "
            "Only alphanumeric characters, periods (.), underscores (_), "
            "and hyphens (-) are allowed",
            context=context,
            parameter="topic",
            value=topic,
        )


__all__ = ["validate_topic_name"]
