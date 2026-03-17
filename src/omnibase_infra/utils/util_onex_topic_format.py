# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ONEX topic format validator.

Validates that Kafka topic names conform to the canonical ONEX 5-segment format:

    onex.<kind>.<producer>.<event-name>.v<N>

where kind is one of: evt, cmd, intent, dlq.

Legacy DLQ topics matching ``<prefix>.dlq.<name>.v<N>`` are accepted with a
distinct result code so callers can distinguish them from fully canonical names.

Kafka-internal topics (prefixed with ``__``) are silently skipped.
"""

from __future__ import annotations

import re
from enum import StrEnum

_RE_ONEX_TOPIC = re.compile(
    r"^onex\.(evt|cmd|intent|dlq)\.[a-z0-9-]+\.[a-z0-9._-]+\.v[1-9]\d*$"
)

_RE_LEGACY_DLQ = re.compile(r"^[a-z][a-z0-9-]*\.dlq\.[a-z0-9-]+\.v[1-9]\d*$")

_KAFKA_INTERNAL_PREFIX = "__"


class TopicValidationResult(StrEnum):
    """Outcome of validating a topic name against the ONEX format."""

    VALID = "valid"
    VALID_LEGACY_DLQ = "valid_legacy_dlq"
    INVALID = "invalid"
    SKIPPED_INTERNAL = "skipped_internal"


def validate_onex_topic_format(topic: str) -> tuple[TopicValidationResult, str]:
    """Validate *topic* against the canonical ONEX topic format.

    Returns a ``(result, reason)`` tuple.  *reason* is an empty string when
    the topic is valid or skipped, and a human-readable explanation otherwise.
    """
    if topic.startswith(_KAFKA_INTERNAL_PREFIX):
        return (TopicValidationResult.SKIPPED_INTERNAL, "")
    if _RE_ONEX_TOPIC.match(topic):
        return (TopicValidationResult.VALID, "")
    if _RE_LEGACY_DLQ.match(topic):
        return (TopicValidationResult.VALID_LEGACY_DLQ, "legacy DLQ format")
    return (
        TopicValidationResult.INVALID,
        f"Topic '{topic}' does not match ONEX format: "
        "onex.(evt|cmd|intent|dlq).<producer>.<event-name>.v<N>",
    )
