# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka consumer group ID utilities.

Provides utilities for normalizing and validating Kafka consumer group identifiers.
Kafka consumer group IDs have specific character and length constraints that this
module helps enforce consistently across the codebase.

Kafka Consumer Group ID Constraints:
    - Maximum length: 255 characters
    - Valid characters: alphanumeric, period (.), underscore (_), hyphen (-)
    - Cannot be empty

This module provides:
    - normalize_kafka_identifier: Normalize strings for use as Kafka consumer group IDs
    - compute_consumer_group_id: Compute canonical consumer group ID from node identity
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumConsumerGroupPurpose

if TYPE_CHECKING:
    from omnibase_infra.models import ModelNodeIdentity

# Maximum length for Kafka consumer group IDs
KAFKA_CONSUMER_GROUP_MAX_LENGTH = 255

# Pattern for invalid characters (anything not alphanumeric, period, underscore, or hyphen)
_INVALID_CHAR_PATTERN = re.compile(r"[^a-z0-9._-]")

# Pattern for consecutive separators (period, underscore, or hyphen)
_CONSECUTIVE_SEPARATOR_PATTERN = re.compile(r"[._-]{2,}")

# Pattern for leading/trailing separators
_EDGE_SEPARATOR_PATTERN = re.compile(r"^[._-]+|[._-]+$")


def normalize_kafka_identifier(value: str) -> str:
    """Normalize a string for use as a Kafka consumer group ID.

    Applies the following transformations in order:
        1. Convert to lowercase
        2. Replace invalid characters (non-alphanumeric, non-separator) with underscore
        3. Collapse consecutive separators (., _, -) into a single separator
        4. Strip leading and trailing separators
        5. Truncate to max length (255) with hash suffix if necessary

    The hash suffix ensures uniqueness when truncation is required. The suffix
    format is `_<8-char-hash>` appended after truncating to fit within 255 chars.

    Args:
        value: The input string to normalize.

    Returns:
        A normalized string safe for use as a Kafka consumer group ID.

    Raises:
        ValueError: If the input is empty or results in an empty string after
            normalization.

    Example:
        >>> normalize_kafka_identifier("My Service!!")
        'my_service'
        >>> normalize_kafka_identifier("foo..bar__baz")
        'foo.bar_baz'
        >>> normalize_kafka_identifier("  UPPER_Case-Test  ")
        'upper_case-test'
        >>> normalize_kafka_identifier("valid.consumer-group_id")
        'valid.consumer-group_id'
        >>> normalize_kafka_identifier("@#$%^&*()")  # Raises ValueError
        Traceback (most recent call last):
            ...
        ValueError: Input '@#$%^&*()' results in empty string after normalization
    """
    if not value:
        raise ValueError("Kafka consumer group ID cannot be empty")

    # Step 1: Lowercase
    result = value.lower()

    # Step 2: Replace invalid characters with underscore
    result = _INVALID_CHAR_PATTERN.sub("_", result)

    # Step 3: Collapse consecutive separators, preserving the first separator type
    result = _CONSECUTIVE_SEPARATOR_PATTERN.sub(lambda m: m.group(0)[0], result)

    # Step 4: Strip leading and trailing separators
    result = _EDGE_SEPARATOR_PATTERN.sub("", result)

    # Check for empty result after normalization
    if not result:
        raise ValueError(f"Input {value!r} results in empty string after normalization")

    # Step 5: Truncate with hash suffix if exceeds max length
    if len(result) > KAFKA_CONSUMER_GROUP_MAX_LENGTH:
        # Generate 8-character hash suffix from original value for determinism
        hash_suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
        # Reserve space for underscore + hash suffix (9 chars total)
        max_prefix_length = KAFKA_CONSUMER_GROUP_MAX_LENGTH - 9
        result = f"{result[:max_prefix_length]}_{hash_suffix}"

    return result


def compute_consumer_group_id(
    identity: ModelNodeIdentity,
    purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
) -> str:
    """Compute canonical Kafka consumer group ID from node identity.

    Generates a deterministic, Kafka-compliant consumer group ID using the
    canonical format: ``{env}.{service}.{node_name}.{purpose}.{version}``

    Each component is normalized using ``normalize_kafka_identifier()`` to ensure
    the result is safe for use as a Kafka consumer group ID. The final result
    is validated against Kafka's 255 character limit.

    Args:
        identity: Node identity containing env, service, node_name, and version.
        purpose: Consumer group purpose classification. Defaults to CONSUME.
            The purpose determines consumer behavior semantics (e.g., offset
            reset policy) and is included in the group ID for disambiguation.

    Returns:
        A canonical consumer group ID in the format:
        ``{env}.{service}.{node_name}.{purpose}.{version}``

        If the combined length exceeds Kafka's 255 character limit, the result
        is truncated with an 8-character hash suffix to preserve uniqueness
        while fitting within the constraint.

    Example:
        >>> from omnibase_infra.models import ModelNodeIdentity
        >>> from omnibase_infra.enums import EnumConsumerGroupPurpose
        >>> identity = ModelNodeIdentity(
        ...     env="dev",
        ...     service="omniintelligence",
        ...     node_name="claude_hook_event_effect",
        ...     version="v1",
        ... )
        >>> compute_consumer_group_id(identity)
        'dev.omniintelligence.claude_hook_event_effect.consume.v1'

        With a different purpose:

        >>> compute_consumer_group_id(identity, EnumConsumerGroupPurpose.INTROSPECTION)
        'dev.omniintelligence.claude_hook_event_effect.introspection.v1'

        Component normalization is applied automatically:

        >>> identity_mixed = ModelNodeIdentity(
        ...     env="DEV",
        ...     service="Omni Intelligence",
        ...     node_name="claude-hook-event-effect",
        ...     version="V1.0.0",
        ... )
        >>> compute_consumer_group_id(identity_mixed)
        'dev.omni_intelligence.claude-hook-event-effect.consume.v1.0.0'

        Long identities are automatically truncated with a hash suffix:

        >>> long_identity = ModelNodeIdentity(
        ...     env="development",
        ...     service="a" * 100,
        ...     node_name="b" * 100,
        ...     version="v1",
        ... )
        >>> result = compute_consumer_group_id(long_identity)
        >>> len(result) <= 255
        True
        >>> result.endswith("_" + result[-8:])  # Has hash suffix
        True

    Note:
        The canonical format uses period (.) as the separator between components.
        This enables hierarchical grouping and filtering in Kafka tooling while
        maintaining compatibility with Kafka's consumer group ID constraints.

    See Also:
        - :func:`normalize_kafka_identifier`: Component normalization rules
        - :class:`~omnibase_infra.enums.EnumConsumerGroupPurpose`: Purpose values
        - :class:`~omnibase_infra.models.ModelNodeIdentity`: Identity model

    .. versionadded:: 0.2.6
        Created as part of OMN-1602.
    """
    # Normalize each component
    normalized_env = normalize_kafka_identifier(identity.env)
    normalized_service = normalize_kafka_identifier(identity.service)
    normalized_node_name = normalize_kafka_identifier(identity.node_name)
    normalized_purpose = normalize_kafka_identifier(purpose.value)
    normalized_version = normalize_kafka_identifier(identity.version)

    # Join with period separator
    group_id = ".".join(
        [
            normalized_env,
            normalized_service,
            normalized_node_name,
            normalized_purpose,
            normalized_version,
        ]
    )

    # Handle length constraint with truncation + hash (same strategy as normalize_kafka_identifier)
    # This can occur when multiple long components combine, even though each was individually
    # truncated to 255 chars by normalize_kafka_identifier.
    if len(group_id) > KAFKA_CONSUMER_GROUP_MAX_LENGTH:
        # Generate deterministic hash from original (pre-normalized) identity components
        # to ensure the same identity always produces the same truncated group ID
        hash_input = (
            f"{identity.env}|{identity.service}|{identity.node_name}|"
            f"{purpose.value}|{identity.version}"
        )
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        # Reserve space for underscore + hash suffix (9 chars total)
        max_prefix_length = KAFKA_CONSUMER_GROUP_MAX_LENGTH - 9
        group_id = f"{group_id[:max_prefix_length]}_{hash_suffix}"

    return group_id


__all__: list[str] = [
    "KAFKA_CONSUMER_GROUP_MAX_LENGTH",
    "compute_consumer_group_id",
    "normalize_kafka_identifier",
]
