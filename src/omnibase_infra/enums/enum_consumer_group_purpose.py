# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consumer group purpose enumeration.

This module defines the purpose classification for Kafka consumer groups,
enabling semantic differentiation of consumer behavior and offset policies.

Consumer Group Purpose Categories:
    - CONSUME: Standard event consumption (default behavior)
    - INTROSPECTION: Node introspection and discovery operations
    - REPLAY: Reprocess historical data from earliest offset
    - AUDIT: Compliance and read-only consumption
    - BACKFILL: One-shot bounded consumers for populating derived state

The purpose determines consumer group naming conventions and default
offset reset policies in the Kafka adapter layer.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.

See Also:
    - ModelKafkaConsumerConfig: Consumer configuration using this enum
    - docs/decisions/adr-consumer-group-naming.md: Naming conventions
"""

from __future__ import annotations

from enum import Enum


class EnumConsumerGroupPurpose(str, Enum):
    """Consumer group purpose classification for Kafka consumers.

    Defines the semantic purpose of a consumer group, which influences:
    - Consumer group naming conventions (suffix added to group ID)
    - Default offset reset policy (earliest vs latest)
    - Expected consumption patterns (continuous vs bounded)

    Values:
        CONSUME: Standard event consumption for normal processing.
            - Default offset reset: latest
            - Naming: {base_group_id}-consume
            - Pattern: Continuous consumption

        INTROSPECTION: Node introspection and service discovery.
            - Default offset reset: latest
            - Naming: {base_group_id}-introspection
            - Pattern: Targeted discovery queries

        REPLAY: Reprocess historical data from the beginning.
            - Default offset reset: earliest
            - Naming: {base_group_id}-replay
            - Pattern: Full topic reprocessing

        AUDIT: Compliance and read-only consumption.
            - Default offset reset: earliest
            - Naming: {base_group_id}-audit
            - Pattern: Full audit trail capture

        BACKFILL: One-shot bounded consumers for derived state.
            - Default offset reset: earliest
            - Naming: {base_group_id}-backfill
            - Pattern: Bounded consumption until caught up

    Example:
        >>> purpose = EnumConsumerGroupPurpose.REPLAY
        >>> f"order-processor-{purpose.value}"
        'order-processor-replay'
    """

    CONSUME = "consume"
    """Standard event consumption (default)."""

    INTROSPECTION = "introspection"
    """Node introspection and discovery."""

    REPLAY = "replay"
    """Reprocess historical data from earliest offset."""

    AUDIT = "audit"
    """Compliance/read-only consumption."""

    BACKFILL = "backfill"
    """One-shot bounded consumers for populating derived state."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumConsumerGroupPurpose"]
