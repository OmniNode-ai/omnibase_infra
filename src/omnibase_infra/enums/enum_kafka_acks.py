# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka producer acknowledgment configuration enumeration.

This module defines the acknowledgment policy enumeration for Kafka producers,
with a conversion method to produce aiokafka-compatible types.

The `acks` configuration controls how many partition replicas must acknowledge
receipt before the producer considers a request complete:
    - ALL ("all"): Wait for all in-sync replicas to acknowledge (most durable)
    - LEADER ("1"): Wait only for the leader to acknowledge (balanced)
    - NONE ("0"): Don't wait for any acknowledgment (fastest, least durable)
    - ALL_REPLICAS ("-1"): Explicit numeric form of ALL (equivalent to "all")

See Also:
    - EventBusKafka: Uses this enum for producer configuration
    - ModelKafkaEventBusConfig: Config model that stores the acks setting
    - https://kafka.apache.org/documentation/#producerconfigs_acks
"""

from __future__ import annotations

from enum import Enum

# Explicit mapping for aiokafka type conversion.
# Defined at module level because str-based enums don't support ClassVar attributes
# (attribute lookup on str enum members uses string indexing, not class attribute lookup).
_AIOKAFKA_MAP: dict[str, int | str] = {
    "all": "all",
    "0": 0,
    "1": 1,
    "-1": -1,
}


class EnumKafkaAcks(str, Enum):
    """Kafka producer acknowledgment policy.

    Defines the acknowledgment behavior for Kafka producers. The aiokafka
    library expects specific types for the `acks` parameter:
    - "all" must be passed as a string
    - Numeric values (0, 1, -1) must be passed as integers

    Use the `to_aiokafka()` method to get the correctly-typed value for
    passing to AIOKafkaProducer.

    Attributes:
        ALL: Wait for all in-sync replicas (string "all")
        NONE: Fire and forget, no acknowledgment (string "0" -> int 0)
        LEADER: Wait for leader acknowledgment only (string "1" -> int 1)
        ALL_REPLICAS: Explicit numeric form of ALL (string "-1" -> int -1)

    Note:
        ALL ("all") and ALL_REPLICAS (-1) are semantically equivalent in Kafka.
        ALL_REPLICAS is provided for explicit numeric configuration compatibility
        when migrating from systems that use the numeric form.

    Example:
        >>> acks = EnumKafkaAcks.ALL
        >>> acks.to_aiokafka()
        'all'
        >>> acks = EnumKafkaAcks.LEADER
        >>> acks.to_aiokafka()
        1
    """

    ALL = "all"
    NONE = "0"
    LEADER = "1"
    ALL_REPLICAS = "-1"

    def to_aiokafka(self) -> int | str:
        """Convert to aiokafka-compatible type.

        aiokafka's AIOKafkaProducer expects:
        - The string "all" for all-replica acknowledgment
        - Integer values (0, 1, -1) for numeric ack levels

        Returns:
            The acks value in the format expected by aiokafka:
            - "all" (str) for ALL
            - 0 (int) for NONE
            - 1 (int) for LEADER
            - -1 (int) for ALL_REPLICAS

        Example:
            >>> EnumKafkaAcks.ALL.to_aiokafka()
            'all'
            >>> EnumKafkaAcks.NONE.to_aiokafka()
            0
            >>> EnumKafkaAcks.LEADER.to_aiokafka()
            1
            >>> EnumKafkaAcks.ALL_REPLICAS.to_aiokafka()
            -1
        """
        return _AIOKAFKA_MAP[self.value]


__all__: list[str] = ["EnumKafkaAcks"]
