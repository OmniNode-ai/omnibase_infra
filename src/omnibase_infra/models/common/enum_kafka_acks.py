"""
Kafka acknowledgment enum.

Strongly typed values for Kafka producer acknowledgment settings.
"""

from enum import Enum


class EnumKafkaAcks(str, Enum):
    """Strongly typed Kafka acknowledgment values."""

    NONE = "0"  # No acknowledgment
    LEADER = "1"  # Leader acknowledgment only
    ALL = "all"  # All in-sync replicas acknowledgment


# Export for use
__all__ = ["EnumKafkaAcks"]