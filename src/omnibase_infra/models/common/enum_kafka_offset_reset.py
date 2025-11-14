"""
Kafka offset reset enum.

Strongly typed values for Kafka consumer offset reset policies.
"""

from enum import Enum


class EnumKafkaOffsetReset(str, Enum):
    """Strongly typed Kafka offset reset policy values."""

    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


# Export for use
__all__ = ["EnumKafkaOffsetReset"]
