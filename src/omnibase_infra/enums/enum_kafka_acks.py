"""Kafka acknowledgment enumeration for producer configuration."""

from enum import Enum


class EnumKafkaAcks(str, Enum):
    """Enumeration for Kafka acknowledgment modes."""

    NONE = "0"
    LEADER = "1"
    ALL = "all"
