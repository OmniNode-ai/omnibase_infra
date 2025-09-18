"""Kafka operation type enumeration."""

from enum import Enum


class EnumKafkaOperationType(str, Enum):
    """Kafka operation type enumeration."""

    PRODUCE = "produce"
    CONSUME = "consume"
    TOPIC_CREATE = "topic_create"
    TOPIC_DELETE = "topic_delete"
    HEALTH_CHECK = "health_check"
    CONNECTION_TEST = "connection_test"
