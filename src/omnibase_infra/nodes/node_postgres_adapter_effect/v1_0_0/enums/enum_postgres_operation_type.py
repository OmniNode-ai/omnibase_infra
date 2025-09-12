"""PostgreSQL operation type enumeration."""

from enum import Enum


class EnumPostgresOperationType(str, Enum):
    """PostgreSQL operation type enumeration."""
    
    QUERY = "query"
    HEALTH_CHECK = "health_check"
    CONNECTION_TEST = "connection_test"