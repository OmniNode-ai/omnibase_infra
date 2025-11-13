"""
Infrastructure package for omninode_bridge.

Provides core infrastructure components including database connection pooling,
Kafka integration, entity registry for type-safe database operations,
and other foundational services.
"""

from omninode_bridge.infrastructure.entity_registry import EntityRegistry
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType
from omninode_bridge.infrastructure.postgres_connection_manager import (
    ConnectionStats,
    ModelPostgresConfig,
    PostgresConnectionManager,
    QueryMetrics,
)

__all__ = [
    "PostgresConnectionManager",
    "ModelPostgresConfig",
    "ConnectionStats",
    "QueryMetrics",
    "EntityRegistry",
    "EnumEntityType",
]
