"""
Event schema versioning strategy for omninode_bridge.

Provides version-aware topic naming, schema registry, and migration support
for backward-compatible event schema evolution.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class EventSchemaVersion(str, Enum):
    """Event schema versions."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

    def __str__(self) -> str:
        """String representation."""
        return self.value


class SchemaEvolutionStrategy(str, Enum):
    """Strategies for schema evolution."""

    BACKWARD_COMPATIBLE = "backward_compatible"  # New fields are optional
    FORWARD_COMPATIBLE = "forward_compatible"  # Old fields remain
    BREAKING_CHANGE = "breaking_change"  # Incompatible changes


class SchemaMetadata(BaseModel):
    """Metadata for event schema."""

    version: EventSchemaVersion
    schema_class: type[BaseModel]
    evolution_strategy: SchemaEvolutionStrategy
    deprecated: bool = False
    deprecation_date: Optional[str] = None
    removal_date: Optional[str] = None
    migration_notes: Optional[str] = None


class EventVersionRegistry:
    """Registry for event schema versions."""

    def __init__(self):
        """Initialize event version registry."""
        self._registry: dict[str, dict[EventSchemaVersion, SchemaMetadata]] = {}
        self._migrations: dict[str, dict[tuple, Callable]] = {}
        self._latest_versions: dict[str, EventSchemaVersion] = {}

    def register(
        self,
        event_type: str,
        version: EventSchemaVersion,
        schema_class: type[BaseModel],
        evolution_strategy: SchemaEvolutionStrategy = SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
        deprecated: bool = False,
        deprecation_date: Optional[str] = None,
        removal_date: Optional[str] = None,
        migration_notes: Optional[str] = None,
    ) -> None:
        """Register event schema for version.

        Args:
            event_type: Event type name (e.g., "NODE_GENERATION_REQUESTED")
            version: Schema version
            schema_class: Pydantic model class
            evolution_strategy: Strategy for schema evolution
            deprecated: Whether this version is deprecated
            deprecation_date: ISO date when version was deprecated
            removal_date: ISO date when version will be removed
            migration_notes: Notes for migrating to newer version
        """
        if event_type not in self._registry:
            self._registry[event_type] = {}

        metadata = SchemaMetadata(
            version=version,
            schema_class=schema_class,
            evolution_strategy=evolution_strategy,
            deprecated=deprecated,
            deprecation_date=deprecation_date,
            removal_date=removal_date,
            migration_notes=migration_notes,
        )

        self._registry[event_type][version] = metadata

        # Update latest version
        if not deprecated:
            current_latest = self._latest_versions.get(event_type)
            if current_latest is None or self._version_is_newer(
                version, current_latest
            ):
                self._latest_versions[event_type] = version

    def get_schema(
        self, event_type: str, version: EventSchemaVersion
    ) -> type[BaseModel]:
        """Get schema for event type and version.

        Args:
            event_type: Event type name
            version: Schema version

        Returns:
            Pydantic model class

        Raises:
            KeyError: If event type or version not found
        """
        if event_type not in self._registry:
            raise KeyError(f"Event type '{event_type}' not registered")
        if version not in self._registry[event_type]:
            raise KeyError(
                f"Version '{version}' not found for event type '{event_type}'"
            )

        return self._registry[event_type][version].schema_class

    def get_metadata(
        self, event_type: str, version: EventSchemaVersion
    ) -> SchemaMetadata:
        """Get metadata for event type and version.

        Args:
            event_type: Event type name
            version: Schema version

        Returns:
            Schema metadata

        Raises:
            KeyError: If event type or version not found
        """
        if event_type not in self._registry:
            raise KeyError(f"Event type '{event_type}' not registered")
        if version not in self._registry[event_type]:
            raise KeyError(
                f"Version '{version}' not found for event type '{event_type}'"
            )

        return self._registry[event_type][version]

    def get_latest_version(self, event_type: str) -> EventSchemaVersion:
        """Get latest non-deprecated version for event type.

        Args:
            event_type: Event type name

        Returns:
            Latest schema version

        Raises:
            KeyError: If event type not found
        """
        if event_type not in self._latest_versions:
            raise KeyError(f"Event type '{event_type}' not registered")

        return self._latest_versions[event_type]

    def register_migration(
        self,
        event_type: str,
        from_version: EventSchemaVersion,
        to_version: EventSchemaVersion,
        migration_func: Callable[[dict], dict],
    ) -> None:
        """Register migration function between versions.

        Args:
            event_type: Event type name
            from_version: Source version
            to_version: Target version
            migration_func: Function to migrate data (dict -> dict)
        """
        if event_type not in self._migrations:
            self._migrations[event_type] = {}

        self._migrations[event_type][(from_version, to_version)] = migration_func

    def migrate(
        self,
        event_type: str,
        data: dict[str, Any],
        from_version: EventSchemaVersion,
        to_version: EventSchemaVersion,
    ) -> dict[str, Any]:
        """Migrate event data between versions.

        Args:
            event_type: Event type name
            data: Event data to migrate
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated event data

        Raises:
            KeyError: If migration path not found
        """
        if from_version == to_version:
            return data

        # Direct migration available?
        if event_type in self._migrations:
            migration_key = (from_version, to_version)
            if migration_key in self._migrations[event_type]:
                return self._migrations[event_type][migration_key](data)

        raise KeyError(
            f"No migration path from {from_version} to {to_version} "
            f"for event type '{event_type}'"
        )

    def validate_and_migrate(
        self,
        event_type: str,
        data: dict[str, Any],
        source_version: EventSchemaVersion,
        target_version: Optional[EventSchemaVersion] = None,
    ) -> BaseModel:
        """Validate and optionally migrate event data.

        Args:
            event_type: Event type name
            data: Event data to validate
            source_version: Version of input data
            target_version: Desired output version (default: latest)

        Returns:
            Validated (and migrated) Pydantic model instance

        Raises:
            ValidationError: If data doesn't match schema
            KeyError: If migration path not found
        """
        # Use latest version if not specified
        if target_version is None:
            target_version = self.get_latest_version(event_type)

        # Migrate if needed
        if source_version != target_version:
            data = self.migrate(event_type, data, source_version, target_version)

        # Validate with target schema
        schema_class = self.get_schema(event_type, target_version)
        return schema_class(**data)

    def list_event_types(self) -> list[str]:
        """List all registered event types.

        Returns:
            List of event type names
        """
        return list(self._registry.keys())

    def list_versions(self, event_type: str) -> list[EventSchemaVersion]:
        """List all versions for event type.

        Args:
            event_type: Event type name

        Returns:
            List of versions

        Raises:
            KeyError: If event type not found
        """
        if event_type not in self._registry:
            raise KeyError(f"Event type '{event_type}' not registered")

        return list(self._registry[event_type].keys())

    def is_deprecated(self, event_type: str, version: EventSchemaVersion) -> bool:
        """Check if schema version is deprecated.

        Args:
            event_type: Event type name
            version: Schema version

        Returns:
            True if deprecated, False otherwise

        Raises:
            KeyError: If event type or version not found
        """
        metadata = self.get_metadata(event_type, version)
        return metadata.deprecated

    @staticmethod
    def _version_is_newer(v1: EventSchemaVersion, v2: EventSchemaVersion) -> bool:
        """Check if v1 is newer than v2."""
        version_order = {
            EventSchemaVersion.V1: 1,
            EventSchemaVersion.V2: 2,
            EventSchemaVersion.V3: 3,
        }
        return version_order[v1] > version_order[v2]


# Global registry instance
event_registry = EventVersionRegistry()


def get_topic_name(
    base_name: str,
    version: EventSchemaVersion = EventSchemaVersion.V1,
    environment: str = "dev",
    service: str = "omninode-bridge",
    domain: str = "codegen",
) -> str:
    """Generate topic name with version.

    Args:
        base_name: Base topic name (e.g., "generation-requested")
        version: Schema version
        environment: Environment (dev, staging, prod)
        service: Service name
        domain: Domain name

    Returns:
        Full topic name (e.g., "dev.omninode-bridge.codegen.generation-requested.v1")

    Example:
        >>> get_topic_name("generation-requested", EventSchemaVersion.V1)
        'dev.omninode-bridge.codegen.generation-requested.v1'
    """
    return f"{environment}.{service}.{domain}.{base_name}.{version}"


def parse_topic_name(topic: str) -> dict[str, str]:
    """Parse topic name to extract components.

    Args:
        topic: Full topic name

    Returns:
        Dictionary with environment, service, domain, base_name, version

    Example:
        >>> parse_topic_name("dev.omninode-bridge.codegen.generation-requested.v1")
        {
            'environment': 'dev',
            'service': 'omninode-bridge',
            'domain': 'codegen',
            'base_name': 'generation-requested',
            'version': 'v1'
        }
    """
    parts = topic.split(".")
    if len(parts) < 5:
        raise ValueError(f"Invalid topic name format: {topic}")

    return {
        "environment": parts[0],
        "service": parts[1],
        "domain": parts[2],
        "base_name": ".".join(parts[3:-1]),  # Handle multi-part base names
        "version": parts[-1],
    }
