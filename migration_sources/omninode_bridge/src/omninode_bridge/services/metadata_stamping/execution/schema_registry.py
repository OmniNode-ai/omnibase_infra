"""
Schema registry for O.N.E. v0.1 protocol.

This module provides schema registration, versioning, and management
for the schema-first execution framework.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SchemaVersion(BaseModel):
    """Schema version information."""

    version: str
    json_schema: str
    created_at: datetime
    deprecated: bool = False
    metadata: dict[str, Any] = {}


class SchemaRegistry:
    """
    Registry for managing Pydantic schemas.

    Provides versioning, deprecation, and schema evolution support.
    """

    def __init__(self):
        """Initialize schema registry."""
        self.schemas: dict[str, dict[str, SchemaVersion]] = {}
        self.schema_classes: dict[str, type[BaseModel]] = {}

    def register_schema(
        self,
        name: str,
        schema_class: type[BaseModel],
        version: str = "1.0.0",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Register a Pydantic schema.

        Args:
            name: Schema name
            schema_class: Pydantic model class
            version: Schema version
            metadata: Optional metadata

        Returns:
            bool: True if registration successful
        """
        try:
            # Generate JSON schema
            json_schema = json.dumps(
                schema_class.model_json_schema(), indent=2, sort_keys=True
            )

            # Initialize schema versions if needed
            if name not in self.schemas:
                self.schemas[name] = {}

            # Create version entry
            self.schemas[name][version] = SchemaVersion(
                version=version,
                json_schema=json_schema,
                created_at=datetime.now(UTC),
                metadata=metadata or {},
            )

            # Store schema class
            schema_key = f"{name}:{version}"
            self.schema_classes[schema_key] = schema_class

            logger.info(f"Registered schema: {name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Schema registration failed: {e}")
            return False

    def get_schema(
        self, name: str, version: Optional[str] = None
    ) -> Optional[SchemaVersion]:
        """
        Get schema by name and version.

        Args:
            name: Schema name
            version: Schema version (latest if not specified)

        Returns:
            SchemaVersion: Schema version info or None
        """
        if name not in self.schemas:
            return None

        if version:
            return self.schemas[name].get(version)

        # Return latest non-deprecated version
        versions = self.schemas[name]
        if not versions:
            return None

        # Sort versions and find latest non-deprecated
        sorted_versions = sorted(versions.items(), key=lambda x: x[0], reverse=True)

        for ver, schema_ver in sorted_versions:
            if not schema_ver.deprecated:
                return schema_ver

        # If all deprecated, return latest anyway
        return sorted_versions[0][1] if sorted_versions else None

    def get_schema_class(
        self, name: str, version: Optional[str] = None
    ) -> Optional[type[BaseModel]]:
        """
        Get schema class by name and version.

        Args:
            name: Schema name
            version: Schema version

        Returns:
            Type[BaseModel]: Schema class or None
        """
        if version:
            schema_key = f"{name}:{version}"
            return self.schema_classes.get(schema_key)

        # Find latest version
        schema_version = self.get_schema(name, version)
        if schema_version:
            schema_key = f"{name}:{schema_version.version}"
            return self.schema_classes.get(schema_key)

        return None

    def list_schemas(self) -> dict[str, list[str]]:
        """
        List all schemas with their versions.

        Returns:
            dict: Map of schema names to version lists
        """
        return {name: list(versions.keys()) for name, versions in self.schemas.items()}

    def list_active_schemas(self) -> dict[str, list[str]]:
        """
        List non-deprecated schemas with their versions.

        Returns:
            dict: Map of schema names to active version lists
        """
        active = {}
        for name, versions in self.schemas.items():
            active_versions = [
                ver for ver, schema_ver in versions.items() if not schema_ver.deprecated
            ]
            if active_versions:
                active[name] = active_versions
        return active

    def deprecate_schema(self, name: str, version: str) -> bool:
        """
        Mark schema version as deprecated.

        Args:
            name: Schema name
            version: Schema version

        Returns:
            bool: True if deprecation successful
        """
        if name in self.schemas and version in self.schemas[name]:
            self.schemas[name][version].deprecated = True
            logger.info(f"Deprecated schema: {name} v{version}")
            return True

        logger.warning(f"Schema not found: {name} v{version}")
        return False

    def get_schema_evolution(self, name: str) -> list[dict[str, Any]]:
        """
        Get schema evolution history.

        Args:
            name: Schema name

        Returns:
            list: Evolution history
        """
        if name not in self.schemas:
            return []

        evolution = []
        sorted_versions = sorted(self.schemas[name].items(), key=lambda x: x[0])

        for version, schema_version in sorted_versions:
            evolution.append(
                {
                    "version": version,
                    "created_at": schema_version.created_at.isoformat(),
                    "deprecated": schema_version.deprecated,
                    "metadata": schema_version.metadata,
                }
            )

        return evolution

    def validate_data(
        self, name: str, data: dict[str, Any], version: Optional[str] = None
    ) -> tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate data against schema.

        Args:
            name: Schema name
            data: Data to validate
            version: Schema version

        Returns:
            tuple: (is_valid, validated_model, error_message)
        """
        schema_class = self.get_schema_class(name, version)
        if not schema_class:
            return False, None, f"Schema not found: {name}"

        try:
            validated = schema_class(**data)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

    def export_schemas(self) -> dict[str, Any]:
        """
        Export all schemas as JSON.

        Returns:
            dict: Exported schemas
        """
        export: dict[str, Any] = {}
        for name, versions in self.schemas.items():
            export[name] = {}
            for version, schema_version in versions.items():
                export[name][version] = {
                    "schema": json.loads(schema_version.json_schema),
                    "created_at": schema_version.created_at.isoformat(),
                    "deprecated": schema_version.deprecated,
                    "metadata": schema_version.metadata,
                }
        return export

    def import_schema(
        self,
        name: str,
        json_schema_str: str,
        version: str = "1.0.0",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Import schema from JSON.

        Args:
            name: Schema name
            json_schema_str: JSON schema string
            version: Schema version
            metadata: Optional metadata

        Returns:
            bool: True if import successful
        """
        try:
            # Validate JSON
            json.loads(json_schema_str)

            # Initialize schema versions if needed
            if name not in self.schemas:
                self.schemas[name] = {}

            # Create version entry
            self.schemas[name][version] = SchemaVersion(
                version=version,
                json_schema=json_schema_str,
                created_at=datetime.now(UTC),
                metadata=metadata or {},
            )

            logger.info(f"Imported schema: {name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Schema import failed: {e}")
            return False


# Global schema registry instance
schema_registry = SchemaRegistry()
