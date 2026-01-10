# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector schema validator protocol.

Provides the protocol definition for schema validators that validate
projection table schemas exist and are correctly structured.

Part of OMN-1168: ProjectorPluginLoader contract discovery loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.models.projectors import ModelProjectorSchema


@runtime_checkable
class ProtocolProjectorSchemaValidator(Protocol):
    """Protocol for projector schema validation.

    Defines the interface that schema validators must implement to validate
    projection table schemas exist and are correctly structured.

    See: tests/unit/runtime/test_projector_schema_validator.py for TDD tests.
    """

    async def ensure_schema_exists(
        self,
        schema: ModelProjectorSchema,
        correlation_id: str | None = None,
    ) -> None:
        """Ensure the schema table exists with required columns.

        Verifies that the projection table exists in the database and has
        all required columns. Does NOT auto-create missing schemas.

        Args:
            schema: Projector schema to validate.
            correlation_id: Optional correlation ID for distributed tracing.

        Raises:
            ProjectorSchemaError: If table does not exist or required columns
                are missing.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If validation query times out.
        """
        ...

    async def table_exists(
        self,
        table_name: str,
        schema_name: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.
            schema_name: Optional database schema name. Defaults to 'public'.
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            True if table exists, False otherwise.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
        """
        ...


__all__ = [
    "ProtocolProjectorSchemaValidator",
]
