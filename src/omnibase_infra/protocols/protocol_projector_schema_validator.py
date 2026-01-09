# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector schema validator protocol.

Provides the protocol definition for schema validators that validate
projection table schemas exist and are correctly structured.

Part of OMN-1168: ProjectorPluginLoader contract discovery loading.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolProjectorSchemaValidator(Protocol):
    """Protocol for projector schema validation.

    Defines the interface that schema validators must implement to validate
    projection table schemas exist and are correctly structured.

    See: tests/unit/runtime/test_projector_schema_validator.py for TDD tests.
    """

    async def ensure_schema_exists(
        self,
        schema: object,
        correlation_id: str | None = None,
    ) -> None:
        """Ensure the schema table exists with required columns."""
        ...

    async def table_exists(
        self,
        table_name: str,
        schema_name: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Check if a table exists in the database."""
        ...


__all__ = [
    "ProtocolProjectorSchemaValidator",
]
