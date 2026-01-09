# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector schema validator protocol.

Provides the protocol definition for schema validators that validate
projection table schemas exist and are correctly structured.

Part of OMN-1168: ProjectorPluginLoader contract discovery loading.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProtocolProjectorSchemaValidator(Protocol):
    """Protocol for projector schema validation.

    Defines the interface that schema validators must implement to validate
    projection table schemas exist and are correctly structured.

    Note:
        Renamed from ProtocolProjectorSchemaManager to follow ONEX naming
        conventions (validators validate, managers manage lifecycle).

    See: tests/unit/runtime/test_projector_schema_manager.py for TDD tests.
    """

    async def ensure_schema_exists(
        self,
        schema: Any,
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


# Backward compatibility alias (deprecated)
ProtocolProjectorSchemaManager = ProtocolProjectorSchemaValidator


__all__ = [
    "ProtocolProjectorSchemaManager",  # Deprecated alias
    "ProtocolProjectorSchemaValidator",
]
