# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Persistence result model for NodeContractPersistenceEffect.

Related:
    - OMN-1845: NodeContractPersistenceEffect implementation
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumPostgresErrorCode


class ModelPersistenceResult(BaseModel):
    """Result of a contract persistence operation.

    Attributes:
        success: Whether the operation succeeded.
        error: Error message if operation failed (sanitized).
        error_code: Typed error code for programmatic handling. Uses
            EnumPostgresErrorCode for strong typing and validation.
        duration_ms: Operation duration in milliseconds.
        correlation_id: Correlation ID for distributed tracing.
        rows_affected: Number of database rows affected.
        timestamp: When the operation completed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(..., description="Whether the operation succeeded.")
    error: str | None = Field(default=None, description="Sanitized error message.")
    error_code: EnumPostgresErrorCode | None = Field(
        default=None,
        description="Typed error code for programmatic handling.",
    )
    duration_ms: float = Field(default=0.0, description="Operation duration in ms.")
    correlation_id: UUID | None = Field(
        default=None, description="Correlation ID for tracing."
    )
    rows_affected: int = Field(default=0, description="Database rows affected.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Operation completion time.",
    )


__all__ = ["ModelPersistenceResult"]
