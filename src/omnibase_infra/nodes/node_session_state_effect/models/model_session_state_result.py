# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for session state effect operations."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelSessionStateResult(BaseModel):
    """Result of a session state filesystem operation.

    Attributes:
        success: Whether the operation completed successfully.
        operation: The operation that was performed.
        correlation_id: Correlation ID for tracing.
        error: Error message if the operation failed.
        error_code: Machine-readable error code.
        files_affected: Number of files read/written/deleted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the operation completed successfully.",
    )
    operation: str = Field(
        ...,
        description="The operation that was performed.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    error: str = Field(
        default="",
        description="Error message (empty if success).",
    )
    error_code: str = Field(
        default="",
        description="Machine-readable error code (empty if success).",
    )
    files_affected: int = Field(
        default=0,
        ge=0,
        description="Number of files read/written/deleted.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``success`` is True. Differs from typical Pydantic behavior.
        """
        return self.success


__all__: list[str] = ["ModelSessionStateResult"]
