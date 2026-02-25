# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Output model for NodeRegistryApiEffect operations.

Ticket: OMN-1441
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelRegistryApiResponse(BaseModel):
    """Output envelope for registry API effect operations.

    Generic response envelope.  The ``data`` field carries operation-specific
    payloads; ``warnings`` surfaces partial-success conditions.

    Attributes:
        operation: Echo of the requested operation.
        correlation_id: Distributed tracing identifier.
        success: Whether the operation completed without fatal errors.
        data: Operation-specific payload (arbitrary JSON-serialisable dict).
        warnings: Non-fatal issues encountered (e.g., one backend unavailable).
        error: Sanitised error message on failure.
    """

    operation: str = Field(description="Echoed operation identifier.")
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing.",
    )
    success: bool = Field(
        description="Whether the operation completed without fatal errors."
    )
    # ONEX_EXCLUDE: any_type - response data is a generic JSON envelope; values are operation-specific
    data: dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific payload."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal issues encountered."
    )
    error: str | None = Field(
        default=None, description="Sanitised error on fatal failure."
    )

    model_config = {"extra": "forbid", "frozen": True}


__all__ = ["ModelRegistryApiResponse"]
