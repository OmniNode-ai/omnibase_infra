# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Warning model for partial success scenarios.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelWarning(BaseModel):
    """Warning message for partial success scenarios.

    Used when one backend succeeds but another fails, allowing the
    API to return partial results with appropriate warnings.

    Attributes:
        source: Source of the warning (e.g., "consul", "postgres")
        message: Human-readable warning message
        code: Optional error code for programmatic handling
        timestamp: When the warning was generated
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(
        ...,
        description="Source of the warning",
    )
    message: str = Field(
        ...,
        description="Human-readable warning message",
    )
    code: str | None = Field(
        default=None,
        description="Optional error code",
    )
    timestamp: datetime = Field(
        ...,
        description="When the warning was generated",
    )


__all__ = ["ModelWarning"]
