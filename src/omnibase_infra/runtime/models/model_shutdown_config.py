# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shutdown Configuration Model.

This module provides the Pydantic model for shutdown configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelShutdownConfig(BaseModel):
    """Shutdown configuration model.

    Defines graceful shutdown parameters.

    Attributes:
        grace_period_seconds: Time in seconds to wait for graceful shutdown.
            Must be >= 0. A value of 0 means immediate shutdown with no grace
            period (use with caution as in-flight operations may be interrupted).

    Edge Cases:
        - 0: Immediate shutdown, no waiting for in-flight operations
        - Values > 3600: Rejected by Pydantic validation (le=3600 constraint);
          consider using reasonable timeouts (30-120 seconds recommended)
        - Negative values: Rejected by Pydantic validation (ge=0 constraint)

    Production Recommendation:
        Set grace_period_seconds between 30-120 seconds for production deployments
        to allow sufficient time for in-flight operations while preventing excessive
        delays during shutdown sequences.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    grace_period_seconds: int = Field(
        default=30,
        ge=0,
        le=3600,  # Max 1 hour to prevent accidental excessive delays
        description="Time in seconds to wait for graceful shutdown (0-3600)",
    )


__all__: list[str] = ["ModelShutdownConfig"]
