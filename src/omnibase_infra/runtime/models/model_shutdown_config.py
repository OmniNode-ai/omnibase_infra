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
        grace_period_seconds: Time in seconds to wait for graceful shutdown
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
        description="Time in seconds to wait for graceful shutdown",
    )


__all__: list[str] = ["ModelShutdownConfig"]
