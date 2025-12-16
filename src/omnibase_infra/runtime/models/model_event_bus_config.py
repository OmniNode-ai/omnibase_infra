# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Configuration Model.

This module provides the Pydantic model for event bus configuration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelEventBusConfig(BaseModel):
    """Event bus configuration model.

    Defines the event bus type and operational parameters.

    Attributes:
        type: Event bus implementation type ('inmemory' or 'kafka')
        environment: Deployment environment name
        max_history: Maximum event history to retain
        circuit_breaker_threshold: Failure count before circuit breaker trips
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    type: Literal["inmemory", "kafka"] = Field(
        default="inmemory",
        description="Event bus implementation type",
    )
    environment: str = Field(
        default="local",
        description="Deployment environment name",
    )
    max_history: int = Field(
        default=1000,
        ge=0,
        description="Maximum event history to retain",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failure count before circuit breaker trips",
    )


__all__: list[str] = ["ModelEventBusConfig"]
