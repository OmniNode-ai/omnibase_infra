# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shutdown Configuration Model.

This module provides the Pydantic model for shutdown configuration.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _coerce_grace_period(v: object) -> int:
    """Coerce float values to int before strict validation.

    This pre-validator handles the case where a float value like 30.0
    is passed. With strict=True, Pydantic would reject floats, but
    this validator ensures whole-number floats are accepted.

    Note:
        Boolean values are explicitly rejected even though bool is a
        subclass of int in Python. This maintains semantic correctness
        and consistency with other integer configuration fields that
        use strict=True without pre-validators.

    Args:
        v: The input value (may be int, float, or other).

    Returns:
        Integer value if input is a valid whole number.

    Raises:
        ValueError: If float has a fractional part.
        TypeError: If input is not numeric or is a boolean.
    """
    # Explicitly reject booleans first - bool is a subclass of int in Python,
    # so isinstance(True, int) returns True. We must check bool before int
    # to maintain strict type semantics and prevent unexpected coercion.
    if isinstance(v, bool):
        raise TypeError("grace_period_seconds must be an integer, got bool")
    if isinstance(v, float):
        if v != int(v):
            raise ValueError(f"grace_period_seconds must be a whole number, got {v}")
        return int(v)
    if isinstance(v, int):
        return v
    raise TypeError(f"grace_period_seconds must be an integer, got {type(v).__name__}")


# Type alias for grace period with pre-validation coercion
_GracePeriodSeconds = Annotated[int, BeforeValidator(_coerce_grace_period)]


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

    grace_period_seconds: _GracePeriodSeconds = Field(
        default=30,
        ge=0,
        le=3600,  # Max 1 hour to prevent accidental excessive delays
        strict=False,  # Override model-level strict=True to allow BeforeValidator coercion
        description="Time in seconds to wait for graceful shutdown (0-3600)",
    )


__all__: list[str] = ["ModelShutdownConfig"]
