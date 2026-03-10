# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Shutdown configuration for ONEX runtime graceful shutdown."""

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

    Defines graceful shutdown parameters including per-handler timeout controls.

    Attributes:
        grace_period_seconds: Time in seconds to wait for overall graceful shutdown.
            Must be >= 0. A value of 0 means immediate shutdown with no grace
            period (use with caution as in-flight operations may be interrupted).
        handler_shutdown_timeout_seconds: Maximum time in seconds to wait for each
            individual handler's shutdown method to complete. If a handler exceeds
            this timeout, the shutdown is considered failed for that handler but
            other handlers continue their shutdown. This prevents a single slow
            handler (e.g., asyncpg pool close with 30s per-connection timeout)
            from blocking the entire shutdown sequence.

    Edge Cases:
        - grace_period_seconds=0: Immediate shutdown, no waiting for in-flight operations
        - Values > 3600: Rejected by Pydantic validation (le=3600 constraint)
        - Negative values: Rejected by Pydantic validation (ge=0 constraint)
        - handler_shutdown_timeout_seconds=0: Not allowed (ge=1 constraint)

    Production Recommendation:
        Set grace_period_seconds between 30-120 seconds for production deployments.
        Set handler_shutdown_timeout_seconds to 10-15 seconds to prevent any single
        handler from consuming the entire grace period. The handler timeout should
        always be less than the overall grace period.
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

    handler_shutdown_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,  # Max 5 minutes per handler
        strict=False,
        description=(
            "Maximum time in seconds to wait for each individual handler shutdown. "
            "Prevents a single slow handler from blocking the entire shutdown sequence."
        ),
    )


__all__: list[str] = ["ModelShutdownConfig"]
