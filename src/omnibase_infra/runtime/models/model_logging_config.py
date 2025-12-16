# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Logging Configuration Model.

This module provides the Pydantic model for logging configuration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelLoggingConfig(BaseModel):
    """Logging configuration model.

    Defines logging level and format for the runtime.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        description="Log message format string",
    )


__all__: list[str] = ["ModelLoggingConfig"]
