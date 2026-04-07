# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handshake retry and timeout configuration model (OMN-7657)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelHandshakeConfig(BaseModel):
    """Configuration for handshake retry and timeout behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retry attempts after initial failure",
    )
    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
        description="Delay between retry attempts in seconds",
    )
    total_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Overall deadline for all handshake attempts",
    )
