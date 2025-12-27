# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for registration orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelOrchestratorConfig(BaseModel):
    """Configuration for the registration orchestrator node.

    Controls workflow execution parameters including timeouts, retry behavior,
    and failure handling strategies.

    Attributes:
        timeout_seconds: Maximum time allowed for the entire workflow.
        max_retries: Maximum retry attempts for individual effect calls.
        initial_delay_ms: Initial delay before first retry attempt.
        max_delay_ms: Maximum delay between retry attempts.
        exponential_base: Base for exponential backoff calculation.
        fail_fast: If True, stop workflow on first failure; if False, continue.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Workflow timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retry attempts for effect calls",
    )
    initial_delay_ms: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Initial retry delay in milliseconds",
    )
    max_delay_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Max retry delay in milliseconds",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=4.0,
        description="Exponential backoff base multiplier",
    )
    fail_fast: bool = Field(
        default=True,
        description="Stop on first failure if True",
    )


__all__ = ["ModelOrchestratorConfig"]
