# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hook execution result model (OMN-7655)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLifecycleHookResult(BaseModel):
    """Structured result from a lifecycle hook execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hook_name: str = Field(
        ...,
        min_length=1,
        description="Lifecycle phase name (on_start, validate_handshake, on_shutdown)",
    )
    success: bool = Field(
        ...,
        description="Whether the hook completed successfully",
    )
    error_message: str = Field(
        default="",
        description="Diagnostic message if the hook failed",
    )
    background_workers: list[str] = Field(
        default_factory=list,
        description="Names of background tasks started by this hook",
    )

    @classmethod
    def succeeded(
        cls,
        hook_name: str,
        background_workers: list[str] | None = None,
    ) -> ModelLifecycleHookResult:
        """Create a successful hook result."""
        return cls(
            hook_name=hook_name,
            success=True,
            background_workers=background_workers or [],
        )

    @classmethod
    def failed(cls, hook_name: str, error_message: str) -> ModelLifecycleHookResult:
        """Create a failed hook result."""
        return cls(
            hook_name=hook_name,
            success=False,
            error_message=error_message,
        )

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success
