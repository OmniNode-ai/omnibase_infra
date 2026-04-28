# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed error envelope for local runtime ingress responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelLocalRuntimeIngressError(BaseModel):
    """Structured error for local runtime ingress responses."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    code: Literal[
        "validation_error",
        "unknown_command",
        "runtime_unavailable",
        "dispatch_timeout",
        "dispatch_error",
    ] = Field(..., description="Stable local ingress error code.")
    message: str = Field(..., min_length=1, description="Human-readable error message.")
    retryable: bool = Field(
        default=False,
        description="Whether callers may retry the same request without mutation.",
    )
    details: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional typed diagnostic details for the caller.",
    )


__all__ = ["ModelLocalRuntimeIngressError"]
