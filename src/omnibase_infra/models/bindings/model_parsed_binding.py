# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Parsed binding model with pre-compiled expression components."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from omnibase_core.types import JsonType


class ModelParsedBinding(BaseModel):
    """Binding with pre-parsed expression for fast resolution.

    The loader parses and validates expressions once at load time.
    The resolver just executes path traversal without re-parsing.

    Example:
        Original expression: "${payload.user.id}"
        Parsed:
            source: "payload"
            path_segments: ("user", "id")
    """

    parameter_name: str = Field(
        ...,
        description="Target handler input field name",
    )
    source: Literal["payload", "envelope", "context"] = Field(
        ...,
        description="Data source for binding resolution",
    )
    path_segments: tuple[str, ...] = Field(
        ...,
        description="Pre-parsed path segments for traversal",
    )
    required: bool = Field(
        default=True,
        description="If True, fail fast when field is missing",
    )
    default: JsonType | None = Field(
        default=None,
        description="Default value if not required and missing",
    )
    original_expression: str = Field(
        ...,
        description="Original ${source.path} expression for error messages",
    )

    model_config = {"frozen": True, "extra": "forbid"}
