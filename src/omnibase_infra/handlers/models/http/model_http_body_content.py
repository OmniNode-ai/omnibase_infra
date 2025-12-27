# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Body Content Model.

This module provides a model to encapsulate HTTP request body content,
grouping the raw body and optional pre-serialized bytes together.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelHttpBodyContent(BaseModel):
    """Encapsulates HTTP request body content for request preparation.

    Groups the raw body and optional pre-serialized bytes together to reduce
    function parameter count while maintaining semantic clarity.

    Attributes:
        body: The request body content (str, dict, bytes, or None).
            For POST requests, this is the payload to send.
        pre_serialized: Pre-serialized JSON bytes for dict bodies.
            When provided, avoids double serialization by reusing cached bytes
            from size validation.
    """

    model_config = ConfigDict(
        strict=False,  # Allow object type for body
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow bytes type
    )

    body: object = Field(
        default=None,
        description="Request body content (str, dict, bytes, or None)",
    )
    pre_serialized: bytes | None = Field(
        default=None,
        description="Pre-serialized JSON bytes for dict bodies",
    )


__all__: list[str] = ["ModelHttpBodyContent"]
