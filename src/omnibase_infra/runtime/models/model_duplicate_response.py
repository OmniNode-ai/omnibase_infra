# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Duplicate Response Model.

This module provides the Pydantic model for duplicate message detection responses.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDuplicateResponse(BaseModel):
    """Response indicating a duplicate message was detected.

    This is NOT an error - duplicates are expected under at-least-once delivery.
    The response indicates successful deduplication, informing the caller that
    the message was already processed and no further action is needed.

    Attributes:
        success: Always True - deduplication is successful behavior.
        status: Always "duplicate" to indicate the response type.
        message: Human-readable description of the deduplication.
        message_id: UUID of the duplicate message that was detected.
        correlation_id: Correlation ID for distributed tracing.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    success: Literal[True] = Field(
        default=True,
        description="Deduplication is successful behavior",
    )
    status: Literal["duplicate"] = Field(
        default="duplicate",
        description="Response type indicator",
    )
    message: str = Field(
        default="Message already processed",
        description="Human-readable deduplication message",
    )
    message_id: UUID = Field(description="UUID of the duplicate message")
    correlation_id: UUID = Field(description="Correlation ID for tracing")


__all__: list[str] = ["ModelDuplicateResponse"]
