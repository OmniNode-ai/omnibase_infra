# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotency Check Result Model.

This module provides the Pydantic model for the result of an idempotency
check operation, indicating whether a message is a duplicate and providing
details about the original processing if applicable.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelIdempotencyCheckResult(BaseModel):
    """Result of an idempotency check operation.

    This model represents the outcome of checking whether a message has
    already been processed. It provides all necessary information for
    the caller to decide how to proceed.

    Attributes:
        is_duplicate: True if the message has already been processed.
            When True, the caller should skip processing this message.
        message_id: The unique identifier of the checked message.
            Echoed back for verification and logging purposes.
        domain: Optional domain/namespace that was checked.
            Matches the domain used in the idempotency check.
        original_processed_at: Timestamp when the message was first processed.
            Only populated when is_duplicate is True. Useful for debugging
            and understanding the timing of duplicate messages.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> # Non-duplicate result
        >>> result = ModelIdempotencyCheckResult(
        ...     is_duplicate=False,
        ...     message_id=uuid4(),
        ...     domain="orders",
        ...     original_processed_at=None,
        ... )
        >>> if not result.is_duplicate:
        ...     print("Processing message...")
        Processing message...

        >>> # Duplicate result
        >>> duplicate_result = ModelIdempotencyCheckResult(
        ...     is_duplicate=True,
        ...     message_id=uuid4(),
        ...     domain="orders",
        ...     original_processed_at=datetime.now(timezone.utc),
        ... )
        >>> if duplicate_result.is_duplicate:
        ...     print(f"Skipping duplicate, first processed at {duplicate_result.original_processed_at}")
        Skipping duplicate, first processed at ...
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    is_duplicate: bool = Field(
        description="True if the message has already been processed",
    )
    message_id: UUID = Field(
        description="Unique identifier of the checked message",
    )
    domain: str | None = Field(
        default=None,
        description="Domain/namespace that was checked",
        max_length=255,
    )
    original_processed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the message was first processed (only if duplicate)",
    )


__all__: list[str] = ["ModelIdempotencyCheckResult"]
