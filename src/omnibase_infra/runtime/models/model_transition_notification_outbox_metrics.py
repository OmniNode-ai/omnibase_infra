# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transition Notification Outbox Metrics Model.

This module provides a strongly-typed Pydantic model for outbox metrics,
replacing the untyped dict return from get_metrics().
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelTransitionNotificationOutboxMetrics(BaseModel):
    """Metrics for transition notification outbox operation.

    This model provides type-safe access to outbox metrics for observability
    and monitoring purposes.

    Attributes:
        table_name: The outbox table name.
        is_running: Whether the background processor is currently running.
        notifications_stored: Total count of notifications stored in outbox.
        notifications_processed: Total count of notifications successfully processed.
        notifications_failed: Total count of notifications that failed processing.
        batch_size: Configured batch size for processing.
        poll_interval_seconds: Configured poll interval in seconds.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    table_name: str = Field(..., description="The outbox table name")
    is_running: bool = Field(default=False, description="Whether processor is running")
    notifications_stored: int = Field(
        default=0, ge=0, description="Total notifications stored"
    )
    notifications_processed: int = Field(
        default=0, ge=0, description="Total notifications successfully processed"
    )
    notifications_failed: int = Field(
        default=0, ge=0, description="Total notifications that failed processing"
    )
    batch_size: int = Field(default=100, ge=1, description="Configured batch size")
    poll_interval_seconds: float = Field(
        default=1.0, gt=0, description="Configured poll interval"
    )


__all__: list[str] = ["ModelTransitionNotificationOutboxMetrics"]
