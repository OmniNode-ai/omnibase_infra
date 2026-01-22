# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for TransitionNotificationOutbox.

This module provides a Pydantic model for configuring the TransitionNotificationOutbox,
which implements the outbox pattern for guaranteed notification delivery of state
transition events.

Related:
    - TransitionNotificationOutbox: The outbox implementation
    - ModelTransitionNotificationOutboxMetrics: Metrics model for observability
    - docs/patterns/retry_backoff_compensation_strategy.md: Outbox pattern docs

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelTransitionNotificationOutboxConfig(BaseModel):
    """Configuration for TransitionNotificationOutbox.

    This model encapsulates all configuration options for the outbox pattern
    implementation that ensures at-least-once delivery semantics for state
    transition notifications.

    The outbox stores notifications in the same database transaction as projections,
    then processes them asynchronously via a background processor.

    Attributes:
        outbox_table: PostgreSQL table name for the outbox.
        batch_size: Maximum notifications to process per batch.
        poll_interval_seconds: Seconds between polls when idle.
        query_timeout_seconds: Timeout for database queries.
        strict_transaction_mode: If True, raises error when store() called
            outside a transaction context.
        shutdown_timeout_seconds: Timeout for graceful shutdown.
        max_retries: Maximum retry attempts before moving to DLQ. None disables DLQ.
        dlq_topic: Topic name for DLQ (for metrics/logging purposes).

    Example:
        >>> config = ModelTransitionNotificationOutboxConfig(
        ...     outbox_table="state_transition_outbox",
        ...     batch_size=50,
        ...     poll_interval_seconds=0.5,
        ...     max_retries=3,
        ...     dlq_topic="notifications-dlq",
        ... )
        >>> outbox = TransitionNotificationOutbox(
        ...     pool=pool,
        ...     publisher=publisher,
        ...     **config.model_dump(),  # Unpack as kwargs
        ... )

    Related:
        - TransitionNotificationOutbox: Uses this configuration
        - ModelTransitionNotificationOutboxMetrics: Runtime metrics
    """

    model_config = ConfigDict(frozen=True)

    outbox_table: str = Field(
        default="transition_notification_outbox",
        min_length=1,
        max_length=63,  # PostgreSQL identifier limit
        description="PostgreSQL table name for the outbox",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum notifications to process per batch",
    )
    poll_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Seconds between background processor polls when idle",
    )
    query_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for database queries",
    )
    strict_transaction_mode: bool = Field(
        default=True,
        description="If True, raises error when store() called outside transaction",
    )
    shutdown_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for graceful shutdown during stop()",
    )
    max_retries: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum retry attempts before moving to DLQ. None disables DLQ",
    )
    dlq_topic: str | None = Field(
        default=None,
        description="Topic name for DLQ (for metrics/logging purposes)",
    )


__all__: list[str] = [
    "ModelTransitionNotificationOutboxConfig",
]
