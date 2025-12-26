# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DLQ Replay Record Model.

This module provides the Pydantic model for DLQ replay records,
representing the state of a message replay operation stored in PostgreSQL.

Related:
    - scripts/dlq_replay.py - CLI tool that uses this model
    - OMN-1032 - PostgreSQL tracking integration ticket
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus


class ModelDlqReplayRecord(BaseModel):
    """Record of a DLQ message replay attempt.

    This model represents a single replay attempt stored in PostgreSQL,
    tracking both the original message context and the replay outcome.

    Table Schema:
        CREATE TABLE IF NOT EXISTS dlq_replay_history (
            id UUID PRIMARY KEY,
            original_message_id UUID NOT NULL,
            replay_correlation_id UUID NOT NULL,
            original_topic VARCHAR(255) NOT NULL,
            target_topic VARCHAR(255) NOT NULL,
            replay_status VARCHAR(50) NOT NULL,
            replay_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            success BOOLEAN NOT NULL,
            error_message TEXT,
            dlq_offset BIGINT NOT NULL,
            dlq_partition INTEGER NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0
        );

    Attributes:
        id: Unique identifier for this replay record (primary key).
        original_message_id: Correlation ID from the original DLQ message.
            This links the replay to the original failed message.
        replay_correlation_id: New correlation ID assigned during replay.
            Used for tracing the replayed message through the system.
        original_topic: The Kafka topic where the message originally failed.
        target_topic: The topic where the message was replayed to.
            Usually same as original_topic, but may differ for rerouting.
        replay_status: Current status of the replay operation.
        replay_timestamp: When the replay was attempted (UTC).
        success: Whether the replay was successful.
            True for COMPLETED status, False for FAILED/SKIPPED.
        error_message: Error details if replay failed. None for success.
        dlq_offset: Kafka offset of the message in the DLQ topic.
        dlq_partition: Kafka partition of the message in the DLQ topic.
        retry_count: Number of times this message has been retried.
            Includes previous attempts before this replay.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> record = ModelDlqReplayRecord(
        ...     id=uuid4(),
        ...     original_message_id=uuid4(),
        ...     replay_correlation_id=uuid4(),
        ...     original_topic="dev.orders.command.v1",
        ...     target_topic="dev.orders.command.v1",
        ...     replay_status=EnumReplayStatus.COMPLETED,
        ...     replay_timestamp=datetime.now(timezone.utc),
        ...     success=True,
        ...     dlq_offset=12345,
        ...     dlq_partition=0,
        ...     retry_count=1,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    id: UUID = Field(
        description="Unique identifier for this replay record (primary key)",
    )
    original_message_id: UUID = Field(
        description="Correlation ID from the original DLQ message",
    )
    replay_correlation_id: UUID = Field(
        description="New correlation ID assigned during replay",
    )
    original_topic: str = Field(
        description="The Kafka topic where the message originally failed",
        min_length=1,
        max_length=255,
    )
    target_topic: str = Field(
        description="The topic where the message was replayed to",
        min_length=1,
        max_length=255,
    )
    replay_status: EnumReplayStatus = Field(
        description="Current status of the replay operation",
    )
    replay_timestamp: datetime = Field(
        description="When the replay was attempted (UTC)",
    )
    success: bool = Field(
        description="Whether the replay was successful",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if replay failed, None for success",
    )
    dlq_offset: int = Field(
        description="Kafka offset of the message in the DLQ topic",
        ge=0,
    )
    dlq_partition: int = Field(
        description="Kafka partition of the message in the DLQ topic",
        ge=0,
    )
    retry_count: int = Field(
        default=0,
        description="Number of times this message has been retried",
        ge=0,
    )


__all__: list[str] = ["ModelDlqReplayRecord"]
