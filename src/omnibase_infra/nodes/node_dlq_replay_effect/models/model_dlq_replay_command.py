# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed COMMAND model for the contract-native DLQ replay node (OMN-12619).

This is the strongly-typed input command for ``NodeDlqReplayEffect``. It
replaces the argparse ``Namespace`` that drove the legacy CLI
(``scripts/dlq_replay.py``) and is the source of truth for a single replay run.

Eligibility rules (max-retry, non-retryable error types, time-range, topic /
error-type / correlation filters) are NOT reimplemented here — they live in
``should_replay()`` which the node reuses unchanged.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from omnibase_infra.nodes.node_dlq_replay_effect.models.enum_dlq_replay_filter_type import (
    EnumDlqReplayFilterType,
)


class ModelDlqReplayCommand(BaseModel):
    """Typed command requesting a DLQ replay run.

    A single command describes which DLQ topic to drain, which messages are
    in-scope (via filters), and whether the run is a dry run. Non-replayable
    messages are quarantined (never dropped); the command does not toggle that
    behaviour — quarantine is unconditional per OMN-12619.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dlq_topic: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="DLQ source topic to drain (e.g. 'onex.dlq.events.v1').",
    )
    max_replay_count: int = Field(
        default=5,
        gt=0,
        description="Maximum total replay attempts per message for eligibility.",
    )
    rate_limit_per_second: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum messages replayed per second (must be > 0).",
    )
    dry_run: bool = Field(
        default=False,
        description="When True, report eligibility without publishing to topics.",
    )
    filter_type: EnumDlqReplayFilterType = Field(
        default=EnumDlqReplayFilterType.ALL,
        description="Selective replay filter strategy.",
    )
    filter_topics: tuple[str, ...] = Field(
        default=(),
        description="Original topics to include when filter_type is BY_TOPIC.",
    )
    filter_error_types: tuple[str, ...] = Field(
        default=(),
        description="Error types to include when filter_type is BY_ERROR_TYPE.",
    )
    filter_correlation_ids: tuple[UUID, ...] = Field(
        default=(),
        description="Correlation IDs to include when filter_type is BY_CORRELATION_ID.",
    )
    filter_start_time: datetime | None = Field(
        default=None,
        description="Inclusive lower bound on message failure timestamp (UTC).",
    )
    filter_end_time: datetime | None = Field(
        default=None,
        description="Inclusive upper bound on message failure timestamp (UTC).",
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Maximum messages to process in this run (None = unlimited).",
    )

    @field_validator("filter_end_time", mode="after")
    @classmethod
    def _end_after_start(
        cls, value: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        if value is not None and info.data:
            start = info.data.get("filter_start_time")
            if start is not None and value < start:
                raise ValueError(
                    f"filter_end_time ({value}) must not precede "
                    f"filter_start_time ({start})"
                )
        return value


__all__ = ["ModelDlqReplayCommand"]
