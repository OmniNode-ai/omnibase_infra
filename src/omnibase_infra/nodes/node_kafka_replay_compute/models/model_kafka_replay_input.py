# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for Kafka replay compute."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.replay.enum_replay_mode import EnumReplayMode
from omnibase_infra.models.projection import ModelSequenceInfo


class ModelKafkaReplayInput(BaseModel):
    """Typed command for replaying Kafka event envelopes from an injected cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topics: list[str] = Field(
        ..., min_length=1, description="Kafka topics to replay from the target cluster."
    )
    from_offset: str = Field(
        default="earliest",
        description="Starting offset. Use 'earliest' or a non-negative numeric offset.",
    )
    to_offset: str = Field(
        default="latest",
        description=(
            "Exclusive ending offset. Use 'latest', a non-negative numeric offset, "
            "or an ISO-8601 wall-clock timestamp."
        ),
    )
    target_cluster_bootstrap: str = Field(
        ...,
        min_length=1,
        description="Injected Kafka/Redpanda bootstrap servers for the isolated target.",
    )
    target_consumer_group: str = Field(
        ..., min_length=1, description="Consumer group used for this replay run."
    )
    mode: EnumReplayMode = Field(
        default=EnumReplayMode.REPLAYING,
        description="Replay mode from omnibase_core deterministic replay contracts.",
    )
    expected_event_count: int | None = Field(
        default=None,
        ge=0,
        description="Optional exact event count assertion for the replay run.",
    )
    resume_from: dict[str, ModelSequenceInfo] = Field(
        default_factory=dict,
        description=(
            "Optional per topic-partition resume offsets. Keys use '<topic>:<partition>'."
        ),
    )
    poll_timeout_ms: int = Field(
        default=1000,
        gt=0,
        description="Kafka poll timeout in milliseconds.",
    )
    max_empty_polls: int = Field(
        default=3,
        gt=0,
        description="Consecutive empty polls allowed before the replay run stops.",
    )
    progress_interval_events: int = Field(
        default=100,
        gt=0,
        description="Emit progress after this many replayed events.",
    )

    @field_validator("topics")
    @classmethod
    def _topics_must_be_non_empty_strings(cls, topics: list[str]) -> list[str]:
        cleaned = [topic.strip() for topic in topics]
        if any(not topic for topic in cleaned):
            raise ValueError("topics must not contain empty topic names")
        return cleaned

    @field_validator("from_offset", mode="before")
    @classmethod
    def _from_offset_must_be_earliest_or_numeric(cls, value: object) -> str:
        normalized = str(value).strip()
        if normalized == "earliest" or normalized.isdecimal():
            return normalized
        raise ValueError("from_offset must be 'earliest' or a non-negative integer")

    @field_validator("to_offset", mode="before")
    @classmethod
    def _to_offset_must_be_latest_numeric_or_timestamp(cls, value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        normalized = str(value).strip()
        if normalized == "latest" or normalized.isdecimal():
            return normalized
        try:
            datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                "to_offset must be 'latest', a non-negative integer, "
                "or an ISO-8601 timestamp"
            ) from exc
        return normalized

    @field_validator("target_cluster_bootstrap")
    @classmethod
    def _bootstrap_must_be_explicit_target(cls, bootstrap: str) -> str:
        cleaned = bootstrap.strip()
        if not cleaned:
            raise ValueError("target_cluster_bootstrap must not be empty")
        if ".201:" in cleaned or cleaned.endswith(".201"):
            raise ValueError(
                "target_cluster_bootstrap points at protected .201 runtime; "
                "use an isolated target cluster"
            )
        return cleaned

    @field_validator("target_consumer_group")
    @classmethod
    def _consumer_group_must_be_explicit(cls, group: str) -> str:
        cleaned = group.strip()
        if not cleaned:
            raise ValueError("target_consumer_group must not be empty")
        return cleaned


__all__ = ["ModelKafkaReplayInput"]
