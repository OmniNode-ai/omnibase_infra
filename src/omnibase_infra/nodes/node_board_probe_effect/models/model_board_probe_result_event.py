# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable event emitted for one completed board probe (OMN-19937)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_probe_outcome import (
    EnumBoardProbeOutcome,
)
from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_subject_kind import (
    EnumBoardSubjectKind,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_board_probe_result import (
    ModelBoardProbeResult,
)


class ModelBoardProbeResultEvent(BaseModel):
    """One immutable board-probe verdict at its durable event grain."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    check_id: str = Field(min_length=1)
    subject_kind: EnumBoardSubjectKind
    subject: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    surface_instance: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    outcome: EnumBoardProbeOutcome
    reasons: tuple[str, ...] = Field(min_length=1)
    evidence_items: tuple[str, ...] = ()
    finished_at: datetime

    @field_validator("finished_at")
    @classmethod
    def _finished_at_is_utc(cls, value: datetime) -> datetime:
        """Require an aware UTC instant and normalize it to ``datetime.UTC``."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("finished_at must be timezone-aware UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("finished_at must use UTC")
        return value.astimezone(UTC)

    @property
    def key(self) -> tuple[str, EnumBoardSubjectKind, str, str, str, str]:
        """Stable identity of one check execution on one surface."""
        return (
            self.check_id,
            self.subject_kind,
            self.repo,
            self.sha,
            self.surface_instance,
            self.execution_id,
        )

    @property
    def partition_key(self) -> str:
        """Keep all observations for one subject on the same partition."""
        return self.subject


def board_probe_result_event_from(
    result: ModelBoardProbeResult,
    *,
    subject_kind: EnumBoardSubjectKind,
    repo: str,
    sha: str,
    surface_instance: str,
    execution_id: str,
    finished_at: datetime,
) -> ModelBoardProbeResultEvent:
    """Map a board-probe result and its execution context to the wire event."""
    return ModelBoardProbeResultEvent(
        check_id=result.check_id.value,
        subject_kind=subject_kind,
        subject=result.subject,
        repo=repo,
        sha=sha,
        surface_instance=surface_instance,
        execution_id=execution_id,
        outcome=result.outcome,
        reasons=result.reasons,
        evidence_items=result.evidence_items,
        finished_at=finished_at,
    )


__all__ = ["ModelBoardProbeResultEvent", "board_probe_result_event_from"]
