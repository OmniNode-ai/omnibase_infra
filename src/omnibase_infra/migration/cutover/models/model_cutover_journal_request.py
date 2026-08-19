# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request for one append-only cutover journal event."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.migration.cutover.enums import EnumCutoverEventKind
from omnibase_infra.migration.cutover.models.model_application_path_write_proof import (
    ModelApplicationPathWriteProof,
)

_IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"

_RECEIPT_EVENTS = {
    EnumCutoverEventKind.BACKFILL_COMPLETED,
    EnumCutoverEventKind.FINAL_DELTA_APPLIED,
    EnumCutoverEventKind.WRITER_CHECKPOINT,
    EnumCutoverEventKind.REVERSE_DELTA_PROVEN,
    EnumCutoverEventKind.FORWARD_FIX_RECORDED,
    EnumCutoverEventKind.MISMATCH_RESOLVED,
}


class ModelCutoverJournalRequest(BaseModel):
    """Validate event-specific durable evidence before persistence."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: EnumCutoverEventKind
    occurred_at: datetime
    evidence_ref: str = Field(..., min_length=1, max_length=500)
    idempotency_key: str = Field(
        ..., min_length=1, max_length=200, pattern=_IDEMPOTENCY_KEY_PATTERN
    )
    receipt_id: UUID | None = None
    source_binding_ref: str = Field(default="", max_length=200)
    target_binding_ref: str = Field(default="", max_length=200)
    application_path_write_proof: ModelApplicationPathWriteProof | None = None
    target_sequence: int | None = Field(default=None, ge=1)
    dual_write_expires_at: datetime | None = None
    observation_ends_at: datetime | None = None
    reverse_delta_proof_id: UUID | None = None

    @field_validator("occurred_at", "dual_write_expires_at", "observation_ends_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("journal timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_event_evidence(self) -> ModelCutoverJournalRequest:
        if self.kind in _RECEIPT_EVENTS and self.receipt_id is None:
            raise ValueError(f"{self.kind.value} requires a receipt_id")

        if self.kind is EnumCutoverEventKind.WRITER_CHECKPOINT:
            if not self.source_binding_ref or not self.target_binding_ref:
                raise ValueError("writer checkpoint requires source/target bindings")

        if self.kind is EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN:
            if self.application_path_write_proof is None:
                raise ValueError(
                    "application-path write proof requires an independently "
                    "verified application_path_write_proof"
                )

        if self.kind is EnumCutoverEventKind.DUAL_WRITE_STARTED:
            if self.dual_write_expires_at is None:
                raise ValueError("dual-write start requires a finite expiry")

        if self.kind is EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED:
            if self.observation_ends_at is None:
                raise ValueError("observation start requires an explicit end")

        if self.kind is EnumCutoverEventKind.WRITER_QUIESCED:
            if self.target_sequence is None:
                raise ValueError("writer quiescence requires final target sequence")

        if self.kind is EnumCutoverEventKind.REVERSE_DELTA_PROVEN:
            if self.reverse_delta_proof_id is None:
                raise ValueError("reverse-delta event requires proof id")

        if self.receipt_id is not None and self.kind not in _RECEIPT_EVENTS:
            raise ValueError(f"{self.kind.value} does not accept receipt_id")
        if (self.source_binding_ref or self.target_binding_ref) and self.kind is not (
            EnumCutoverEventKind.WRITER_CHECKPOINT
        ):
            raise ValueError("binding refs are only valid on writer checkpoint")
        if self.application_path_write_proof is not None and self.kind is not (
            EnumCutoverEventKind.APPLICATION_PATH_WRITE_PROVEN
        ):
            raise ValueError(
                "application_path_write_proof is only valid on write-proof events"
            )
        if self.target_sequence is not None and self.kind is not (
            EnumCutoverEventKind.WRITER_QUIESCED
        ):
            raise ValueError("target_sequence is not valid for this event")
        if self.dual_write_expires_at is not None and self.kind is not (
            EnumCutoverEventKind.DUAL_WRITE_STARTED
        ):
            raise ValueError("dual-write expiry is only valid on dual-write start")
        if self.observation_ends_at is not None and self.kind is not (
            EnumCutoverEventKind.OBSERVATION_WINDOW_STARTED
        ):
            raise ValueError("observation end is only valid on observation start")
        if self.reverse_delta_proof_id is not None and self.kind is not (
            EnumCutoverEventKind.REVERSE_DELTA_PROVEN
        ):
            raise ValueError("reverse-delta proof id is only valid on its proof event")
        return self


__all__ = ["ModelCutoverJournalRequest"]
