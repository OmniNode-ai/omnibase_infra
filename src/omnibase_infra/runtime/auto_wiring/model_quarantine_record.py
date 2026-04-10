# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Quarantine record for contracts that fail handshake validation (OMN-7657)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.enum_handshake_failure_reason import (
    HandshakeFailureReason,
)


class ModelQuarantineRecord(BaseModel):
    """Record of a quarantined contract that failed handshake validation.

    Created by the auto-wiring engine when a required handshake hook
    fails after all retries are exhausted. Quarantined contracts are
    excluded from handler wiring and surfaced in health/readiness
    endpoints.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    handler_id: str = Field(  # ONEX_EXCLUDE: uuid_field - handler_id is a dotted class path, not a UUID
        ...,
        min_length=1,
        description="Handler ID of the quarantined contract",
    )
    node_kind: str = Field(
        ...,
        min_length=1,
        description="Node kind of the quarantined contract",
    )
    failure_reason: HandshakeFailureReason = Field(
        ...,
        description="Classified reason for quarantine",
    )
    error_message: str = Field(
        default="",
        description="Diagnostic message from the last failed attempt",
    )
    attempts: int = Field(
        ...,
        ge=1,
        description="Total number of handshake attempts made",
    )
    quarantined_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp when the contract was quarantined",
    )


__all__ = ["ModelQuarantineRecord"]
