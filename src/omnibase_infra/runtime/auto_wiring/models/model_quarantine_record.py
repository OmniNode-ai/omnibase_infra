# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Quarantine record for contracts that failed handshake validation (OMN-7657)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.enum_handshake_failure_reason import (
    HandshakeFailureReason,
)


class ModelQuarantineRecord(BaseModel):
    """Record of a quarantined contract that failed handshake validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    handler_id: str = Field(
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
