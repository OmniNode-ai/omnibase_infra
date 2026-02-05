# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ledger event model for failed database query."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from omnibase_infra.models.ledger.model_ledger_event_base import ModelLedgerEventBase


class ModelDbQueryFailed(ModelLedgerEventBase):
    """Ledger event emitted when a database query fails.

    Emitted when PostgresRepositoryRuntime.call() raises an exception.

    Attributes:
        event_type: Always "db.query.failed".
        duration_ms: Time until failure in milliseconds.
        error_type: Exception class name (e.g., "RepositoryTimeoutError").
        error_message: Sanitized error message (no sensitive data).
        retriable: Whether the error is retriable.
        circuit_breaker_state: Current circuit breaker state if applicable.
    """

    event_type: Literal["db.query.failed"] = Field(
        default="db.query.failed",
        description="Event type discriminator.",
    )

    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Time until failure in milliseconds.",
    )

    error_type: str = Field(
        ...,
        min_length=1,
        description="Exception class name (e.g., 'RepositoryTimeoutError').",
    )

    error_message: str = Field(
        default="",
        description="Sanitized error message (no sensitive data).",
    )

    retriable: bool = Field(
        ...,
        description="Whether the error is retriable.",
    )

    circuit_breaker_state: str | None = Field(
        default=None,
        description="Current circuit breaker state if applicable ('closed', 'open', 'half_open').",
    )


__all__ = ["ModelDbQueryFailed"]
