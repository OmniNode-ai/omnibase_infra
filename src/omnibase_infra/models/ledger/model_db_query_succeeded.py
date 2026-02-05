# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ledger event model for successful database query."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from omnibase_infra.models.ledger.model_ledger_event_base import ModelLedgerEventBase


class ModelDbQuerySucceeded(ModelLedgerEventBase):
    """Ledger event emitted when a database query succeeds.

    Emitted after PostgresRepositoryRuntime.call() completes successfully.

    Attributes:
        event_type: Always "db.query.succeeded".
        duration_ms: Query execution time in milliseconds.
        rows_returned: Number of rows returned (0 for writes, count for reads).
        result_fingerprint: Hash of result for replay verification (optional).
    """

    event_type: Literal["db.query.succeeded"] = Field(
        default="db.query.succeeded",
        description="Event type discriminator.",
    )

    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Query execution time in milliseconds.",
    )

    rows_returned: int = Field(
        ...,
        ge=0,
        description="Number of rows returned (0 for writes, count for reads).",
    )

    result_fingerprint: str | None = Field(
        default=None,
        description="Hash of result for replay verification (Tier-2, optional).",
    )


__all__ = ["ModelDbQuerySucceeded"]
