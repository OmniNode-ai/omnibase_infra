# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ledger event model for database query request."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from omnibase_infra.models.ledger.model_ledger_event_base import ModelLedgerEventBase


class ModelDbQueryRequested(ModelLedgerEventBase):
    """Ledger event emitted when a database query is requested.

    Emitted at the entry point of PostgresRepositoryRuntime.call() before
    any database operation is attempted.

    Attributes:
        event_type: Always "db.query.requested".
        query_fingerprint: Hash of operation name + stable param shape (NOT raw SQL).
    """

    event_type: Literal["db.query.requested"] = Field(
        default="db.query.requested",
        description="Event type discriminator.",
    )

    query_fingerprint: str = Field(
        ...,
        min_length=1,
        description=(
            "Hash of operation name + stable param shape. "
            "Does NOT include raw SQL or param values."
        ),
    )


__all__ = ["ModelDbQueryRequested"]
