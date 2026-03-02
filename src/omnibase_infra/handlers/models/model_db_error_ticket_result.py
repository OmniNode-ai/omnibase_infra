# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output model for HandlerLinearDbErrorReporter.

``ModelDbErrorTicketResult`` carries the result of a single
``report_error`` operation executed by ``HandlerLinearDbErrorReporter``:
whether a new Linear ticket was created for the PostgreSQL error, or an
existing record was deduplicated (occurrence_count incremented).

Related Tickets:
    - OMN-3408: Kafka Consumer -> Linear Ticket Reporter (ONEX Node)
    - OMN-3407: PostgreSQL Error Emitter (hard prerequisite)
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelDbErrorTicketResult(BaseModel):
    """Result returned by HandlerLinearDbErrorReporter.

    Carries the outcome of a single ``report_error`` operation: whether a
    new Linear ticket was created or an existing record was updated (dedup).

    States:
        - ``created=True``  — new Linear ticket created, DB row inserted
        - ``skipped=True``  — fingerprint already in db_error_tickets;
          occurrence_count was incremented, no new ticket created
        - neither           — operation failed; see ``error`` field
    """

    correlation_id: UUID = Field(default_factory=uuid4)
    """UUID for distributed tracing."""

    created: bool = False
    """True when a new Linear ticket was created for this fingerprint."""

    skipped: bool = False
    """True when the fingerprint already exists — occurrence_count incremented."""

    issue_id: str | None = None
    """Linear issue ID (e.g. ``"abc-123-uuid"``).

    Present on both ``created`` and ``skipped`` outcomes.
    """

    issue_url: str | None = None
    """Linear issue URL.

    Present on both ``created`` and ``skipped`` outcomes.
    """

    occurrence_count: int = 1
    """Current ``occurrence_count`` from ``db_error_tickets`` after the operation."""

    error: str | None = None
    """Sanitized error message when the operation failed (neither created nor skipped)."""


__all__ = ["ModelDbErrorTicketResult"]
