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

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for distributed tracing of this report_error operation.",
    )

    created: bool = Field(
        False,
        description="True when a new Linear ticket was created for this fingerprint.",
    )

    skipped: bool = Field(
        False,
        description="True when the fingerprint already exists — occurrence_count incremented.",
    )

    issue_id: UUID | None = Field(
        None,
        description=(
            "Linear issue UUID. "
            "Present on both created and skipped outcomes; None on failure."
        ),
    )

    issue_url: str = Field(
        "",
        description=(
            "Linear issue URL. "
            "Present on both created and skipped outcomes; empty string on failure."
        ),
    )

    occurrence_count: int = Field(
        1,
        description="Current occurrence_count from db_error_tickets after the operation.",
    )

    error: str = Field(
        "",
        description="Sanitized error message when the operation failed (neither created nor skipped).",
    )


__all__ = ["ModelDbErrorTicketResult"]
