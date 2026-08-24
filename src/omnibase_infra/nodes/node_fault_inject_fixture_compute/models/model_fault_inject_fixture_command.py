# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Command model for the fault-injection fixture node.

Ticket: OMN-16265
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Upper bound on the requested padding size. This is deliberately well above
# any known broker message.max.bytes / producer max_request_size ceiling
# observed on onex-dev (OMN-14498 measured ~1,048,588 bytes live) so a caller
# can always drive an oversized-result publish failure, while still being a
# hard cap that keeps a misconfigured caller from requesting an unbounded
# allocation inside the handler process.
MAX_INFLATE_RESULT_BYTES = 16_000_000


class ModelFaultInjectFixtureCommand(BaseModel):
    """Command requesting a deterministic, size-controlled fixture result.

    Ticket: OMN-16265 (OMN-14498 follow-on). ``inflate_result_bytes`` lets the
    caller reproduce the OMN-14498 live-probe technique deterministically: a
    small command amplified into an oversized result that fails the runtime's
    primary publish leg, driving ``BoundaryApplyPublishError`` / offset
    withholding when this fixture is deployed with a private
    ``dead_letter_topic`` that also cannot accept the write (see the node's
    ``contract.yaml`` and the deployment runbook for the DLQ-leg setup).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for tracing this fault-injection run.",
    )
    inflate_result_bytes: int = Field(
        default=0,
        ge=0,
        le=MAX_INFLATE_RESULT_BYTES,
        description=(
            "Target size in bytes for the handler-amplified `padding` field "
            "in the result. 0 produces a minimal, non-faulting result. A "
            "value tuned past the live broker's message.max.bytes / "
            "producer max_request_size triggers the primary-publish failure "
            "this fixture exists to drive."
        ),
    )
    marker: str = Field(
        default="",
        description=(
            "Free-text marker (e.g. a ticket id or run label) echoed nowhere "
            "structurally but useful for log/correlation grepping during a "
            "fault-injection run. Not validated or interpreted."
        ),
    )


__all__: list[str] = ["ModelFaultInjectFixtureCommand", "MAX_INFLATE_RESULT_BYTES"]
