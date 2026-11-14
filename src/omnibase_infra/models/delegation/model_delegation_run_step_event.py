# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One numbered step of one delegation run, as the runtime dispatch port emits it.

OMN-19972. The deployed delegate-skill handler dispatches through
``RuntimeDelegationDispatchPort``, so that port numbers a run's steps and
publishes them on the topic the delegate-skill contract declares as
``runtime_dispatch.run_step_event``. The run-trace projection folds them.

This is the wire shape of that topic. omnibase_infra never imports the
consumer's package (market depends on infra, not the reverse), so the shape is
spelled here field for field and the consumer validates the same JSON with its
own model: a tenant, a run and a step sequence form the key, and
``step_sequence`` is the only ordering authority within a run.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

RunStepKind = Literal[
    "request", "routing_decision", "provider_call", "quality_gate", "terminal"
]
RunStepOutcome = Literal["succeeded", "failed"]
# A named tenant slug, as the event envelope records it, not a UUID.
TenantSlug = Annotated[str, Field(min_length=1, pattern=r"\S")]


class ModelDelegationRunStepEvent(BaseModel):
    """One step of one delegation run, keyed by (tenant, run, step sequence)."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tenant_id: TenantSlug
    correlation_id: UUID
    step_sequence: int = Field(ge=1, strict=True)

    step_kind: RunStepKind
    started_at: AwareDatetime
    duration_ms: int | None = Field(default=None, ge=0)
    outcome: RunStepOutcome
    failure_cause: str | None = None

    backend_id: str | None = None
    model_id: str | None = None
    attempt_number: int | None = Field(default=None, ge=1)
    reason: str | None = None
    candidates: tuple[str, ...] = ()

    # Set only on the terminal of a failed run: the earlier step that failed it.
    failing_step_sequence: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def failing_step_belongs_to_a_failed_terminal(self) -> Self:
        if self.failing_step_sequence is None:
            return self
        if self.step_kind != "terminal" or self.outcome != "failed":
            raise ValueError("only the terminal of a failed run names a failing step")
        if self.failing_step_sequence >= self.step_sequence:
            raise ValueError("the failing step precedes the terminal that names it")
        return self


__all__ = [
    "ModelDelegationRunStepEvent",
    "RunStepKind",
    "RunStepOutcome",
    "TenantSlug",
]
