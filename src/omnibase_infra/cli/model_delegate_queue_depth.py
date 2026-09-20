# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How deep the command queue was when a delegation's wait expired (OMN-18852).

The tri-state here is the point. ``records_ahead`` is an ``int`` when the
broker answered, and ``None`` when it did not -- and ``None`` always carries a
``unresolved_reason`` naming why. Collapsing the second case to ``0`` would
tell a caller whose record sat behind a 445-second backlog that the queue was
empty, which is both false and the most expensive thing this refusal could
say. Absence of a measurement is reported as absence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["ModelDelegateQueueDepth"]


class ModelDelegateQueueDepth(BaseModel):
    """An observation of the command queue, or a named reason there is none."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records_ahead: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Records the observed consumer group had yet to commit on the "
            "command topic, excluding this caller's own. None means the depth "
            "could NOT be observed; read unresolved_reason. Never a "
            "caller-supplied or assumed figure."
        ),
    )
    backlog_records: int | None = Field(
        default=None,
        ge=0,
        description=(
            "The group's total uncommitted backlog on the command topic, "
            "INCLUDING this caller's own record. records_ahead is this minus "
            "one. Reported alongside rather than folded in, because the "
            "subtraction is the one assumption in the figure."
        ),
    )
    consumer_group: str = Field(
        default="",
        description=(
            "The consumer group whose backlog was measured, or whose absence "
            "made the measurement impossible. Empty when no group was known "
            "to probe at all."
        ),
    )
    unresolved_reason: str = Field(
        default="",
        description=(
            "Why the depth is absent, in the caller's own terms. Empty iff "
            "records_ahead is set."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_of_depth_or_reason(self) -> ModelDelegateQueueDepth:
        """A depth and a reason are mutually exclusive, and one is required.

        Both set would let a reader take the number while the reason says it
        is not trustworthy; neither set is a silent unknown, which is the
        shape this model exists to make impossible.
        """
        has_depth = self.records_ahead is not None
        has_reason = bool(self.unresolved_reason)
        if has_depth == has_reason:
            raise ValueError(
                "ModelDelegateQueueDepth requires exactly one of "
                f"records_ahead ({self.records_ahead!r}) and "
                f"unresolved_reason ({self.unresolved_reason!r})"
            )
        if has_depth and self.backlog_records is None:
            raise ValueError(
                "an observed records_ahead must report the backlog_records it "
                "was derived from"
            )
        return self

    def describe(self) -> str:
        """One human sentence for the refusal line."""
        if self.records_ahead is None:
            return f"queue depth could not be resolved: {self.unresolved_reason}"
        if self.records_ahead == 0:
            return (
                "no records were ahead of it on "
                f"{self.consumer_group} (observed backlog "
                f"{self.backlog_records}), so the wait was not queueing"
            )
        return (
            f"{self.records_ahead} record(s) were ahead of it, still "
            f"uncommitted by {self.consumer_group} (observed backlog "
            f"{self.backlog_records})"
        )
