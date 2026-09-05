# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-ticket outcome record for the evidence autoclose sweep."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_arm import (
    EnumEvidenceAutocloseArm,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)


class ModelEvidenceAutocloseOutcome(BaseModel):
    """One companion-PR / bound-ticket pair's terminal sweep decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(
        default="", description="Bound Linear ticket id, e.g. OMN-16106."
    )
    companion_pr_number: int = Field(
        default=0, description="Merged OCC companion PR number."
    )
    companion_pr_url: str = Field(
        default="", description="Merged OCC companion PR URL."
    )
    decision: EnumEvidenceAutocloseDecision = Field(
        ..., description="Terminal classification for this pair."
    )
    reason: str = Field(default="", description="Human-readable explanation.")
    # OMN-17342. Which enumeration arm offered this pair. Defaults to FORWARD so
    # every pre-existing construction site keeps its meaning unchanged — the
    # forward window was the only arm that existed. It is recorded on EVERY
    # outcome rather than only on backfilled ones because the absence of a
    # marker is not a readable signal: "no arm field" and "the forward arm" have
    # to be distinguishable in a receipt read months later, and a default that
    # is also a real value only works if it is written out.
    enumeration_arm: EnumEvidenceAutocloseArm = Field(
        default=EnumEvidenceAutocloseArm.FORWARD,
        description="Enumeration arm that selected this (companion, ticket) pair.",
    )
    dod_verify_total_checks: int = Field(default=0, ge=0)
    dod_verify_verified_count: int = Field(default=0, ge=0)
    dod_verify_failed_count: int = Field(default=0, ge=0)
    # OMN-15911: how many of the passing checks actually executed the claimed
    # behavior, as reported by dod_verify's own verdict. Recorded on every
    # outcome that reached a verdict — including the flip — so the sweep
    # result says on what STRENGTH of evidence a ticket was closed, not merely
    # that a count matched.
    dod_verify_behavior_proving_count: int = Field(default=0, ge=0)
    # OMN-16821: how many checks executed, exited 0, and could not have exited
    # otherwise for a product reason (OMN-15391). Recorded because the flip
    # equality is `verified + non_probative == total`, so without this field a
    # flip reads as an unexplained "6/12 verified" in the structured record and
    # is auditable only from the free-text reason — the counts-without-detail
    # problem OMN-16788 already hit once.
    dod_verify_non_probative_count: int = Field(default=0, ge=0)
    uncovered_acceptance_criteria: tuple[str, ...] = Field(
        default=(),
        description=(
            "Acceptance criteria found in the ticket's Linear description that "
            "dod_verify's checks do not cover (GAP_AC_COVERAGE only). Recorded "
            "on the outcome as well as in the comment so a DRY-RUN, which posts "
            "no comment, still names exactly what blocked the flip."
        ),
    )
    # ------------------------------------------------------------------
    # OMN-17658 — the BOUND READBACK. Three ids that make one flip checkable
    # by somebody who was not there: which companion carried the evidence
    # (above), which dod_verify verdict released it (`verdict_fingerprint`),
    # and which Linear state-history entry the write actually produced
    # (`readback_entry_id`, which must differ from `pre_write_head_entry_id`).
    #
    # The last pair is the point. `issueUpdate` returning `success: true` is
    # the API agreeing to the request, not evidence the board changed —
    # exactly the class of claim the deterministic-truth doctrine refuses. A
    # flip is recorded as FLIPPED only when a post-write read of the ticket's
    # own history shows a completed segment the pre-write read did not have.
    pre_write_head_entry_id: str = Field(
        default="",
        description=(
            "Newest `stateHistory` entry id observed BEFORE the flip was "
            "written. Empty on every non-applying path."
        ),
    )
    readback_entry_id: str = Field(
        default="",
        description=(
            "`stateHistory` entry id of the completed segment the flip "
            "produced, read back AFTER the write. Empty on every non-applying "
            "path, and empty on an applying path whose readback did not "
            "confirm — which is ERROR_READBACK_UNCONFIRMED, never FLIPPED."
        ),
    )
    verdict_fingerprint: str = Field(
        default="",
        description=(
            "Stable digest of the dod_verify counters that released or "
            "withheld this decision (total/verified/failed/non-probative/"
            "behaviour-proving). Lets two receipts be compared for 'same "
            "verdict' without re-parsing free text."
        ),
    )
    linear_comment_posted: bool = Field(
        default=False, description="Whether an audit/gap comment was posted."
    )
    applied: bool = Field(
        default=False,
        description="True only when a real Linear mutation was made (apply=True run).",
    )


__all__ = ["ModelEvidenceAutocloseOutcome"]
