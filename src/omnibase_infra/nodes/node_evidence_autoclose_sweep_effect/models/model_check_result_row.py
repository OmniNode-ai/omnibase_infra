# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One dod_verify check, as the sweep's own record of it (OMN-18490)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_ac_binding_check_status import (
    EnumAcBindingCheckStatus,
)

#: OMN-18490. How much of a check's own message travels on the outcome.
#:
#: A pytest failure message runs to kilobytes and a contract can declare
#: ninety-four checks, so carrying them whole would make the receipt the
#: unreadable thing — which is the defect this model exists to fix, not a
#: price worth paying for it. The excerpt is the first line of triage: enough
#: to tell an assertion failure from a missing credential from a timeout, and
#: a pointer to go read the rest.
MESSAGE_EXCERPT_MAX_CHARS = 200


class ModelCheckResultRow(BaseModel):
    """One evidence check's verdict, recorded per outcome rather than counted.

    WHY THIS EXISTS. The sweep reported ``30/94 ACs verified, 3 failed`` and
    named none of the three — on the outcome, in the comment it posted to the
    ticket, and in the job log. A collaborator asked twice which three, and
    the answer was not recoverable: the verifier's own payload is a subprocess
    result the run read four counters out of and discarded.

    Nothing here is newly COLLECTED. ``evidence_id``, ``status``, ``message``,
    ``proof_class`` and ``binds_ac`` are already on the dod_verify terminal
    payload, and the handler already walks that list three times — for the
    unreachable-live-surface classifier, for the never-executed classifier,
    and for the gap fingerprint. This model is the part that was missing:
    somewhere for what it read to be written down.

    These rows are DESCRIPTIVE and decide nothing. Every counter the flip
    predicate consults still comes from the verdict's own count fields, so a
    verdict whose declared tally disagrees with its own check list is recorded
    faithfully as the disagreement it is, rather than silently re-derived into
    agreement.

    Field names and the status type are deliberately those of its sibling
    :class:`ModelAcBindingRow`, which records the same checks from the other
    direction — criterion-first rather than check-first. Two records of one
    verdict that spelled the same fact two ways would be the next reader's
    problem, and ``EnumAcBindingCheckStatus`` already carries the honesty this
    row needs: an ``UNKNOWN`` member for a status the verifier reported and
    this repo does not know, never coerced to a neighbour.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    evidence_check: str = Field(
        ...,
        description=(
            "The check's `evidence_id` from the OCC contract — a contract "
            "author's label such as `dod-tests`, never a uuid. "
            "`<unnamed check>` when the record carries none, which is a "
            "contract-authoring defect and is recorded rather than dropped."
        ),
    )
    status: EnumAcBindingCheckStatus = Field(
        default=EnumAcBindingCheckStatus.UNKNOWN,
        description=(
            "This check's dod_verify verdict. `UNKNOWN` means the verifier "
            "reported a status this repo does not know — recorded as the "
            "unknown it is rather than guessed at, because a row whose status "
            "was inferred would be the counters problem again one level down."
        ),
    )
    proof_class: str = Field(
        default="",
        description=(
            "What a passing check BOUND — behaviour, merge state, surrogate "
            "(OMN-15911). Empty when the record declares none. This is the "
            "difference between a check that ran the claimed behaviour and "
            "one that read a merge state, which two rows with the same "
            "`verified` are otherwise indistinguishable on."
        ),
    )
    binds_ac: tuple[str, ...] = Field(
        default=(),
        description=(
            "Acceptance criteria this check DECLARES it covers (OMN-18056). "
            "Empty means it declares none, which is a different fact from a "
            "criterion nothing declares — that one is reported by "
            "`ModelEvidenceAutocloseOutcome.ac_binding_rows`, from the other "
            "direction."
        ),
    )
    message_excerpt: str = Field(
        default="",
        max_length=MESSAGE_EXCERPT_MAX_CHARS,
        description=(
            "The leading characters of the check's own message, bounded by "
            "`MESSAGE_EXCERPT_MAX_CHARS`. Empty when the record carries no "
            "message. An id with no excerpt sends a reader to a job log that "
            "GitHub may no longer hold, so the excerpt is what makes the row "
            "answerable months later."
        ),
    )


__all__ = ["MESSAGE_EXCERPT_MAX_CHARS", "ModelCheckResultRow"]
