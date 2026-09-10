# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One hop of a delegation chain, exactly as it was observed (OMN-16964).

This is EVIDENCE, not a verdict. Every field is something that was read off a
recorded envelope; nothing on this model is derived, scored, or judged. The
derivation happens in ``chain_replay`` and lands on
``ModelLedgerChainRow`` — keeping the two apart is what makes it possible to
prove that the replay re-derived something rather than copying a flag that
arrived pre-set.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelObservedHop(BaseModel):
    """A single recorded envelope belonging to one correlation's chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(
        description="The topic this envelope was observed on. Becomes the hop name."
    )
    envelope_id: UUID = Field(
        description=(
            "This envelope's own identifier. The NEXT hop's recorded parent "
            "must re-derive to this value, which is what the replay checks."
        )
    )
    parent_envelope_id: UUID | None = Field(
        default=None,
        description=(
            "The envelope this one records as its cause. NONE on the head of "
            "the chain, which has no predecessor to point at. None is the "
            "honest encoding: an empty string would be a parent that exists "
            "and is blank, which is a different claim."
        ),
    )
    correlation_id: UUID | None = Field(
        default=None,
        description=(
            "The correlation id carried on THIS envelope. Recorded per hop "
            "rather than assumed from the chain, because a correlation id that "
            "changes mid-chain is precisely the OMN-16931 defect class the "
            "replay exists to catch — and it is uncatchable if the chain's id "
            "is simply copied onto every hop."
        ),
    )


__all__ = ["ModelObservedHop"]
