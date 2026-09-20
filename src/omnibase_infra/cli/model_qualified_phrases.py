# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Ambiguous selection phrases and the qualifiers that settle them (OMN-18831)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelQualifiedPhrases"]


class ModelQualifiedPhrases(BaseModel):
    """Phrases that claim a prompt only with a declared qualifier nearby (OMN-18831).

    WHY THIS EXISTS. Some of the most useful selection phrases are ordinary
    English before they are technical. ``"write a"`` opens almost any
    instruction-shaped request, and ``"assertion"`` is the ordinary word for a
    claim. Declared as plain ``phrases`` they routed prose to classes whose
    acceptance is deterministic -- a 388-word request opening ``"Write a
    GitHub PR body in markdown"`` was graded on whether its answer compiled as
    Python, and five correct English answers were refused in a row.

    WHY NOT DELETE THEM. ``"write a parser"`` is a code request and ``"add
    assertions to the auth tests"`` is a test request, and no other declared
    phrase claims either. Deleting the phrase trades one misroute for another.

    WHAT IS DECLARED HERE. The counter-signal is the OBJECT of the verb: a
    gated phrase counts only when one of ``qualifiers`` occurs within
    ``within_words`` words before or after it. That is a closed vocabulary --
    the code artifacts a request can name -- where "every way prose can be
    phrased" is not, so the gate is stated positively and does not have to be
    guessed at. The vocabulary IS the word sense, so it belongs in the
    contract beside the phrases rather than in whichever consumer evaluates it.

    Every field is required and non-empty. A block with no qualifiers, no
    phrases, or a zero-word window would degrade to "the phrase matches
    unconditionally", which is precisely the defect the field removes -- so it
    is refused at load rather than allowed to pass through silently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    within_words: int = Field(
        ge=1,
        description=(
            "How many words either side of the phrase are searched for a "
            "qualifier. Counted in words so the declared number survives "
            "rewording."
        ),
    )
    phrases: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "The ambiguous phrases. Matched on word boundaries exactly as "
            "`phrases` are, then gated."
        ),
    )
    qualifiers: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "The terms whose presence near a phrase settles its word sense. "
            "Matched on word boundaries, so multi-word qualifiers work."
        ),
    )
