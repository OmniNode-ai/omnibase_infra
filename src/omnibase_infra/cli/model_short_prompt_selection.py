# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""How a prompt below a class's word floor can still select it (OMN-19140)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["ModelShortPromptSelection"]


class ModelShortPromptSelection(BaseModel):
    """Admit a short prompt that OPENS with an imperative the class declares.

    WHY THIS EXISTS. ``summarization`` declares ``min_words: 120``, and a shape
    gate is evaluated before any phrase, so "Summarize in one sentence: ..." was
    structurally ineligible for the one class that exists for it. The OMN-19136
    shadow run traced 15 of its 19 disagreements to that line.

    WHY THE FLOOR STAYS. Measured over every recorded delegation prompt,
    removing the floor moved 48 prompts to ``summarization``. The 44 claimed on
    the verb all opened with it and all asked for a summary. The 4 claimed on
    the noun ``summary`` asked for something else. The floor keeps a class's
    ordinary nouns from claiming short prompts that merely mention them.

    WHAT IS DECLARED. Between ``min_words`` and the class's own floor, the
    class is eligible only when the prompt opens with one of
    ``opening_phrases``. Each must also be a plain phrase of the class, so the
    same request padded past the floor is still claimed: the block widens
    eligibility downwards and never narrows it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_words: int = Field(
        ge=1,
        description=(
            "Shortest prompt, in words, admitted by an opening phrase. Below "
            "it a prompt is too thin to hold anything to act on."
        ),
    )
    opening_phrases: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Phrases that admit a short prompt when the prompt opens with one. "
            "Matched at the start of the prompt, on a word boundary."
        ),
    )

    @field_validator("opening_phrases")
    @classmethod
    def _validate_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        invalid = sorted(
            term for term in value if not term or term != term.strip().lower()
        )
        if invalid:
            raise ValueError(
                f"opening phrases must be non-empty, trimmed and lowercase: {invalid}"
            )
        return value
