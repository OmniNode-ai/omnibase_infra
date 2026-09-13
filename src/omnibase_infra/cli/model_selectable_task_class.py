# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One gateway-exposed task class and the predicate that selects it (OMN-18305)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelSelectableTaskClass"]


class ModelSelectableTaskClass(BaseModel):
    """One gateway-exposed task class and the predicate that selects it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    priority: int = Field(ge=0)
    phrases: tuple[str, ...]
    min_words: int | None = None
    max_words: int | None = None

    def shape_admits(self, word_count: int) -> bool:
        """Return whether a prompt of this length is eligible at all."""
        if self.min_words is not None and word_count < self.min_words:
            return False
        return not (self.max_words is not None and word_count > self.max_words)

    def matching_phrase(self, lowered_prompt: str) -> str | None:
        """Return the most specific declared phrase present on a word boundary.

        Longest first, then alphabetically: when a class declares both "test"
        and "write a test", the reason line should name the phrase that
        actually describes the request, and declaration order in a YAML map is
        not a thing a contract author should have to reason about.
        """
        for phrase in sorted(self.phrases, key=lambda item: (-len(item), item)):
            normalized = phrase.lower()
            if normalized and re.search(
                rf"(?<!\w){re.escape(normalized)}(?!\w)", lowered_prompt
            ):
                return phrase
        return None
