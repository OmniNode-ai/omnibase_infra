# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One gateway-exposed task class and the predicate that selects it (OMN-18305)."""

from __future__ import annotations

import re
from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.model_qualified_phrases import ModelQualifiedPhrases
from omnibase_infra.cli.request_instruction import is_negated

__all__ = ["ModelSelectableTaskClass"]

#: One run of non-space characters. Used to count words OUT from a matched
#: phrase, so ``within_words`` means the same thing it means in the contract's
#: prose: words, not characters.
_WORD = re.compile(r"\S+")


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Return the word-boundary matcher for one declared phrase.

    ``(?<!\\w)`` / ``(?!\\w)`` rather than ``\\b`` so a phrase ending in
    punctuation still matches; the OMN-18305 defect was a bare ``in`` test with
    no boundary at all ("latest" contains "test").
    """
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")


class ModelSelectableTaskClass(BaseModel):
    """One gateway-exposed task class and the predicate that selects it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    priority: int = Field(ge=0)
    phrases: tuple[str, ...]
    min_words: int | None = None
    max_words: int | None = None
    qualified_phrases: ModelQualifiedPhrases | None = None
    vetoed_by: tuple[str, ...] = ()

    def shape_admits(self, word_count: int) -> bool:
        """Return whether a prompt of this length is eligible at all."""
        if self.min_words is not None and word_count < self.min_words:
            return False
        return not (self.max_words is not None and word_count > self.max_words)

    def matching_phrase(self, lowered_prompt: str) -> str | None:
        """Return the most specific declared phrase that claims this prompt.

        Longest first, then alphabetically: when a class declares both "test"
        and "write a test", the reason line should name the phrase that
        actually describes the request, and declaration order in a YAML map is
        not a thing a contract author should have to reason about.

        A phrase declared under ``qualified_phrases`` claims the prompt only at
        an occurrence with a qualifier nearby. Every occurrence is considered,
        not only the first: a prompt that says "write a note now, and later
        write a parser for it" is a code request, and letting the earlier
        ordinary use veto the later technical one would be frequency deciding
        the answer, which the contract forbids.
        """
        for phrase, _start in self._claiming_occurrences(lowered_prompt):
            return phrase
        return None

    def opening_words(self) -> frozenset[str]:
        """Return the first word of every phrase this class declares (OMN-19523).

        A request that opens with one of these words ("write", "review",
        "summarize", "fix") opens with the verb of a declared request, so its
        opening sentence is where the work is named. Gated phrases count: the
        word opens a request whether or not its object qualifies it.
        """
        gated = self.qualified_phrases
        phrases = list(self.phrases) + (list(gated.phrases) if gated else [])
        return frozenset(
            phrase.lower().split()[0] for phrase in phrases if phrase.split()
        )

    def _claiming_occurrences(self, lowered_prompt: str) -> Iterator[tuple[str, int]]:
        """Yield ``(phrase, start)`` for every occurrence that claims the prompt.

        Phrases in the order :meth:`matching_phrase` reports them (longest
        first, then alphabetically), occurrences left to right within each.
        """
        gated = self.qualified_phrases
        gated_set = (
            frozenset(phrase.lower() for phrase in gated.phrases)
            if gated is not None
            else frozenset()
        )
        candidates = list(self.phrases) + (
            list(gated.phrases) if gated is not None else []
        )
        for phrase in sorted(candidates, key=lambda item: (-len(item), item)):
            normalized = phrase.lower()
            if not normalized:
                continue
            for occurrence in _phrase_pattern(normalized).finditer(lowered_prompt):
                # OMN-19523: "no summary of the change" and "not a review" say
                # what the caller does NOT want; a negated occurrence claims
                # nothing, and a later plain one still can.
                if is_negated(lowered_prompt, occurrence.start()):
                    continue
                if normalized not in gated_set or self._qualifier_near(
                    lowered_prompt, occurrence.span()
                ):
                    yield phrase, occurrence.start()

    def vetoing_phrase(self, lowered_prompt: str) -> str | None:
        """Return the declared veto phrase this prompt names, if any (OMN-18831).

        A veto names a requested PROSE artifact or a no-code instruction ("a
        pull request description", "in prose"). The contract declares it on the
        classes graded by deterministic acceptance, where a prompt that only
        DESCRIBES code work ("the unit tests passed") would otherwise be graded
        on compilation. Presence on word boundaries, like every other phrase;
        longest first, so the reason names the most specific veto.
        """
        for phrase in sorted(self.vetoed_by, key=lambda item: (-len(item), item)):
            normalized = phrase.lower()
            if not normalized:
                continue
            for occurrence in _phrase_pattern(normalized).finditer(lowered_prompt):
                # OMN-19523: "do not write a PR description" names no prose
                # output; only a plain occurrence vetoes.
                if not is_negated(lowered_prompt, occurrence.start()):
                    return phrase
        return None

    def _qualifier_near(self, lowered_prompt: str, span: tuple[int, int]) -> bool:
        """Return whether a declared qualifier sits within the declared window.

        The phrase's own span is excluded from the search, so a phrase can
        never be its own counter-signal.
        """
        gated = self.qualified_phrases
        if gated is None:
            return False
        before, after = _windows(lowered_prompt, span, gated.within_words)
        return any(
            _phrase_pattern(qualifier.lower()).search(before) is not None
            or _phrase_pattern(qualifier.lower()).search(after) is not None
            for qualifier in gated.qualifiers
            if qualifier
        )


def _windows(text: str, span: tuple[int, int], within_words: int) -> tuple[str, str]:
    """Return the text of the ``within_words`` words on each side of ``span``."""
    start, end = span
    word_starts = [match.start() for match in _WORD.finditer(text, 0, start)]
    word_ends = [match.end() for match in _WORD.finditer(text, end)]
    left = word_starts[-within_words] if len(word_starts) >= within_words else 0
    right = word_ends[within_words - 1] if len(word_ends) >= within_words else len(text)
    return text[left:start], text[end:right]
