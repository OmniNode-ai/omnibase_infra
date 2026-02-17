# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Edit-distance utilization scorer for static context attribution.

Measures how much of each static context section was actually used in a
model response. Uses edit-distance anchoring between section content
fragments and response content for evidence-based attribution.

The approach:
1. Split each section into fragments (sentences / significant lines)
2. For each fragment, search for approximate matches in the response
   using normalized edit distance
3. Score = matched_fragments / total_fragments

This is evidence-based, not heuristic. A fragment is "matched" only
when its normalized edit distance to some substring of the response
falls below a configurable threshold.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import logging
import re

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)
from omnibase_infra.services.observability.static_context_attribution.model_section_attribution import (
    ModelSectionAttribution,
)

logger = logging.getLogger(__name__)

# Minimum fragment length (characters) to consider for matching.
# Very short fragments produce too many false positives.
_MIN_FRAGMENT_LENGTH = 20

# Fragment length threshold above which we use n-gram overlap instead of
# the sliding-window Levenshtein approach.  Fragments longer than this are
# matched via n-gram overlap which is O(F + R) instead of O(R * F * min(F,W)).
_NGRAM_FALLBACK_THRESHOLD = 80

# Maximum number of Levenshtein comparisons allowed in the sliding window
# path before giving up.  This prevents unbounded computation when the
# response is very large relative to the fragment.
_MAX_LEVENSHTEIN_COMPARISONS = 5_000

# Regex to split text into sentence-like fragments
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n(?:\s*\n)+|\n(?=[-*#|>])")


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace sequences to single spaces and strip.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_fragments(content: str) -> list[str]:
    """Extract meaningful text fragments from section content.

    Splits on sentence boundaries, paragraph breaks, and list items.
    Filters out fragments shorter than ``_MIN_FRAGMENT_LENGTH``.

    Args:
        content: Section content to fragment.

    Returns:
        List of normalized text fragments.
    """
    # Split into candidate fragments
    raw_fragments = _SENTENCE_SPLIT.split(content)

    fragments: list[str] = []
    for raw in raw_fragments:
        normalized = _normalize_whitespace(raw)
        # Skip heading-only lines, very short fragments, and table separators
        if (
            len(normalized) >= _MIN_FRAGMENT_LENGTH
            and not normalized.startswith("#")
            and not re.match(r"^[\|\-\s:]+$", normalized)
        ):
            fragments.append(normalized)

    return fragments


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity ratio.

    Returns a value in [0.0, 1.0] where 1.0 means identical strings.
    Uses the standard dynamic programming approach with O(min(m,n))
    space optimization.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity ratio in [0.0, 1.0].
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Ensure s1 is the shorter string for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    # Single-row DP
    previous_row = list(range(m + 1))
    for j in range(1, n + 1):
        current_row = [j] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            current_row[i] = min(
                current_row[i - 1] + 1,  # insertion
                previous_row[i] + 1,  # deletion
                previous_row[i - 1] + cost,  # substitution
            )
        previous_row = current_row

    distance = previous_row[m]
    max_len = max(m, n)
    return 1.0 - (distance / max_len)


def _build_ngram_set(text: str, n: int = 4) -> frozenset[str]:
    """Build a set of character n-grams from text.

    Pre-computing n-gram sets allows O(1) membership tests instead of
    O(R) substring scans, and the set can be reused across multiple
    fragment comparisons against the same text.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        Frozen set of n-grams extracted from *text*.
    """
    if len(text) < n:
        return frozenset()
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def _find_best_match_in_response(
    fragment: str,
    response_normalized: str,
    threshold: float,
    response_ngrams: frozenset[str],
) -> bool:
    """Check if a fragment has a sufficiently close match in the response.

    Uses a sliding window approach over the response text. The window
    size is based on the fragment length (with some tolerance for
    insertions/deletions).

    For performance, first checks for exact substring containment,
    then falls back to approximate matching via edit distance.

    Fragments longer than 80 characters use n-gram overlap instead of
    the sliding window to avoid O(R * F * min(F,W)) Levenshtein
    computations that become prohibitively slow on moderately large inputs.

    The sliding window path is further bounded by a total comparison
    budget (``_MAX_LEVENSHTEIN_COMPARISONS``) to prevent unbounded
    computation on large responses.

    Args:
        fragment: Normalized text fragment from a section.
        response_normalized: Normalized response text.
        threshold: Minimum similarity ratio for a match.
        response_ngrams: Pre-computed n-gram set from the response,
            built once via ``_build_ngram_set`` and reused across
            all fragment comparisons.

    Returns:
        True if a sufficiently close match was found.
    """
    # Fast path: exact substring match
    if fragment in response_normalized:
        return True

    # For fragments >= 80 chars, use n-gram overlap instead of full edit
    # distance to avoid O(n*m) complexity on long strings.  The previous
    # threshold of 200 left a wide band of fragments (80-200 chars) in the
    # expensive sliding-window path.
    if len(fragment) > _NGRAM_FALLBACK_THRESHOLD:
        return _ngram_overlap(fragment, response_ngrams, threshold)

    # Sliding window approximate match
    frag_len = len(fragment)
    # Window tolerance: allow +-30% length variation
    min_window = max(1, int(frag_len * 0.7))
    max_window = int(frag_len * 1.3)

    # Step size: use frag_len // 3 to reduce total comparisons while
    # keeping sufficient overlap between windows.
    step = max(1, frag_len // 3)

    comparisons = 0
    for window_size in (frag_len, min_window, max_window):
        if window_size > len(response_normalized):
            continue
        for start in range(0, len(response_normalized) - window_size + 1, step):
            candidate = response_normalized[start : start + window_size]
            ratio = _levenshtein_ratio(fragment, candidate)
            if ratio >= threshold:
                return True
            comparisons += 1
            if comparisons >= _MAX_LEVENSHTEIN_COMPARISONS:
                return False

    return False


def _ngram_overlap(
    text1: str,
    response_ngrams: frozenset[str],
    threshold: float,
    n: int = 4,
) -> bool:
    """Check n-gram overlap between a fragment and pre-computed response n-grams.

    Used for long fragments where full edit distance is too expensive.
    Uses set intersection for O(F) complexity instead of O(F * R)
    substring scanning.

    Args:
        text1: The fragment text.
        response_ngrams: Pre-computed n-gram set from the response,
            built via ``_build_ngram_set``.
        threshold: Minimum overlap ratio.
        n: N-gram size.

    Returns:
        True if overlap ratio meets threshold.
    """
    if len(text1) < n:
        # Fragment too short for n-grams; fall back to membership check.
        # response_ngrams cannot help here, so check against any single
        # n-gram presence is not meaningful. Return False as a safe default
        # since fragments this short are below _MIN_FRAGMENT_LENGTH anyway.
        return False

    ngrams1 = {text1[i : i + n] for i in range(len(text1) - n + 1)}
    if not ngrams1:
        return False

    matched = len(ngrams1 & response_ngrams)
    overlap = matched / len(ngrams1)
    return overlap >= threshold


class ServiceUtilizationScorer:
    """Edit-distance utilization scorer for static context sections.

    Measures how much of each section's content appeared in a model
    response using fragment-level edit-distance matching.

    Args:
        similarity_threshold: Minimum normalized Levenshtein similarity
            for a fragment to be considered "matched". Default 0.7.

    Usage:
        >>> scorer = ServiceUtilizationScorer()
        >>> sections = [ModelContextSection(content="Use pytest for testing.\\nRun with: uv run pytest tests/")]
        >>> response = "I'll run the tests with uv run pytest tests/"
        >>> attributions = scorer.score(sections, response)
        >>> attributions[0].utilization_score > 0.0
        True
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        """Initialize with similarity threshold.

        Args:
            similarity_threshold: Minimum similarity for match.
                Must be in (0.0, 1.0].
        """
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0.0, 1.0], got {similarity_threshold}"
            )
        self._threshold = similarity_threshold

    def score(
        self,
        sections: list[ModelContextSection],
        response: str,
    ) -> list[ModelSectionAttribution]:
        """Score utilization of each section against a model response.

        Args:
            sections: Parsed and token-counted sections.
            response: Model response text.

        Returns:
            List of ``ModelSectionAttribution`` in same order as input.
        """
        if not response or not response.strip():
            return [
                ModelSectionAttribution(
                    section=section,
                    utilization_score=0.0,
                    matched_fragments=0,
                    total_fragments=len(_extract_fragments(section.content)),
                )
                for section in sections
            ]

        response_normalized = _normalize_whitespace(response)
        # Pre-compute response n-grams once and reuse across all sections
        # and fragments.  This avoids rebuilding the set (or doing O(R)
        # substring scans) for every fragment of every section.
        response_ngrams = _build_ngram_set(response_normalized)

        attributions: list[ModelSectionAttribution] = []
        for section in sections:
            fragments = _extract_fragments(section.content)
            total_fragments = len(fragments)

            if total_fragments == 0:
                attributions.append(
                    ModelSectionAttribution(
                        section=section,
                        utilization_score=0.0,
                        matched_fragments=0,
                        total_fragments=0,
                    )
                )
                continue

            matched = sum(
                1
                for frag in fragments
                if _find_best_match_in_response(
                    frag, response_normalized, self._threshold, response_ngrams
                )
            )

            score = matched / total_fragments

            attributions.append(
                ModelSectionAttribution(
                    section=section,
                    utilization_score=round(score, 4),
                    matched_fragments=matched,
                    total_fragments=total_fragments,
                )
            )

        total_matched = sum(a.matched_fragments for a in attributions)
        total_frags = sum(a.total_fragments for a in attributions)
        logger.debug(
            "Utilization scoring complete: %d/%d fragments matched across %d sections",
            total_matched,
            total_frags,
            len(sections),
        )

        return attributions


__all__ = ["ServiceUtilizationScorer"]
