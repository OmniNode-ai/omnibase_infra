# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceUtilizationScorer.

Tests edit-distance utilization scoring for static context attribution.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)
from omnibase_infra.services.observability.static_context_attribution.service_utilization_scorer import (
    ServiceUtilizationScorer,
    _extract_fragments,
    _levenshtein_ratio,
    _normalize_whitespace,
)


@pytest.mark.unit
class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_collapses_multiple_spaces(self) -> None:
        """Multiple spaces collapse to single space."""
        assert _normalize_whitespace("hello   world") == "hello world"

    def test_collapses_newlines(self) -> None:
        """Newlines collapse to single space."""
        assert _normalize_whitespace("hello\n\nworld") == "hello world"

    def test_strips_edges(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_lowercases(self) -> None:
        """Text is lowercased."""
        assert _normalize_whitespace("Hello World") == "hello world"


@pytest.mark.unit
class TestLevenshteinRatio:
    """Tests for the Levenshtein similarity ratio."""

    def test_identical_strings(self) -> None:
        """Identical strings have ratio 1.0."""
        assert _levenshtein_ratio("hello", "hello") == 1.0

    def test_completely_different(self) -> None:
        """Completely different strings have low ratio."""
        ratio = _levenshtein_ratio("abc", "xyz")
        assert ratio < 0.5

    def test_empty_strings(self) -> None:
        """Two empty strings are identical."""
        assert _levenshtein_ratio("", "") == 1.0

    def test_one_empty(self) -> None:
        """One empty string gives ratio 0.0."""
        assert _levenshtein_ratio("hello", "") == 0.0
        assert _levenshtein_ratio("", "hello") == 0.0

    def test_similar_strings(self) -> None:
        """Similar strings have high ratio."""
        ratio = _levenshtein_ratio("kitten", "sitting")
        assert ratio > 0.5

    def test_single_char_difference(self) -> None:
        """Single character difference gives high ratio for longer strings."""
        ratio = _levenshtein_ratio("hello world", "helo world")
        assert ratio > 0.8

    def test_symmetric(self) -> None:
        """Ratio is symmetric."""
        r1 = _levenshtein_ratio("abc", "axc")
        r2 = _levenshtein_ratio("axc", "abc")
        assert abs(r1 - r2) < 1e-10


@pytest.mark.unit
class TestExtractFragments:
    """Tests for fragment extraction."""

    def test_empty_content(self) -> None:
        """Empty content produces no fragments."""
        assert _extract_fragments("") == []

    def test_short_content_filtered(self) -> None:
        """Very short content is filtered out."""
        assert _extract_fragments("Hi") == []

    def test_sentence_extraction(self) -> None:
        """Sentences are extracted as fragments."""
        content = (
            "This is the first sentence with enough characters to pass the filter. "
            "This is the second sentence also with enough characters to pass."
        )
        fragments = _extract_fragments(content)
        assert len(fragments) >= 1

    def test_heading_lines_filtered(self) -> None:
        """Lines starting with # are filtered."""
        content = "## Heading\nThis is actual content with enough characters to pass the minimum length filter."
        fragments = _extract_fragments(content)
        for frag in fragments:
            assert not frag.startswith("#")

    def test_fragments_are_normalized(self) -> None:
        """Extracted fragments are normalized (lowercase, single spaces)."""
        content = "This  is   content   with   MIXED   case and extra spaces that should be normalized for matching."
        fragments = _extract_fragments(content)
        for frag in fragments:
            assert frag == frag.lower()
            assert "  " not in frag


@pytest.mark.unit
class TestServiceUtilizationScorer:
    """Tests for the utilization scorer service."""

    def setup_method(self) -> None:
        """Create scorer instance for each test."""
        self.scorer = ServiceUtilizationScorer()

    def test_invalid_threshold(self) -> None:
        """Threshold outside (0.0, 1.0] raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            ServiceUtilizationScorer(similarity_threshold=0.0)
        with pytest.raises(ValueError, match="similarity_threshold"):
            ServiceUtilizationScorer(similarity_threshold=1.5)

    def test_empty_response(self) -> None:
        """Empty response gives 0 utilization for all sections."""
        sections = [
            ModelContextSection(
                content="This is a sufficiently long section with enough text to produce fragments for testing.",
                line_start=1,
                line_end=1,
            )
        ]
        attributions = self.scorer.score(sections, "")
        assert len(attributions) == 1
        assert attributions[0].utilization_score == 0.0

    def test_exact_match_high_score(self) -> None:
        """Content from a section appearing exactly in response gets high score."""
        section_text = (
            "Always use uv run for all Python commands including pytest and mypy. "
            "Never use direct pip or python commands in this repository."
        )
        sections = [
            ModelContextSection(
                content=section_text,
                line_start=1,
                line_end=2,
                token_count=25,
            )
        ]
        # Response includes the exact content
        response = (
            "I'll follow the guidelines. "
            "Always use uv run for all Python commands including pytest and mypy. "
            "I will never use direct pip or python commands in this repository."
        )
        attributions = self.scorer.score(sections, response)
        assert attributions[0].utilization_score > 0.0

    def test_no_match_zero_score(self) -> None:
        """Completely unrelated response gives 0 utilization."""
        sections = [
            ModelContextSection(
                content="PostgreSQL connection configuration requires specific host and port settings for the remote infrastructure server.",
                line_start=1,
                line_end=1,
                token_count=20,
            )
        ]
        response = "The weather today is sunny and warm with a gentle breeze from the ocean bringing refreshing air."
        attributions = self.scorer.score(sections, response)
        assert attributions[0].utilization_score == 0.0

    def test_multiple_sections_scored_independently(self) -> None:
        """Each section gets its own utilization score."""
        sections = [
            ModelContextSection(
                heading="Database Config",
                content="Configure PostgreSQL with host 192.168.86.200 and port 5436 for external connections to the remote database server.",
                line_start=1,
                line_end=1,
                token_count=20,
            ),
            ModelContextSection(
                heading="Python Standards",
                content="Use ruff for linting and formatting as it replaces both black and isort in this project configuration.",
                line_start=2,
                line_end=2,
                token_count=15,
            ),
        ]
        # Response only references the second section
        response = "I will use ruff for linting and formatting as it replaces both black and isort in this project configuration."
        attributions = self.scorer.score(sections, response)
        assert len(attributions) == 2
        # Second section should have higher utilization
        assert attributions[1].utilization_score >= attributions[0].utilization_score

    def test_attributed_tokens_computed(self) -> None:
        """attributed_tokens is score * token_count."""
        sections = [
            ModelContextSection(
                content="Some sufficiently long content that will generate at least one fragment for utilization scoring purposes.",
                line_start=1,
                line_end=1,
                token_count=100,
            )
        ]
        response = "Some sufficiently long content that will generate at least one fragment for utilization scoring purposes."
        attributions = self.scorer.score(sections, response)
        attr = attributions[0]
        expected = round(attr.utilization_score * 100)
        assert attr.attributed_tokens == expected

    def test_section_with_no_fragments(self) -> None:
        """Section with only short content (no valid fragments) gets 0 score."""
        sections = [
            ModelContextSection(
                content="## H\nShort",
                line_start=1,
                line_end=2,
                token_count=5,
            )
        ]
        attributions = self.scorer.score(sections, "Some long response text")
        assert attributions[0].utilization_score == 0.0
        assert attributions[0].total_fragments == 0

    def test_custom_threshold(self) -> None:
        """Custom similarity threshold affects matching strictness."""
        # Very strict threshold
        strict_scorer = ServiceUtilizationScorer(similarity_threshold=0.99)
        # Very lenient threshold
        lenient_scorer = ServiceUtilizationScorer(similarity_threshold=0.3)

        sections = [
            ModelContextSection(
                content="Use pytest as the test framework for all unit and integration test execution.",
                line_start=1,
                line_end=1,
                token_count=15,
            )
        ]
        # Response is a paraphrase
        response = "Run the pytest framework to execute all unit and integration tests."

        strict_attrs = strict_scorer.score(sections, response)
        lenient_attrs = lenient_scorer.score(sections, response)

        # Lenient should match at least as much as strict
        assert lenient_attrs[0].utilization_score >= strict_attrs[0].utilization_score
