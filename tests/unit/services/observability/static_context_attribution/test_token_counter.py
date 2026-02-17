# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceTokenCounter and tokenizer estimators.

Tests token counting heuristics and the pluggable tokenizer interface.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)
from omnibase_infra.services.observability.static_context_attribution.service_token_counter import (
    ServiceTokenCounter,
    estimate_tokens_char_ratio,
    estimate_tokens_word_boundary,
)


@pytest.mark.unit
class TestEstimateTokensWordBoundary:
    """Tests for the word-boundary token estimator."""

    def test_empty_string(self) -> None:
        """Empty string returns 0 tokens."""
        assert estimate_tokens_word_boundary("") == 0

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns 0 tokens."""
        assert estimate_tokens_word_boundary("   \n\n  ") == 0

    def test_single_word(self) -> None:
        """Single short word is at least 1 token."""
        result = estimate_tokens_word_boundary("hello")
        assert result >= 1

    def test_multiple_words(self) -> None:
        """Multiple words produce more tokens than a single word."""
        single = estimate_tokens_word_boundary("hello")
        multiple = estimate_tokens_word_boundary("hello world foo bar")
        assert multiple > single

    def test_long_word_gets_more_tokens(self) -> None:
        """Long words are estimated as multiple tokens."""
        short = estimate_tokens_word_boundary("hi")
        long_word = estimate_tokens_word_boundary("supercalifragilisticexpialidocious")
        assert long_word > short

    def test_newlines_add_tokens(self) -> None:
        """Newlines contribute to token count."""
        no_newlines = estimate_tokens_word_boundary("hello world")
        with_newlines = estimate_tokens_word_boundary("hello\nworld")
        assert with_newlines >= no_newlines

    def test_code_block_tokens(self) -> None:
        """Code blocks produce reasonable token counts."""
        code = "```python\ndef hello():\n    return 'world'\n```"
        result = estimate_tokens_word_boundary(code)
        assert result > 0

    def test_reasonable_approximation(self) -> None:
        """Token estimate is in a reasonable range for typical text.

        The ~4 chars/token heuristic should give roughly 25 tokens
        for a 100-character sentence. We allow +/- 50% tolerance.
        """
        # ~100 chars
        text = "The quick brown fox jumps over the lazy dog and runs through the meadow at full speed."
        result = estimate_tokens_word_boundary(text)
        # Should be roughly 15-40 tokens for this text
        assert 10 <= result <= 50


@pytest.mark.unit
class TestEstimateTokensCharRatio:
    """Tests for the character-ratio token estimator."""

    def test_empty_string(self) -> None:
        """Empty string returns 0."""
        assert estimate_tokens_char_ratio("") == 0

    def test_short_string(self) -> None:
        """Short strings return at least 1."""
        assert estimate_tokens_char_ratio("hi") >= 1

    def test_proportional_to_length(self) -> None:
        """Longer text produces proportionally more tokens."""
        short = estimate_tokens_char_ratio("abc")
        long = estimate_tokens_char_ratio("a" * 100)
        assert long > short

    def test_exact_4_char_ratio(self) -> None:
        """Exactly 4 characters produces 1 token."""
        assert estimate_tokens_char_ratio("abcd") == 1

    def test_5_chars_produces_2_tokens(self) -> None:
        """5 characters produces 2 tokens (ceil(5/4))."""
        assert estimate_tokens_char_ratio("abcde") == 2


@pytest.mark.unit
class TestServiceTokenCounter:
    """Tests for the token counter service."""

    def test_default_tokenizer(self) -> None:
        """Default tokenizer produces non-zero counts for non-empty text."""
        counter = ServiceTokenCounter()
        assert counter.count_text("Hello world") > 0

    def test_custom_tokenizer(self) -> None:
        """Custom tokenizer function is used when provided."""
        # Custom tokenizer: always returns 42
        counter = ServiceTokenCounter(tokenizer_fn=lambda _: 42)
        assert counter.count_text("anything") == 42

    def test_count_sections(self) -> None:
        """count_sections updates token_count on all sections."""
        counter = ServiceTokenCounter()
        sections = [
            ModelContextSection(content="Hello world", line_start=1, line_end=1),
            ModelContextSection(
                content="Another section with more content",
                line_start=2,
                line_end=2,
            ),
        ]
        counted = counter.count_sections(sections)
        assert len(counted) == 2
        assert counted[0].token_count > 0
        assert counted[1].token_count > 0
        # Second section is longer, should have more tokens
        assert counted[1].token_count > counted[0].token_count

    def test_count_sections_preserves_other_fields(self) -> None:
        """count_sections preserves all non-token fields."""
        counter = ServiceTokenCounter()
        section = ModelContextSection(
            heading="Test",
            heading_level=2,
            content="Content here",
            source_file="test.md",
            line_start=5,
            line_end=10,
            has_code_block=True,
            has_table=False,
        )
        counted = counter.count_sections([section])
        result = counted[0]
        assert result.heading == "Test"
        assert result.heading_level == 2
        assert result.source_file == "test.md"
        assert result.line_start == 5
        assert result.line_end == 10
        assert result.has_code_block is True
        assert result.has_table is False
        assert result.token_count > 0

    def test_empty_sections_list(self) -> None:
        """Empty section list returns empty list."""
        counter = ServiceTokenCounter()
        assert counter.count_sections([]) == []

    def test_count_text_empty(self) -> None:
        """Empty text produces 0 tokens."""
        counter = ServiceTokenCounter()
        assert counter.count_text("") == 0
