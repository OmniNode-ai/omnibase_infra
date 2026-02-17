# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Token counting service for static context sections.

Provides token counting via a pluggable tokenizer strategy. Ships with
a word-boundary estimator (no external dependencies) that approximates
BPE token counts using the standard ~4 chars/token heuristic.

For exact counts, callers can inject a tiktoken-based counter via the
``tokenizer_fn`` parameter.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)

logger = logging.getLogger(__name__)

# Type alias for tokenizer function: text -> token count
TokenizerFn = Callable[[str], int]


def estimate_tokens_word_boundary(text: str) -> int:
    """Estimate token count using word-boundary heuristic.

    Uses the standard approximation: 1 token per ~4 characters.
    This is a reasonable estimate for English text with BPE tokenizers
    (GPT-4, Claude). The estimate is deterministic and zero-cost.

    The heuristic:
    1. Split on whitespace to get words
    2. For each word, estimate tokens as max(1, ceil(len(word) / 4))
    3. Add tokens for whitespace separators

    Args:
        text: Input text to estimate token count for.

    Returns:
        Estimated token count. 0 for empty text.
    """
    if not text or not text.strip():
        return 0

    # Split on whitespace boundaries
    words = text.split()
    if not words:
        return 0

    token_count = 0
    for word in words:
        # Each word is at least 1 token. Longer words get more.
        # ~4 chars per token for BPE tokenizers.
        word_tokens = max(1, -(-len(word) // 4))  # ceil division
        token_count += word_tokens

    # Whitespace between words contributes roughly 1 token per ~4 spaces
    # but since most separators are single spaces, they typically merge
    # into adjacent tokens. We add a small overhead for newlines and
    # multi-space sequences.
    newline_count = text.count("\n")
    token_count += newline_count  # newlines typically are their own token

    return token_count


def estimate_tokens_char_ratio(text: str) -> int:
    """Estimate token count using simple character ratio.

    A simpler estimator that divides total character count by 4.
    Less accurate for code-heavy content but very fast.

    Args:
        text: Input text to estimate token count for.

    Returns:
        Estimated token count. 0 for empty text.
    """
    if not text:
        return 0
    return max(1, -(-len(text) // 4))  # ceil(len / 4)


class ServiceTokenCounter:
    """Token counting service with pluggable tokenizer.

    Default tokenizer uses the word-boundary heuristic. For exact
    token counts, inject a tiktoken-based function:

        >>> import tiktoken
        >>> enc = tiktoken.encoding_for_model("gpt-4")
        >>> counter = ServiceTokenCounter(tokenizer_fn=lambda t: len(enc.encode(t)))

    Usage:
        >>> counter = ServiceTokenCounter()
        >>> sections = [ModelContextSection(content="Hello world")]
        >>> counted = counter.count_sections(sections)
        >>> counted[0].token_count > 0
        True
    """

    def __init__(
        self,
        tokenizer_fn: TokenizerFn | None = None,
    ) -> None:
        """Initialize with an optional custom tokenizer function.

        Args:
            tokenizer_fn: Function that takes text and returns token count.
                Defaults to ``estimate_tokens_word_boundary``.
        """
        self._tokenizer_fn = tokenizer_fn or estimate_tokens_word_boundary

    def count_text(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        return self._tokenizer_fn(text)

    def count_sections(
        self,
        sections: list[ModelContextSection],
    ) -> list[ModelContextSection]:
        """Count tokens for each section and return updated copies.

        Args:
            sections: Parsed sections to count tokens for.

        Returns:
            New list of sections with ``token_count`` populated.
        """
        counted: list[ModelContextSection] = []
        total_tokens = 0

        for section in sections:
            token_count = self._tokenizer_fn(section.content)
            counted.append(section.with_token_count(token_count))
            total_tokens += token_count

        logger.debug(
            "Counted tokens for %d sections: %d total tokens",
            len(sections),
            total_tokens,
        )
        return counted


__all__ = [
    "ServiceTokenCounter",
    "TokenizerFn",
    "estimate_tokens_char_ratio",
    "estimate_tokens_word_boundary",
]
