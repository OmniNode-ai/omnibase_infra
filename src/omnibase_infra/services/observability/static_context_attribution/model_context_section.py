# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pydantic model for a parsed static context section.

A section is a deterministic heading-delimited block extracted from
markdown-formatted static context (CLAUDE.md, memory files, etc.).

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)


class ModelContextSection(BaseModel):
    """A single section parsed from static context markdown.

    Sections are heading-delimited blocks. The parser splits on H2 (``##``)
    and H3 (``###``) boundaries to produce reproducible, deterministic
    sections.

    Attributes:
        heading: The heading text (without ``#`` prefix). Empty string for
            the preamble (content before the first heading).
        heading_level: Heading depth (2 for H2, 3 for H3). 0 for preamble.
        content: Full section content including the heading line.
        source_file: Origin file path or identifier.
        line_start: 1-based line number where the section starts.
        line_end: 1-based line number where the section ends (inclusive).
        token_count: Number of tokens in this section (0 until counted).
        category: Semantic category (UNCATEGORIZED until LLM pass).
        has_code_block: Whether section contains fenced code blocks.
        has_table: Whether section contains markdown tables.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    heading: str = Field(
        default="",
        description="Heading text without '#' prefix. Empty for preamble.",
    )
    heading_level: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Heading depth (2=H2, 3=H3). 0 for preamble.",
    )
    content: str = Field(
        ...,
        description="Full section content including heading line.",
    )
    source_file: str = Field(
        default="",
        description="Origin file path or identifier.",
    )
    line_start: int = Field(
        default=1,
        ge=1,
        description="1-based line number where section starts.",
    )
    line_end: int = Field(
        default=1,
        ge=1,
        description="1-based line number where section ends (inclusive).",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in this section. 0 until counted.",
    )
    category: EnumContextSectionCategory = Field(
        default=EnumContextSectionCategory.UNCATEGORIZED,
        description="Semantic category. UNCATEGORIZED until LLM pass.",
    )
    has_code_block: bool = Field(
        default=False,
        description="Whether section contains fenced code blocks.",
    )
    has_table: bool = Field(
        default=False,
        description="Whether section contains markdown tables.",
    )

    def with_token_count(self, count: int) -> ModelContextSection:
        """Return a copy with the given token count.

        Args:
            count: Token count to assign.

        Returns:
            New ModelContextSection with updated token_count.
        """
        return self.model_copy(update={"token_count": count})

    def with_category(
        self, category: EnumContextSectionCategory
    ) -> ModelContextSection:
        """Return a copy with the given semantic category.

        Args:
            category: Semantic category to assign.

        Returns:
            New ModelContextSection with updated category.
        """
        return self.model_copy(update={"category": category})


__all__ = ["ModelContextSection"]
