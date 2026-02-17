# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceStaticContextParser.

Tests deterministic heading-based markdown parsing into sections.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.observability.static_context_attribution.service_static_context_parser import (
    ServiceStaticContextParser,
)


@pytest.mark.unit
class TestServiceStaticContextParser:
    """Tests for the deterministic markdown parser."""

    def setup_method(self) -> None:
        """Create parser instance for each test."""
        self.parser = ServiceStaticContextParser()

    # ------------------------------------------------------------------
    # Basic parsing
    # ------------------------------------------------------------------

    def test_empty_content_returns_empty_list(self) -> None:
        """Empty input produces no sections."""
        assert self.parser.parse("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        """Whitespace-only input produces no sections."""
        assert self.parser.parse("   \n\n  ") == []

    def test_single_h2_section(self) -> None:
        """Single H2 heading produces one section."""
        content = "## Overview\nSome content here"
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert sections[0].heading == "Overview"
        assert sections[0].heading_level == 2
        assert "Some content here" in sections[0].content

    def test_multiple_h2_sections(self) -> None:
        """Multiple H2 headings produce multiple sections."""
        content = "## First\nContent 1\n## Second\nContent 2\n## Third\nContent 3"
        sections = self.parser.parse(content)
        assert len(sections) == 3
        assert sections[0].heading == "First"
        assert sections[1].heading == "Second"
        assert sections[2].heading == "Third"

    def test_h3_sections(self) -> None:
        """H3 headings also split sections."""
        content = "### Sub-section\nContent here\n### Another\nMore content"
        sections = self.parser.parse(content)
        assert len(sections) == 2
        assert sections[0].heading == "Sub-section"
        assert sections[0].heading_level == 3
        assert sections[1].heading == "Another"

    def test_mixed_h2_h3_sections(self) -> None:
        """Mixed H2 and H3 headings produce correct sections."""
        content = "## Main\nIntro\n### Detail\nDetail content\n## Next\nMore"
        sections = self.parser.parse(content)
        assert len(sections) == 3
        assert sections[0].heading == "Main"
        assert sections[0].heading_level == 2
        assert sections[1].heading == "Detail"
        assert sections[1].heading_level == 3
        assert sections[2].heading == "Next"

    # ------------------------------------------------------------------
    # Preamble handling
    # ------------------------------------------------------------------

    def test_preamble_before_first_heading(self) -> None:
        """Content before the first heading is captured as preamble."""
        content = "This is preamble text\n\n## First Section\nContent"
        sections = self.parser.parse(content)
        assert len(sections) == 2
        assert sections[0].heading == ""
        assert sections[0].heading_level == 0
        assert "preamble" in sections[0].content
        assert sections[1].heading == "First Section"

    def test_no_headings_produces_single_preamble(self) -> None:
        """Content without headings is a single preamble section."""
        content = "Just some text\nwith multiple lines\nand stuff"
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert sections[0].heading == ""
        assert sections[0].heading_level == 0

    # ------------------------------------------------------------------
    # Code fence handling
    # ------------------------------------------------------------------

    def test_headings_inside_code_fences_ignored(self) -> None:
        """Headings inside fenced code blocks are not section boundaries."""
        content = (
            "## Real Section\n"
            "```python\n"
            "## This is a comment in code\n"
            "### Not a heading\n"
            "```\n"
            "## Next Section\n"
            "Content"
        )
        sections = self.parser.parse(content)
        assert len(sections) == 2
        assert sections[0].heading == "Real Section"
        assert sections[1].heading == "Next Section"
        # Code content should be in first section
        assert "## This is a comment in code" in sections[0].content

    def test_code_block_detection(self) -> None:
        """Sections with code blocks are flagged."""
        content = "## Code Example\n```bash\necho hello\n```\nDone"
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert sections[0].has_code_block is True

    def test_section_without_code_block(self) -> None:
        """Sections without code blocks have has_code_block=False."""
        content = "## Plain Text\nNo code here"
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert sections[0].has_code_block is False

    # ------------------------------------------------------------------
    # Table detection
    # ------------------------------------------------------------------

    def test_table_detection(self) -> None:
        """Sections with markdown tables are flagged."""
        content = (
            "## Table Section\n"
            "| Header | Value |\n"
            "|--------|-------|\n"
            "| A      | B     |\n"
        )
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert sections[0].has_table is True

    def test_section_without_table(self) -> None:
        """Sections without tables have has_table=False."""
        content = "## No Table\nJust text"
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert sections[0].has_table is False

    # ------------------------------------------------------------------
    # Line numbers
    # ------------------------------------------------------------------

    def test_line_numbers_are_correct(self) -> None:
        """Line numbers accurately reflect section positions."""
        content = "## First\nLine 2\n## Second\nLine 4\nLine 5"
        sections = self.parser.parse(content)
        assert sections[0].line_start == 1
        assert sections[0].line_end == 2
        assert sections[1].line_start == 3
        assert sections[1].line_end == 5

    # ------------------------------------------------------------------
    # Source file tracking
    # ------------------------------------------------------------------

    def test_source_file_propagated(self) -> None:
        """Source file path is recorded in sections."""
        content = "## Test\nContent"
        sections = self.parser.parse(content, source_file="/path/to/CLAUDE.md")
        assert sections[0].source_file == "/path/to/CLAUDE.md"

    # ------------------------------------------------------------------
    # H1 and H4+ handling
    # ------------------------------------------------------------------

    def test_h1_headings_do_not_split(self) -> None:
        """H1 headings do not create section boundaries (only H2/H3)."""
        content = "# Title\nPreamble\n## Section\nContent"
        sections = self.parser.parse(content)
        assert len(sections) == 2
        assert sections[0].heading == ""  # preamble includes H1
        assert sections[1].heading == "Section"

    def test_h4_headings_do_not_split(self) -> None:
        """H4+ headings do not create section boundaries."""
        content = "## Main\nContent\n#### Sub-detail\nMore content"
        sections = self.parser.parse(content)
        assert len(sections) == 1
        assert "Sub-detail" in sections[0].content

    # ------------------------------------------------------------------
    # parse_multiple
    # ------------------------------------------------------------------

    def test_parse_multiple_files(self) -> None:
        """parse_multiple combines sections from multiple files."""
        files = {
            "file1.md": "## A\nContent A",
            "file2.md": "## B\nContent B\n## C\nContent C",
        }
        sections = self.parser.parse_multiple(files)
        assert len(sections) == 3
        assert sections[0].source_file == "file1.md"
        assert sections[1].source_file == "file2.md"
        assert sections[2].source_file == "file2.md"

    def test_parse_multiple_empty_files(self) -> None:
        """Empty files in parse_multiple are skipped."""
        files = {
            "empty.md": "",
            "real.md": "## Content\nHere",
        }
        sections = self.parser.parse_multiple(files)
        assert len(sections) == 1
        assert sections[0].source_file == "real.md"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_heading_with_special_characters(self) -> None:
        """Headings with special characters are preserved."""
        content = "## C/C++ & Python: A Comparison\nContent"
        sections = self.parser.parse(content)
        assert sections[0].heading == "C/C++ & Python: A Comparison"

    def test_consecutive_headings(self) -> None:
        """Consecutive headings without content between them still produce sections."""
        # The first heading will have content (just the heading line itself).
        # But the second heading-only section also has content (the heading line).
        content = "## First\n## Second\nActual content"
        sections = self.parser.parse(content)
        # "## First" alone is its content; "## Second\nActual content" is the next
        assert len(sections) == 2

    def test_unclosed_code_fence(self) -> None:
        """Unclosed code fence does not crash; remaining lines stay in section."""
        content = "## Section\n```python\ndef foo():\n    pass\n## Not a heading"
        sections = self.parser.parse(content)
        # Unclosed fence means "## Not a heading" is inside the fence
        assert len(sections) == 1
        assert "Not a heading" in sections[0].content
