# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Deterministic heading-based parser for static context sections.

Pass 1 of the two-pass architecture. Splits markdown-formatted static
context (CLAUDE.md, memory files) by H2/H3 headings into reproducible
sections. Zero cost, fully deterministic.

The parser handles:
    - H2 (``##``) and H3 (``###``) heading boundaries
    - Fenced code blocks (triple backticks) within sections
    - Markdown tables within sections
    - Preamble content before the first heading

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import logging
import re

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)

logger = logging.getLogger(__name__)

# Regex for heading lines: starts with ## or ### (but not ####+)
_HEADING_PATTERN = re.compile(r"^(#{2,3})\s+(.+)$")

# Regex for fenced code block delimiter
_FENCE_PATTERN = re.compile(r"^```")

# Regex for table row (starts with |)
_TABLE_PATTERN = re.compile(r"^\|.+\|")


class ServiceStaticContextParser:
    """Deterministic parser that splits markdown into heading-delimited sections.

    Sections are split on H2 (``##``) and H3 (``###``) boundaries.
    Content before the first heading is captured as a preamble section
    with ``heading_level=0``.

    Fenced code blocks are tracked but do not split sections. Heading-like
    lines inside fenced code blocks are correctly ignored.

    Usage:
        >>> parser = ServiceStaticContextParser()
        >>> sections = parser.parse("## Overview\\nSome content\\n## Next\\nMore content")
        >>> len(sections)
        2
        >>> sections[0].heading
        'Overview'
    """

    def parse(
        self,
        content: str,
        source_file: str = "",
    ) -> list[ModelContextSection]:
        """Parse markdown content into heading-delimited sections.

        Args:
            content: Markdown content to parse.
            source_file: Optional source file path for provenance.

        Returns:
            List of ``ModelContextSection`` in document order.
            Empty list if content is empty or whitespace-only.
        """
        if not content or not content.strip():
            return []

        lines = content.split("\n")
        sections: list[ModelContextSection] = []

        # Accumulator state
        current_heading = ""
        current_level = 0
        current_lines: list[str] = []
        section_start = 1
        in_code_fence = False
        has_code_block = False
        has_table = False

        for line_idx, line in enumerate(lines):
            line_num = line_idx + 1  # 1-based

            # Track fenced code blocks to ignore headings inside them
            if _FENCE_PATTERN.match(line.strip()):
                in_code_fence = not in_code_fence
                if not in_code_fence:
                    # Just closed a code fence
                    has_code_block = True
                else:
                    has_code_block = True
                current_lines.append(line)
                continue

            # Inside a code fence, all lines belong to current section
            if in_code_fence:
                current_lines.append(line)
                continue

            # Check for heading boundary
            heading_match = _HEADING_PATTERN.match(line.strip())
            if heading_match:
                # Flush the previous section if it has content
                if current_lines:
                    section_content = "\n".join(current_lines)
                    if section_content.strip():
                        sections.append(
                            ModelContextSection(
                                heading=current_heading,
                                heading_level=current_level,
                                content=section_content,
                                source_file=source_file,
                                line_start=section_start,
                                line_end=line_num - 1,
                                has_code_block=has_code_block,
                                has_table=has_table,
                            )
                        )

                # Start new section
                hashes, heading_text = heading_match.groups()
                current_heading = heading_text.strip()
                current_level = len(hashes)
                current_lines = [line]
                section_start = line_num
                has_code_block = False
                has_table = False
                continue

            # Track tables
            if _TABLE_PATTERN.match(line.strip()):
                has_table = True

            current_lines.append(line)

        # Flush final section
        if current_lines:
            section_content = "\n".join(current_lines)
            if section_content.strip():
                sections.append(
                    ModelContextSection(
                        heading=current_heading,
                        heading_level=current_level,
                        content=section_content,
                        source_file=source_file,
                        line_start=section_start,
                        line_end=len(lines),
                        has_code_block=has_code_block,
                        has_table=has_table,
                    )
                )

        logger.debug(
            "Parsed %d sections from %s (%d lines)",
            len(sections),
            source_file or "<inline>",
            len(lines),
        )
        return sections

    def parse_multiple(
        self,
        files: dict[str, str],
    ) -> list[ModelContextSection]:
        """Parse multiple files and combine sections.

        Args:
            files: Mapping of file path to markdown content.

        Returns:
            Combined list of sections from all files, in file order.
        """
        all_sections: list[ModelContextSection] = []
        for source_file, content in files.items():
            all_sections.extend(self.parse(content, source_file=source_file))
        return all_sections


__all__ = ["ServiceStaticContextParser"]
