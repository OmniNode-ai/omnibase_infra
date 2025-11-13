#!/usr/bin/env python3
"""
Pattern Formatter for LLM Context Building.

Formats production patterns from PatternMatcher into LLM-consumable markdown
with code examples, usage guidelines, and token budget management.

This module integrates Phase 3's PatternMatcher with Phase 2's BusinessLogicGenerator
to provide rich pattern context for code generation.

Performance Target: <10ms to format 5 patterns
Token Budget: 1800-2200 tokens (400 tokens per pattern × 5 patterns)
"""

import logging
from typing import Optional

# Import from metadata_stamping where patterns are defined
from metadata_stamping.code_gen.patterns.models import (
    ModelPatternMatch,
    ModelPatternMetadata,
)

logger = logging.getLogger(__name__)


class PatternFormatter:
    """
    Format production patterns for LLM consumption.

    Takes matched patterns from PatternMatcher and formats them into
    structured markdown suitable for LLM prompts, including:
    - Pattern metadata (name, category, score)
    - Code examples from production nodes
    - Usage guidelines
    - Token budget management

    Attributes:
        max_code_lines: Maximum lines of code to include per example
        max_guidelines: Maximum number of usage guidelines per pattern

    Example:
        >>> formatter = PatternFormatter()
        >>> matches = pattern_matcher.match_patterns(...)
        >>> formatted = formatter.format_patterns_for_llm(
        ...     patterns=matches,
        ...     max_tokens=2000,
        ... )
        >>> print(formatted)  # Markdown-formatted patterns for LLM
    """

    def __init__(
        self,
        max_code_lines: int = 50,
        max_guidelines: int = 5,
    ):
        """
        Initialize pattern formatter.

        Args:
            max_code_lines: Maximum lines of code per example
            max_guidelines: Maximum usage guidelines per pattern
        """
        self.max_code_lines = max_code_lines
        self.max_guidelines = max_guidelines

        logger.debug(
            f"PatternFormatter initialized (max_code_lines={max_code_lines}, "
            f"max_guidelines={max_guidelines})"
        )

    def format_patterns_for_llm(
        self,
        patterns: list[ModelPatternMatch],
        max_tokens: int = 2000,
    ) -> str:
        """
        Format patterns for LLM context.

        Generates markdown-formatted text with:
        - Pattern ranking by score
        - Code examples from production nodes
        - Usage guidelines
        - Token budget management (truncate if needed)

        Args:
            patterns: List of pattern matches from PatternMatcher
            max_tokens: Maximum tokens to use (default: 2000)

        Returns:
            Formatted markdown string suitable for LLM prompt

        Example Output:
            ```markdown
            # Production Patterns

            ## Pattern 1: Connection Pooling (score: 0.89)
            **Category**: Integration
            **Tags**: database, connection_pooling, async

            **Description**:
            Implement connection pooling using asyncpg for PostgreSQL operations.

            **Code Example**:
            ```python
            class NodeDatabaseEffect:
                def __init__(self, container: ModelContainer):
                    self.pool: Optional[asyncpg.Pool] = None
                    ...
            ```

            **Usage Guidelines**:
            1. Initialize pool in node's `initialize()` method
            2. Use `async with self.pool.acquire() as conn:` for queries
            ...
            ```

        Performance: <10ms for 5 patterns
        """
        if not patterns:
            return "# Production Patterns\n\n(No patterns matched)"

        # Sort by score descending
        sorted_patterns = sorted(patterns, key=lambda p: p.score, reverse=True)

        # Format each pattern
        sections = []
        sections.append("# Production Patterns\n")

        for idx, match in enumerate(sorted_patterns, start=1):
            pattern_section = self._format_single_pattern(
                match=match,
                pattern_number=idx,
            )
            sections.append(pattern_section)

        # Join sections
        formatted = "\n".join(sections)

        # Check token budget (approximate: 1 token ≈ 4 characters)
        estimated_tokens = len(formatted) // 4

        if estimated_tokens > max_tokens:
            # Truncate to fit budget
            formatted = self._truncate_to_budget(
                sections=sections,
                max_tokens=max_tokens,
            )

            logger.info(
                f"Truncated patterns to fit budget: {estimated_tokens} → ~{max_tokens} tokens"
            )

        logger.debug(f"Formatted {len(patterns)} patterns (~{estimated_tokens} tokens)")

        return formatted

    def _format_single_pattern(
        self,
        match: ModelPatternMatch,
        pattern_number: int,
    ) -> str:
        """
        Format a single pattern match.

        Args:
            match: Pattern match to format
            pattern_number: Pattern number for display

        Returns:
            Formatted markdown section for this pattern
        """
        pattern = match.pattern
        sections = []

        # Header with pattern name and score
        sections.append(
            f"## Pattern {pattern_number}: {pattern.name} (score: {match.score:.2f})"
        )

        # Metadata
        sections.append(f"**Category**: {pattern.category.value.title()}")
        sections.append(f"**Tags**: {', '.join(pattern.tags[:6])}")

        if match.matched_features:
            sections.append(
                f"**Matched Features**: {', '.join(match.matched_features[:5])}"
            )

        # Description
        sections.append("\n**Description**:")
        sections.append(pattern.description)

        # Code example (if available)
        if pattern.examples:
            code_example = self._extract_code_example(pattern)
            if code_example:
                sections.append("\n**Code Example**:")
                sections.append("```python")
                sections.append(code_example)
                sections.append("```")
        elif pattern.code_template:
            # Use code template as example
            sections.append("\n**Code Template**:")
            sections.append("```python")
            sections.append(self._clean_code(pattern.code_template))
            sections.append("```")

        # Usage guidelines
        guidelines = self._generate_usage_guideline(pattern)
        if guidelines:
            sections.append("\n**Usage Guidelines**:")
            for i, guideline in enumerate(guidelines[: self.max_guidelines], start=1):
                sections.append(f"{i}. {guideline}")

        # Prerequisites (if any)
        if pattern.prerequisites:
            sections.append(
                f"\n**Prerequisites**: {', '.join(pattern.prerequisites[:3])}"
            )

        # Add spacing
        sections.append("\n---\n")

        return "\n".join(sections)

    def _extract_code_example(
        self,
        pattern: ModelPatternMetadata,
    ) -> Optional[str]:
        """
        Extract clean code example from pattern.

        Selects the first available example and cleans it for display.

        Args:
            pattern: Pattern metadata with examples

        Returns:
            Cleaned code example or None if no examples available
        """
        if not pattern.examples:
            return None

        # Get first example
        example = pattern.examples[0]
        code = example.code_snippet

        # Clean and truncate code
        cleaned = self._clean_code(code)

        return cleaned

    def _clean_code(self, code: str) -> str:
        """
        Clean code for display in LLM context.

        - Removes excessive blank lines
        - Truncates to max_code_lines
        - Preserves indentation

        Args:
            code: Raw code string

        Returns:
            Cleaned code string
        """
        # Split into lines
        lines = code.split("\n")

        # Remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Truncate if too long
        if len(lines) > self.max_code_lines:
            lines = lines[: self.max_code_lines]
            lines.append("    # ... (truncated for brevity)")

        # Remove excessive blank lines (max 1 consecutive)
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue  # Skip consecutive blank line
            cleaned_lines.append(line)
            prev_blank = is_blank

        return "\n".join(cleaned_lines)

    def _generate_usage_guideline(
        self,
        pattern: ModelPatternMetadata,
    ) -> list[str]:
        """
        Generate usage guidelines for LLM.

        Creates actionable guidelines based on pattern metadata.

        Args:
            pattern: Pattern metadata

        Returns:
            List of usage guideline strings
        """
        guidelines = []

        # Add use case guidelines
        if pattern.use_cases:
            for use_case in pattern.use_cases[:2]:
                guidelines.append(f"Use for: {use_case}")

        # Add prerequisite guidelines
        if pattern.prerequisites:
            prereq_list = ", ".join(pattern.prerequisites[:2])
            guidelines.append(f"Ensure these are available: {prereq_list}")

        # Add configuration guidelines
        if pattern.configuration:
            config_keys = list(pattern.configuration.keys())[:2]
            if config_keys:
                guidelines.append(f"Configure: {', '.join(config_keys)}")

        # Add complexity-based guidelines
        if pattern.complexity >= 4:
            guidelines.append("This is a complex pattern - test thoroughly")
        elif pattern.complexity <= 2:
            guidelines.append(
                "This is a simple pattern - good for quick implementation"
            )

        # Add category-specific guidelines
        if pattern.category.value == "resilience":
            guidelines.append("Implement error handling and retry logic")
        elif pattern.category.value == "integration":
            guidelines.append("Test with external system integration")
        elif pattern.category.value == "observability":
            guidelines.append("Include logging and metrics collection")

        # Generic guideline
        if not guidelines:
            guidelines.append(f"Follow the {pattern.name} pattern structure")

        return guidelines

    def _truncate_to_budget(
        self,
        sections: list[str],
        max_tokens: int,
    ) -> str:
        """
        Truncate sections to fit within token budget.

        Strategy:
        1. Keep header and first pattern
        2. Reduce number of patterns if needed
        3. Truncate code examples if still over budget

        Args:
            sections: List of formatted sections
            max_tokens: Maximum token budget

        Returns:
            Truncated formatted string
        """
        # Start with header
        result = [sections[0]]  # Header

        # Add patterns one by one until budget exceeded
        current_text = result[0]
        for section in sections[1:]:
            test_text = current_text + "\n" + section
            estimated_tokens = len(test_text) // 4

            if estimated_tokens > max_tokens:
                # Budget exceeded - stop adding patterns
                result.append(
                    "\n\n(Additional patterns truncated to fit token budget)\n"
                )
                break

            result.append(section)
            current_text = test_text

        return "\n".join(result)

    def format_pattern_summary(
        self,
        patterns: list[ModelPatternMatch],
    ) -> str:
        """
        Format a brief summary of matched patterns.

        Useful for logging and debugging.

        Args:
            patterns: List of pattern matches

        Returns:
            Brief summary string

        Example:
            >>> formatter.format_pattern_summary(matches)
            "5 patterns: Connection Pooling (0.89), Error Handling (0.95), ..."
        """
        if not patterns:
            return "No patterns matched"

        sorted_patterns = sorted(patterns, key=lambda p: p.score, reverse=True)

        pattern_names = [
            f"{p.pattern.name} ({p.score:.2f})" for p in sorted_patterns[:3]
        ]

        summary = f"{len(patterns)} patterns: {', '.join(pattern_names)}"

        if len(patterns) > 3:
            summary += f", +{len(patterns) - 3} more"

        return summary


__all__ = ["PatternFormatter"]
