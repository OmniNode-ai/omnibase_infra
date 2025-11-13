#!/usr/bin/env python3
"""
Unit tests for PatternFormatter.

Tests pattern formatting for LLM context building.
"""

import pytest

from metadata_stamping.code_gen.patterns.models import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternExample,
    ModelPatternMatch,
    ModelPatternMetadata,
)
from omninode_bridge.codegen.business_logic.llm.pattern_formatter import (
    PatternFormatter,
)


@pytest.fixture
def sample_pattern():
    """Create sample pattern for testing."""
    return ModelPatternMetadata(
        pattern_id="connection_pooling_v1",
        name="Database Connection Pooling",
        version="1.0.0",
        category=EnumPatternCategory.INTEGRATION,
        applicable_to=[EnumNodeType.EFFECT],
        tags=["database", "connection_pooling", "async", "resource_management"],
        description="Implement connection pooling using asyncpg for PostgreSQL operations.",
        prerequisites=["asyncpg", "asyncio"],
        code_template="self.pool = await asyncpg.create_pool(...)",
        examples=[
            ModelPatternExample(
                node_name="NodeDatabaseEffect",
                node_type=EnumNodeType.EFFECT,
                code_snippet="async def initialize(self):\n    self.pool = await asyncpg.create_pool()",
                description="Initialize connection pool in node",
            )
        ],
    )


@pytest.fixture
def sample_pattern_match(sample_pattern):
    """Create sample pattern match for testing."""
    return ModelPatternMatch(
        pattern=sample_pattern,
        score=0.89,
        rationale="Applicable to EFFECT nodes. Matches features: database, connection_pooling. Score: 0.89",
        matched_features=["database", "connection_pooling"],
    )


class TestPatternFormatter:
    """Test suite for PatternFormatter."""

    def test_initialization(self):
        """Test formatter initialization."""
        formatter = PatternFormatter()

        assert formatter.max_code_lines == 50
        assert formatter.max_guidelines == 5

    def test_format_single_pattern(self, sample_pattern_match):
        """Test formatting a single pattern."""
        formatter = PatternFormatter()

        patterns = [sample_pattern_match]
        formatted = formatter.format_patterns_for_llm(patterns, max_tokens=2000)

        # Check that formatted output contains expected sections
        assert "# Production Patterns" in formatted
        assert "Database Connection Pooling" in formatted
        assert "score: 0.89" in formatted
        assert "**Category**: Integration" in formatted
        assert "database, connection_pooling" in formatted

    def test_format_multiple_patterns(self, sample_pattern_match):
        """Test formatting multiple patterns."""
        formatter = PatternFormatter()

        # Create second pattern
        pattern2 = ModelPatternMetadata(
            pattern_id="error_handling_v1",
            name="Error Handling Pattern",
            version="1.0.0",
            category=EnumPatternCategory.RESILIENCE,
            applicable_to=[EnumNodeType.EFFECT, EnumNodeType.COMPUTE],
            tags=["error_handling", "resilience"],
            description="Standard error handling with ModelOnexError.",
            code_template="try:\n    pass\nexcept Exception as e:\n    raise ModelOnexError(...)",
        )

        match2 = ModelPatternMatch(
            pattern=pattern2,
            score=0.95,
            rationale="High score match",
            matched_features=["error_handling"],
        )

        patterns = [sample_pattern_match, match2]
        formatted = formatter.format_patterns_for_llm(patterns, max_tokens=3000)

        # Check both patterns are present
        assert "Database Connection Pooling" in formatted
        assert "Error Handling Pattern" in formatted
        assert "Pattern 1:" in formatted
        assert "Pattern 2:" in formatted

    def test_format_patterns_empty(self):
        """Test formatting with no patterns."""
        formatter = PatternFormatter()

        formatted = formatter.format_patterns_for_llm([], max_tokens=2000)

        assert "# Production Patterns" in formatted
        assert "(No patterns matched)" in formatted

    def test_format_pattern_summary(self, sample_pattern_match):
        """Test pattern summary generation."""
        formatter = PatternFormatter()

        patterns = [sample_pattern_match]
        summary = formatter.format_pattern_summary(patterns)

        assert "1 patterns:" in summary
        assert "Database Connection Pooling" in summary
        assert "(0.89)" in summary

    def test_format_pattern_summary_empty(self):
        """Test pattern summary with no patterns."""
        formatter = PatternFormatter()

        summary = formatter.format_pattern_summary([])

        assert summary == "No patterns matched"

    def test_code_cleaning(self):
        """Test code cleaning functionality."""
        formatter = PatternFormatter()

        # Code with excessive blank lines
        messy_code = """

        def foo():
            pass


        def bar():
            pass


        """

        cleaned = formatter._clean_code(messy_code)

        # Should remove leading/trailing blank lines
        assert not cleaned.startswith("\n\n")
        assert not cleaned.endswith("\n\n\n")

        # Should preserve function definitions
        assert "def foo():" in cleaned
        assert "def bar():" in cleaned

    def test_code_truncation(self):
        """Test code truncation for long code."""
        formatter = PatternFormatter(max_code_lines=10)

        # Generate long code
        long_code = "\n".join([f"line_{i} = {i}" for i in range(50)])

        cleaned = formatter._clean_code(long_code)

        # Should be truncated
        lines = cleaned.split("\n")
        assert len(lines) <= 11  # max_code_lines + truncation message

        # Should have truncation notice
        assert "truncated" in cleaned.lower()

    def test_token_budget_management(self, sample_pattern_match):
        """Test token budget truncation."""
        formatter = PatternFormatter()

        patterns = [sample_pattern_match]

        # Set very low token budget
        formatted = formatter.format_patterns_for_llm(patterns, max_tokens=100)

        # Should include header
        assert "# Production Patterns" in formatted

        # Estimate tokens (1 token ≈ 4 chars)
        estimated_tokens = len(formatted) // 4

        # Should be close to budget (allow 20% variance)
        assert estimated_tokens <= 120  # 100 + 20% tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
