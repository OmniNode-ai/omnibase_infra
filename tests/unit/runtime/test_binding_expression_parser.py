# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for BindingExpressionParser.

This module provides comprehensive tests for the BindingExpressionParser class,
covering:

- Happy path parsing for all valid sources (payload, envelope, context)
- Guardrail enforcement (max length, max segments, array access, empty segments)
- Malformed expression detection
- Edge cases and boundary conditions

.. versionadded:: 0.2.6
    Created as part of OMN-1518 Phase 6 - Testing.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.binding_resolver import (
    MAX_EXPRESSION_LENGTH,
    MAX_PATH_SEGMENTS,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
    BindingExpressionParser,
)


class TestBindingExpressionParserHappyPath:
    """Happy path tests for BindingExpressionParser.parse()."""

    @pytest.fixture
    def parser(self) -> BindingExpressionParser:
        """Create a fresh parser instance for each test."""
        return BindingExpressionParser()

    # -------------------------------------------------------------------------
    # Payload source tests
    # -------------------------------------------------------------------------

    def test_parse_payload_single_segment(
        self, parser: BindingExpressionParser
    ) -> None:
        """${payload.id} parses to source='payload' with single segment."""
        source, segments = parser.parse("${payload.id}")
        assert source == "payload"
        assert segments == ("id",)

    def test_parse_payload_two_segments(self, parser: BindingExpressionParser) -> None:
        """${payload.user.id} parses to source='payload' with two segments."""
        source, segments = parser.parse("${payload.user.id}")
        assert source == "payload"
        assert segments == ("user", "id")

    def test_parse_payload_three_segments(
        self, parser: BindingExpressionParser
    ) -> None:
        """${payload.user.profile.email} parses correctly with three segments."""
        source, segments = parser.parse("${payload.user.profile.email}")
        assert source == "payload"
        assert segments == ("user", "profile", "email")

    def test_parse_payload_deep_nested_path(
        self, parser: BindingExpressionParser
    ) -> None:
        """${payload.a.b.c.d.e} parses correctly with five segments."""
        source, segments = parser.parse("${payload.a.b.c.d.e}")
        assert source == "payload"
        assert segments == ("a", "b", "c", "d", "e")

    def test_parse_payload_with_underscores(
        self, parser: BindingExpressionParser
    ) -> None:
        """${payload.user_id} parses field names with underscores."""
        source, segments = parser.parse("${payload.user_id}")
        assert source == "payload"
        assert segments == ("user_id",)

    def test_parse_payload_with_numbers(self, parser: BindingExpressionParser) -> None:
        """${payload.field123} parses field names with numbers."""
        source, segments = parser.parse("${payload.field123}")
        assert source == "payload"
        assert segments == ("field123",)

    def test_parse_payload_mixed_case(self, parser: BindingExpressionParser) -> None:
        """${payload.userId} parses camelCase field names."""
        source, segments = parser.parse("${payload.userId}")
        assert source == "payload"
        assert segments == ("userId",)

    # -------------------------------------------------------------------------
    # Envelope source tests
    # -------------------------------------------------------------------------

    def test_parse_envelope_correlation_id(
        self, parser: BindingExpressionParser
    ) -> None:
        """${envelope.correlation_id} parses correctly."""
        source, segments = parser.parse("${envelope.correlation_id}")
        assert source == "envelope"
        assert segments == ("correlation_id",)

    def test_parse_envelope_single_segment(
        self, parser: BindingExpressionParser
    ) -> None:
        """${envelope.timestamp} parses envelope with single segment."""
        source, segments = parser.parse("${envelope.timestamp}")
        assert source == "envelope"
        assert segments == ("timestamp",)

    def test_parse_envelope_nested(self, parser: BindingExpressionParser) -> None:
        """${envelope.metadata.version} parses nested envelope fields."""
        source, segments = parser.parse("${envelope.metadata.version}")
        assert source == "envelope"
        assert segments == ("metadata", "version")

    # -------------------------------------------------------------------------
    # Context source tests
    # -------------------------------------------------------------------------

    def test_parse_context_now_iso(self, parser: BindingExpressionParser) -> None:
        """${context.now_iso} parses correctly."""
        source, segments = parser.parse("${context.now_iso}")
        assert source == "context"
        assert segments == ("now_iso",)

    def test_parse_context_dispatcher_id(self, parser: BindingExpressionParser) -> None:
        """${context.dispatcher_id} parses correctly."""
        source, segments = parser.parse("${context.dispatcher_id}")
        assert source == "context"
        assert segments == ("dispatcher_id",)

    def test_parse_context_correlation_id(
        self, parser: BindingExpressionParser
    ) -> None:
        """${context.correlation_id} parses correctly."""
        source, segments = parser.parse("${context.correlation_id}")
        assert source == "context"
        assert segments == ("correlation_id",)

    def test_parse_all_valid_context_paths(
        self, parser: BindingExpressionParser
    ) -> None:
        """All valid context paths parse correctly."""
        for ctx_path in VALID_CONTEXT_PATHS:
            source, segments = parser.parse(f"${{context.{ctx_path}}}")
            assert source == "context", f"Failed for context path: {ctx_path}"
            assert segments[0] == ctx_path, f"Failed for context path: {ctx_path}"

    # -------------------------------------------------------------------------
    # Boundary tests
    # -------------------------------------------------------------------------

    def test_parse_max_segments_exactly(self, parser: BindingExpressionParser) -> None:
        """Path with exactly MAX_PATH_SEGMENTS segments parses correctly."""
        # Create path with exactly MAX_PATH_SEGMENTS segments
        path = ".".join(["a"] * MAX_PATH_SEGMENTS)
        expression = f"${{payload.{path}}}"
        source, segments = parser.parse(expression)
        assert source == "payload"
        assert len(segments) == MAX_PATH_SEGMENTS

    def test_parse_max_length_exactly(self, parser: BindingExpressionParser) -> None:
        """Expression at exactly MAX_EXPRESSION_LENGTH parses correctly."""
        # Calculate padding needed: ${payload.xxx...} = 10 chars + padding
        # ${payload.} = 10 chars, } = 1 char, so padding = MAX - 11
        prefix = "${payload."
        suffix = "}"
        padding_length = MAX_EXPRESSION_LENGTH - len(prefix) - len(suffix)
        path = "a" * padding_length
        expression = f"{prefix}{path}{suffix}"
        assert len(expression) == MAX_EXPRESSION_LENGTH
        source, _segments = parser.parse(expression)
        assert source == "payload"


class TestBindingExpressionParserGuardrails:
    """Guardrail enforcement tests for BindingExpressionParser."""

    @pytest.fixture
    def parser(self) -> BindingExpressionParser:
        """Create a fresh parser instance for each test."""
        return BindingExpressionParser()

    # -------------------------------------------------------------------------
    # Invalid source tests
    # -------------------------------------------------------------------------

    def test_invalid_source_fails(self, parser: BindingExpressionParser) -> None:
        """${invalid.path} raises ValueError with 'Invalid source' message."""
        with pytest.raises(ValueError, match="Invalid source"):
            parser.parse("${invalid.path}")

    def test_unknown_source_fails(self, parser: BindingExpressionParser) -> None:
        """${unknown.field} raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source"):
            parser.parse("${unknown.field}")

    def test_capitalized_source_fails(self, parser: BindingExpressionParser) -> None:
        """${Payload.id} fails because source must be lowercase."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${Payload.id}")

    def test_uppercase_source_fails(self, parser: BindingExpressionParser) -> None:
        """${PAYLOAD.id} fails because source must be lowercase."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${PAYLOAD.id}")

    # -------------------------------------------------------------------------
    # Max segments tests
    # -------------------------------------------------------------------------

    def test_path_too_deep_fails(self, parser: BindingExpressionParser) -> None:
        """More than MAX_PATH_SEGMENTS segments raises ValueError."""
        # Create path with MAX_PATH_SEGMENTS + 1 segments
        deep_path = ".".join(["a"] * (MAX_PATH_SEGMENTS + 1))
        with pytest.raises(ValueError, match="max segments"):
            parser.parse(f"${{payload.{deep_path}}}")

    def test_path_way_too_deep_fails(self, parser: BindingExpressionParser) -> None:
        """Path with 50 segments (well over limit) raises ValueError."""
        deep_path = ".".join(["a"] * 50)
        with pytest.raises(ValueError, match="max segments"):
            parser.parse(f"${{payload.{deep_path}}}")

    # -------------------------------------------------------------------------
    # Max length tests
    # -------------------------------------------------------------------------

    def test_expression_too_long_fails(self, parser: BindingExpressionParser) -> None:
        """Expression exceeding MAX_EXPRESSION_LENGTH raises ValueError."""
        long_path = "a" * 250
        expression = f"${{payload.{long_path}}}"
        assert len(expression) > MAX_EXPRESSION_LENGTH
        with pytest.raises(ValueError, match="max length"):
            parser.parse(expression)

    def test_expression_way_too_long_fails(
        self, parser: BindingExpressionParser
    ) -> None:
        """Expression at 500+ chars raises ValueError."""
        long_path = "a" * 500
        expression = f"${{payload.{long_path}}}"
        with pytest.raises(ValueError, match="max length"):
            parser.parse(expression)

    # -------------------------------------------------------------------------
    # Array access tests
    # -------------------------------------------------------------------------

    def test_array_access_fails(self, parser: BindingExpressionParser) -> None:
        """${payload.items[0]} raises ValueError for array access."""
        with pytest.raises(ValueError, match="Array access"):
            parser.parse("${payload.items[0]}")

    def test_wildcard_array_fails(self, parser: BindingExpressionParser) -> None:
        """${payload.items[*].id} raises ValueError for wildcard array."""
        with pytest.raises(ValueError, match="Array access"):
            parser.parse("${payload.items[*].id}")

    def test_negative_array_index_fails(self, parser: BindingExpressionParser) -> None:
        """${payload.items[-1]} raises ValueError for negative index."""
        with pytest.raises(ValueError, match="Array access"):
            parser.parse("${payload.items[-1]}")

    def test_bracket_in_middle_fails(self, parser: BindingExpressionParser) -> None:
        """${payload.data[key].value} raises ValueError."""
        with pytest.raises(ValueError, match="Array access"):
            parser.parse("${payload.data[key].value}")

    def test_empty_brackets_fail(self, parser: BindingExpressionParser) -> None:
        """${payload.items[]} raises ValueError."""
        with pytest.raises(ValueError, match="Array access"):
            parser.parse("${payload.items[]}")

    # -------------------------------------------------------------------------
    # Empty segment tests
    # -------------------------------------------------------------------------

    def test_empty_segment_fails(self, parser: BindingExpressionParser) -> None:
        """${payload..id} raises ValueError for empty segment."""
        with pytest.raises(ValueError, match="Empty path segment"):
            parser.parse("${payload..id}")

    def test_multiple_empty_segments_fail(
        self, parser: BindingExpressionParser
    ) -> None:
        """${payload...id} raises ValueError."""
        with pytest.raises(ValueError, match="Empty path segment"):
            parser.parse("${payload...id}")

    def test_trailing_dot_fails(self, parser: BindingExpressionParser) -> None:
        """${payload.id.} raises ValueError for trailing dot."""
        with pytest.raises(ValueError, match="Empty path segment"):
            parser.parse("${payload.id.}")

    # -------------------------------------------------------------------------
    # Invalid context path tests
    # -------------------------------------------------------------------------

    def test_invalid_context_path_fails(self, parser: BindingExpressionParser) -> None:
        """${context.unknown_field} raises ValueError."""
        with pytest.raises(ValueError, match="Invalid context path"):
            parser.parse("${context.unknown_field}")

    def test_context_typo_fails(self, parser: BindingExpressionParser) -> None:
        """${context.now_isos} (typo) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid context path"):
            parser.parse("${context.now_isos}")

    def test_context_arbitrary_path_fails(
        self, parser: BindingExpressionParser
    ) -> None:
        """${context.arbitrary.nested.path} raises ValueError."""
        with pytest.raises(ValueError, match="Invalid context path"):
            parser.parse("${context.arbitrary.nested.path}")


class TestBindingExpressionParserMalformedExpressions:
    """Malformed expression tests for BindingExpressionParser."""

    @pytest.fixture
    def parser(self) -> BindingExpressionParser:
        """Create a fresh parser instance for each test."""
        return BindingExpressionParser()

    # -------------------------------------------------------------------------
    # Missing syntax elements
    # -------------------------------------------------------------------------

    def test_missing_dollar_sign_fails(self, parser: BindingExpressionParser) -> None:
        """Missing $ raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("{payload.id}")

    def test_missing_opening_brace_fails(self, parser: BindingExpressionParser) -> None:
        """Missing { raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("$payload.id}")

    def test_missing_closing_brace_fails(self, parser: BindingExpressionParser) -> None:
        """Missing } raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${payload.id")

    def test_missing_all_braces_fails(self, parser: BindingExpressionParser) -> None:
        """Missing both braces raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("$payload.id")

    def test_missing_dollar_and_braces_fails(
        self, parser: BindingExpressionParser
    ) -> None:
        """Plain path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("payload.id")

    # -------------------------------------------------------------------------
    # Missing path tests
    # -------------------------------------------------------------------------

    def test_missing_path_fails(self, parser: BindingExpressionParser) -> None:
        """${payload} without path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${payload}")

    def test_source_only_with_dot_fails(self, parser: BindingExpressionParser) -> None:
        """${payload.} without path segment raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${payload.}")

    # -------------------------------------------------------------------------
    # Extra characters tests
    # -------------------------------------------------------------------------

    def test_extra_characters_after_brace_fails(
        self, parser: BindingExpressionParser
    ) -> None:
        """Extra chars after } raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${payload.id}extra")

    def test_leading_whitespace_fails(self, parser: BindingExpressionParser) -> None:
        """Leading whitespace raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse(" ${payload.id}")

    def test_trailing_whitespace_fails(self, parser: BindingExpressionParser) -> None:
        """Trailing whitespace raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${payload.id} ")

    def test_internal_whitespace_fails(self, parser: BindingExpressionParser) -> None:
        """Whitespace in expression raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${payload. id}")

    def test_prefix_text_fails(self, parser: BindingExpressionParser) -> None:
        """Text before expression raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("prefix${payload.id}")

    # -------------------------------------------------------------------------
    # Wrong delimiter tests
    # -------------------------------------------------------------------------

    def test_wrong_opening_delimiter_fails(
        self, parser: BindingExpressionParser
    ) -> None:
        """$(payload.id) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("$(payload.id)")

    def test_double_braces_fail(self, parser: BindingExpressionParser) -> None:
        """${{payload.id}} raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${{payload.id}}")

    def test_angle_brackets_fail(self, parser: BindingExpressionParser) -> None:
        """$<payload.id> raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("$<payload.id>")

    # -------------------------------------------------------------------------
    # Empty and invalid input tests
    # -------------------------------------------------------------------------

    def test_empty_string_fails(self, parser: BindingExpressionParser) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("")

    def test_only_braces_fails(self, parser: BindingExpressionParser) -> None:
        """${} raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("${}")

    def test_only_dollar_fails(self, parser: BindingExpressionParser) -> None:
        """$ alone raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            parser.parse("$")


class TestBindingExpressionParserConstants:
    """Tests for module-level constants."""

    def test_max_expression_length_is_256(self) -> None:
        """MAX_EXPRESSION_LENGTH is 256."""
        assert MAX_EXPRESSION_LENGTH == 256

    def test_max_path_segments_is_20(self) -> None:
        """MAX_PATH_SEGMENTS is 20."""
        assert MAX_PATH_SEGMENTS == 20

    def test_valid_sources_contains_expected_values(self) -> None:
        """VALID_SOURCES contains payload, envelope, context."""
        assert frozenset({"payload", "envelope", "context"}) == VALID_SOURCES

    def test_valid_context_paths_contains_expected_values(self) -> None:
        """VALID_CONTEXT_PATHS contains expected runtime context paths."""
        expected = frozenset({"now_iso", "dispatcher_id", "correlation_id"})
        assert expected == VALID_CONTEXT_PATHS


class TestBindingExpressionParserThreadSafety:
    """Thread safety tests for BindingExpressionParser.

    The parser is documented as stateless and thread-safe.
    These tests verify the parser can be reused across multiple parses.
    """

    def test_parser_reusable_across_parses(self) -> None:
        """Parser can be reused for multiple parse operations."""
        parser = BindingExpressionParser()

        # Parse multiple expressions with same parser instance
        source1, _segments1 = parser.parse("${payload.id}")
        source2, _segments2 = parser.parse("${envelope.correlation_id}")
        source3, _segments3 = parser.parse("${context.now_iso}")

        assert source1 == "payload"
        assert source2 == "envelope"
        assert source3 == "context"

    def test_parser_no_state_leakage(self) -> None:
        """Parsing one expression doesn't affect subsequent parses."""
        parser = BindingExpressionParser()

        # Parse a valid expression
        parser.parse("${payload.user.id}")

        # Parse an invalid expression - should fail independently
        with pytest.raises(ValueError):
            parser.parse("${invalid.source}")

        # Parse another valid expression - should still work
        source, segments = parser.parse("${payload.other.field}")
        assert source == "payload"
        assert segments == ("other", "field")
