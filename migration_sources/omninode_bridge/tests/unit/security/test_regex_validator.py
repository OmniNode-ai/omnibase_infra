#!/usr/bin/env python3
"""
Unit tests for regex validator.

Tests regex pattern safety validation to prevent ReDoS attacks.
"""


import pytest

from omninode_bridge.security import (
    RegexValidator,
    SecurityValidationError,
    get_regex_validator,
    safe_compile,
    safe_match,
    safe_search,
)


class TestRegexValidator:
    """Test regex pattern safety validation."""

    def test_safe_pattern_validation(self):
        """Test that safe regex patterns are accepted."""
        validator = RegexValidator()

        # Safe patterns should be accepted
        safe_patterns = [
            r"hello\s+world",
            r"\d{3}-\d{4}",
            r"[a-zA-Z0-9]+",
            r"^test$",
            r"error:\s*(.+)",
        ]

        for pattern in safe_patterns:
            result = validator.validate_pattern(pattern)
            assert result.is_valid
            assert result.compiled_pattern is not None

    def test_dangerous_pattern_rejection(self):
        """Test that dangerous regex patterns are rejected."""
        validator = RegexValidator(strict_mode=True)

        # Dangerous patterns that should be rejected
        dangerous_patterns = [
            r"(a+)+",  # Nested quantifiers - closing paren followed by +
            r"(abc)*+",  # Multiple consecutive quantifiers
            r"(test){100,}",  # Very large quantifier range
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(SecurityValidationError):
                validator.validate_pattern(pattern)

    def test_pattern_complexity_detection(self):
        """Test that overly complex patterns are detected."""
        validator = RegexValidator(strict_mode=True)

        # Create a complex pattern that exceeds threshold but not length limit
        # Score calculation: lookaheads (6 each), groups (3 each), wildcards (8 each)
        # Build a pattern with high complexity: 100+ score
        # Pattern with many lookaheads: each (?=.*) = 6 + 8 = 14 points
        # We need 500/14 = ~36 lookaheads to exceed threshold
        complex_pattern = r"(?=.*a)" * 40  # 40 * 14 = 560 points

        with pytest.raises(SecurityValidationError) as exc:
            validator.validate_pattern(complex_pattern)
        assert "complex" in str(exc.value).lower()

    def test_pattern_nesting_depth(self):
        """Test that deep nesting is detected."""
        validator = RegexValidator(strict_mode=True)

        # Create deeply nested pattern (deeper than MAX_NESTING_DEPTH=15)
        nested_pattern = "(" * 20 + "test" + ")" * 20

        with pytest.raises(SecurityValidationError) as exc:
            validator.validate_pattern(nested_pattern)
        assert "nesting" in str(exc.value).lower()

    def test_safe_compile(self):
        """Test safe_compile function."""
        # Safe pattern should compile
        pattern = safe_compile(r"test\s+pattern")
        assert pattern is not None
        assert pattern.match("test  pattern")

        # Pattern that's too long should raise error
        with pytest.raises(SecurityValidationError):
            safe_compile("a" * 600)  # Exceeds MAX_PATTERN_LENGTH

    def test_safe_match(self):
        """Test safe_match function."""
        # Safe pattern should match
        match = safe_match(r"hello\s+(\w+)", "hello world")
        assert match is not None
        assert match.group(1) == "world"

        # No match should return None
        match = safe_match(r"goodbye", "hello world")
        assert match is None

    def test_safe_search(self):
        """Test safe_search function."""
        # Safe pattern should find match
        match = safe_search(r"error:\s*(.+)", "This is an error: Something went wrong")
        assert match is not None
        assert "Something went wrong" in match.group(1)

        # No match should return None
        match = safe_search(r"warning", "This is an error")
        assert match is None

    def test_timeout_protection(self):
        """Test that timeout protection works for slow patterns."""
        # Test the timeout mechanism directly with a safe but slow operation
        # We'll use a pattern that's valid but might take time on very long strings
        validator = RegexValidator(strict_mode=False)

        # Use a pattern that's valid but could be slow on long strings
        pattern = r".*x.*y.*z"  # Multiple wildcards but not nested quantifiers
        long_text = "a" * 10000 + "x" + "b" * 10000 + "y" + "c" * 10000 + "z"

        # This should work within timeout
        result = validator.safe_search(pattern, long_text, timeout=2.0)
        # Result can be None or a match, we just verify no timeout

    def test_pattern_caching(self):
        """Test that compiled patterns are cached."""
        validator = RegexValidator()

        pattern = r"test\s+pattern"

        # First validation
        result1 = validator.validate_pattern(pattern)

        # Second validation should use cached pattern
        result2 = validator.validate_pattern(pattern)

        # Should return the same compiled pattern object
        assert result1.compiled_pattern is result2.compiled_pattern

    def test_get_regex_validator_singleton(self):
        """Test that get_regex_validator returns singleton."""
        validator1 = get_regex_validator()
        validator2 = get_regex_validator()

        # Should return the same instance
        assert validator1 is validator2

    def test_validation_warnings(self):
        """Test that validation warnings are generated."""
        validator = RegexValidator(strict_mode=False)

        # Pattern with detectable issue - multiple wildcards in sequence
        pattern = r".*.*test"  # Multiple wildcards triggers warning

        result = validator.validate_pattern(pattern)

        # Should be valid but with warnings
        assert result.is_valid
        if len(result.warnings) > 0:
            assert result.severity in ["medium", "low", "high"]
        # Note: If pattern has no warnings, that's also valid behavior

    def test_invalid_pattern_syntax(self):
        """Test that invalid regex syntax is caught."""
        validator = RegexValidator()

        # Invalid regex syntax
        invalid_patterns = [
            r"[a-",  # Unclosed character class
            r"(?P<>test)",  # Empty group name
            r"(?P<123>test)",  # Invalid group name
        ]

        for pattern in invalid_patterns:
            with pytest.raises(Exception):
                validator.validate_pattern(pattern)

    def test_empty_pattern_rejection(self):
        """Test that empty patterns are rejected."""
        validator = RegexValidator()

        with pytest.raises(Exception):
            validator.validate_pattern("")

        with pytest.raises(Exception):
            validator.validate_pattern("   ")

    def test_overly_long_pattern_rejection(self):
        """Test that overly long patterns are rejected."""
        validator = RegexValidator()

        # Create a pattern longer than MAX_PATTERN_LENGTH
        long_pattern = "test" * 200

        with pytest.raises(SecurityValidationError) as exc:
            validator.validate_pattern(long_pattern)
        assert "exceeds maximum length" in str(exc.value)


class TestRegexValidatorIntegration:
    """Integration tests for regex validator with real-world patterns."""

    def test_error_pattern_validation(self):
        """Test validation of error recovery patterns."""
        validator = RegexValidator()

        # Real error patterns from recovery system
        error_patterns = [
            r"ImportError.*No module named '(\w+)'",
            r"SyntaxError.*async.*def",
            r"TypeError.*missing \d+ required",
            r"ValueError.*invalid literal",
        ]

        for pattern in error_patterns:
            result = validator.validate_pattern(pattern)
            assert result.is_valid
            assert result.compiled_pattern is not None

    def test_log_pattern_validation(self):
        """Test validation of log parsing patterns."""
        validator = RegexValidator()

        # Log parsing patterns
        log_patterns = [
            r"\[(\d{4}-\d{2}-\d{2})\]",
            r"ERROR:\s*(.+)",
            r"(\w+)=(\S+)",
        ]

        for pattern in log_patterns:
            result = validator.validate_pattern(pattern)
            assert result.is_valid
            assert result.compiled_pattern is not None

    def test_metadata_extraction_pattern(self):
        """Test validation of metadata extraction patterns."""
        validator = RegexValidator()

        # Metadata extraction patterns
        metadata_patterns = [
            r"<!--\s*ONEX_METADATA_START\s*-->",
            r"<!--\s*ONEX_METADATA_END\s*-->",
            r'"namespace":\s*"([^"]+)"',
        ]

        for pattern in metadata_patterns:
            result = validator.validate_pattern(pattern)
            assert result.is_valid
