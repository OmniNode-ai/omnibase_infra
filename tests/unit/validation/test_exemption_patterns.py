"""
Tests for regex-based exemption pattern matching in infrastructure validators.

Validates that:
- ExemptionPattern TypedDict is properly defined
- Regex-based pattern matching works correctly
- Exemptions are resilient to code changes (no hardcoded line numbers)
- Pattern matching is precise and doesn't over-match
"""

from omnibase_infra.validation.infra_validators import (
    ExemptionPattern,
    _filter_exempted_errors,
)


class TestExemptionPatternType:
    """Test ExemptionPattern TypedDict structure."""

    def test_exemption_pattern_has_required_fields(self) -> None:
        """Verify ExemptionPattern has all expected field definitions."""
        # TypedDict fields are stored in __annotations__
        annotations = ExemptionPattern.__annotations__
        assert "file_pattern" in annotations
        assert "class_pattern" in annotations
        assert "method_pattern" in annotations
        assert "violation_pattern" in annotations

    def test_exemption_pattern_fields_are_strings(self) -> None:
        """Verify all ExemptionPattern fields are typed as strings."""
        annotations = ExemptionPattern.__annotations__
        assert annotations["file_pattern"] is str
        assert annotations["class_pattern"] is str
        assert annotations["method_pattern"] is str
        assert annotations["violation_pattern"] is str

    def test_exemption_pattern_is_total_false(self) -> None:
        """Verify ExemptionPattern allows optional fields (total=False)."""
        # total=False means fields can be omitted
        assert ExemptionPattern.__total__ is False

    def test_exemption_pattern_creation(self) -> None:
        """Verify ExemptionPattern can be created with various field combinations."""
        # Full pattern
        full_pattern: ExemptionPattern = {
            "file_pattern": r"test\.py",
            "class_pattern": r"Class 'Test'",
            "method_pattern": r"Function 'test'",
            "violation_pattern": r"has \d+ methods",
        }
        assert full_pattern["file_pattern"] == r"test\.py"

        # Partial pattern (only required fields for specific use case)
        partial_pattern: ExemptionPattern = {
            "file_pattern": r"test\.py",
            "violation_pattern": r"has \d+ parameters",
        }
        assert "class_pattern" not in partial_pattern
        assert partial_pattern["file_pattern"] == r"test\.py"


class TestFilterExemptedErrorsBasic:
    """Test basic exemption filtering functionality."""

    def test_empty_errors_list(self) -> None:
        """Verify filtering empty errors list returns empty list."""
        patterns: list[ExemptionPattern] = [
            {"file_pattern": r"test\.py", "violation_pattern": r"has \d+ methods"}
        ]
        result = _filter_exempted_errors([], patterns)
        assert result == []

    def test_empty_patterns_list(self) -> None:
        """Verify filtering with no patterns returns all errors."""
        errors = ["error1.py:10: Some error", "error2.py:20: Another error"]
        result = _filter_exempted_errors(errors, [])
        assert result == errors

    def test_no_matching_patterns(self) -> None:
        """Verify errors not matching any pattern are preserved."""
        errors = ["other_file.py:10: Some error"]
        patterns: list[ExemptionPattern] = [
            {"file_pattern": r"event_bus_kafka\.py", "violation_pattern": r"methods"}
        ]
        result = _filter_exempted_errors(errors, patterns)
        assert result == errors


class TestFilterExemptedErrorsRegexMatching:
    """Test regex-based pattern matching for exemptions."""

    def test_simple_file_pattern_match(self) -> None:
        """Verify simple file pattern matching works."""
        errors = ["event_bus_kafka.py:100: Some violation"]
        patterns: list[ExemptionPattern] = [
            {"file_pattern": r"event_bus_kafka\.py", "violation_pattern": r"violation"}
        ]
        result = _filter_exempted_errors(errors, patterns)
        assert result == []  # Error should be filtered

    def test_regex_digit_pattern_matching(self) -> None:
        """Verify regex patterns with \\d+ work correctly."""
        errors = [
            "event_bus_kafka.py:100: Class 'EventBusKafka' has 14 methods",
            "event_bus_kafka.py:200: Class 'EventBusKafka' has 20 methods",
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        assert result == []  # Both errors should be filtered (any count matches)

    def test_method_pattern_matching(self) -> None:
        """Verify method pattern matching works correctly."""
        errors = [
            "event_bus_kafka.py:50: Function '__init__' has 10 parameters",
            "event_bus_kafka.py:100: Function 'connect' has 5 parameters",
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "method_pattern": r"Function '__init__'",
                "violation_pattern": r"has \d+ parameters",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        # Only __init__ should be filtered
        assert len(result) == 1
        assert "Function 'connect'" in result[0]

    def test_all_patterns_must_match(self) -> None:
        """Verify all specified patterns must match for exemption."""
        errors = [
            "event_bus_kafka.py:100: Class 'EventBusKafka' has 14 methods",
            "other_file.py:100: Class 'EventBusKafka' has 14 methods",
            "event_bus_kafka.py:100: Class 'OtherClass' has 14 methods",
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        # Only first error matches all patterns
        assert len(result) == 2
        assert "other_file.py" in result[0]
        assert "Class 'OtherClass'" in result[1]


class TestFilterExemptedErrorsRobustness:
    """Test exemption pattern robustness to code changes."""

    def test_line_number_changes_dont_break_exemption(self) -> None:
        """Verify exemptions work regardless of line number changes."""
        # Same violation at different line numbers
        errors_v1 = ["event_bus_kafka.py:100: Class 'EventBusKafka' has 14 methods"]
        errors_v2 = ["event_bus_kafka.py:200: Class 'EventBusKafka' has 14 methods"]
        errors_v3 = ["event_bus_kafka.py:999: Class 'EventBusKafka' has 14 methods"]

        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            }
        ]

        # All should be filtered regardless of line number
        assert _filter_exempted_errors(errors_v1, patterns) == []
        assert _filter_exempted_errors(errors_v2, patterns) == []
        assert _filter_exempted_errors(errors_v3, patterns) == []

    def test_count_changes_dont_break_exemption(self) -> None:
        """Verify exemptions work with different violation counts."""
        # Same violation with different counts
        errors = [
            "event_bus_kafka.py:100: Class 'EventBusKafka' has 10 methods",
            "event_bus_kafka.py:100: Class 'EventBusKafka' has 14 methods",
            "event_bus_kafka.py:100: Class 'EventBusKafka' has 20 methods",
        ]

        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            }
        ]

        # All should be filtered regardless of count
        result = _filter_exempted_errors(errors, patterns)
        assert result == []


class TestFilterExemptedErrorsPrecision:
    """Test exemption pattern precision and specificity."""

    def test_file_pattern_substring_matching(self) -> None:
        """Verify file patterns use substring matching by default."""
        errors = [
            "event_bus_kafka.py:100: violation",
            "other_file.py:100: violation",
        ]
        # Pattern matches substring
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "violation_pattern": r"violation",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        # Only event_bus_kafka.py should be filtered
        assert len(result) == 1
        assert "other_file.py" in result[0]

    def test_class_pattern_must_match_exactly(self) -> None:
        """Verify class pattern doesn't over-match similar class names."""
        errors = [
            "file.py:10: Class 'EventBusKafka' has issue",
            "file.py:20: Class 'KafkaEventBusTest' has issue",
            "file.py:30: Class 'TestKafkaEventBus' has issue",
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"file\.py",
                "class_pattern": r"Class 'EventBusKafka'",  # Exact class name
                "violation_pattern": r"has issue",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        # All match because regex searches for substring
        # To match exactly, pattern should be: r"Class 'EventBusKafka'(?!\w)"
        assert len(result) == 2

    def test_violation_type_specificity(self) -> None:
        """Verify violation pattern distinguishes between violation types."""
        errors = [
            "file.py:10: Class 'Test' has 10 methods",
            "file.py:20: Function 'test' has 10 parameters",
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"file\.py",
                "violation_pattern": r"has \d+ methods",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        # Only method count should be filtered
        assert len(result) == 1
        assert "parameters" in result[0]


class TestFilterExemptedErrorsMultiplePatterns:
    """Test multiple exemption patterns."""

    def test_multiple_patterns_apply_independently(self) -> None:
        """Verify multiple patterns can exempt different errors."""
        errors = [
            "event_bus_kafka.py:100: Class 'EventBusKafka' has 14 methods",
            "event_bus_kafka.py:200: Function '__init__' has 10 parameters",
            "other_file.py:50: Class 'Other' has 5 methods",
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            },
            {
                "file_pattern": r"event_bus_kafka\.py",
                "method_pattern": r"Function '__init__'",
                "violation_pattern": r"has \d+ parameters",
            },
        ]
        result = _filter_exempted_errors(errors, patterns)
        # First two should be filtered, last one preserved
        assert len(result) == 1
        assert "other_file.py" in result[0]

    def test_overlapping_patterns_dont_cause_issues(self) -> None:
        """Verify overlapping patterns work correctly."""
        errors = ["file.py:10: Class 'Test' has 10 methods"]
        patterns: list[ExemptionPattern] = [
            {"file_pattern": r"file\.py", "violation_pattern": r"has \d+ methods"},
            {
                "file_pattern": r"file\.py",
                "class_pattern": r"Class 'Test'",
                "violation_pattern": r"methods",
            },
        ]
        result = _filter_exempted_errors(errors, patterns)
        # Error matches both patterns, should be filtered once
        assert result == []


class TestRealWorldKafkaEventBusExemptions:
    """Test actual EventBusKafka exemption patterns from infra_validators.py."""

    def test_event_bus_kafka_method_count_exemption(self) -> None:
        """Verify EventBusKafka method count violations are exempted."""
        errors = [
            "src/omnibase_infra/event_bus/event_bus_kafka.py:323: Class 'EventBusKafka' has 14 methods (threshold: 10)"
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        assert result == []

    def test_event_bus_kafka_init_parameter_exemption(self) -> None:
        """Verify EventBusKafka __init__ parameter violations are exempted."""
        errors = [
            "src/omnibase_infra/event_bus/event_bus_kafka.py:50: Function '__init__' has 10 parameters (threshold: 5)"
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "method_pattern": r"Function '__init__'",
                "violation_pattern": r"has \d+ parameters",
            }
        ]
        result = _filter_exempted_errors(errors, patterns)
        assert result == []

    def test_other_kafka_violations_not_exempted(self) -> None:
        """Verify non-exempted EventBusKafka violations are preserved."""
        errors = [
            "src/omnibase_infra/event_bus/event_bus_kafka.py:100: Function 'publish' has too many local variables"
        ]
        patterns: list[ExemptionPattern] = [
            {
                "file_pattern": r"event_bus_kafka\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \d+ methods",
            },
            {
                "file_pattern": r"event_bus_kafka\.py",
                "method_pattern": r"Function '__init__'",
                "violation_pattern": r"has \d+ parameters",
            },
        ]
        result = _filter_exempted_errors(errors, patterns)
        # This violation doesn't match patterns, should be preserved
        assert len(result) == 1
        assert "too many local variables" in result[0]
