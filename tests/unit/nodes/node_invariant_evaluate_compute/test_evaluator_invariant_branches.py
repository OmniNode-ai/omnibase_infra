# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Branch-coverage tests for InvariantEvaluator (OMN-13744).

Covers:
- _is_import_path_allowed: all allow-list branches
- _safe_regex_search: match / no-match / invalid pattern / input too long
- _is_regex_safe: catastrophic-backtracking pattern rejection
- evaluate(): LATENCY, THRESHOLD, FIELD_VALUE (regex), SCHEMA invariant types
"""

from __future__ import annotations

import pytest

from omnibase_core.enums import EnumInvariantType, EnumSeverity
from omnibase_core.models.invariant import ModelInvariant
from omnibase_infra.nodes.node_invariant_evaluate_compute.evaluator_invariant import (
    InvariantEvaluator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invariant(
    inv_type: EnumInvariantType,
    config: dict[str, object],
    name: str = "test_invariant",
) -> ModelInvariant:
    return ModelInvariant(
        name=name,
        type=inv_type,
        severity=EnumSeverity.WARNING,
        config=config,
    )


# ---------------------------------------------------------------------------
# _is_import_path_allowed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsImportPathAllowed:
    def test_no_allowlist_returns_true(self) -> None:
        evaluator = InvariantEvaluator(allowed_import_paths=None)
        assert evaluator._is_import_path_allowed("any.module.function") is True

    def test_exact_match_returns_true(self) -> None:
        evaluator = InvariantEvaluator(allowed_import_paths=["myapp.validators"])
        assert evaluator._is_import_path_allowed("myapp.validators") is True

    def test_prefix_match_with_dot_boundary_returns_true(self) -> None:
        evaluator = InvariantEvaluator(allowed_import_paths=["myapp.validators"])
        assert (
            evaluator._is_import_path_allowed("myapp.validators.check_output") is True
        )

    def test_prefix_match_with_colon_boundary_returns_true(self) -> None:
        evaluator = InvariantEvaluator(allowed_import_paths=["myapp.validators"])
        assert (
            evaluator._is_import_path_allowed("myapp.validators:check_output") is True
        )

    def test_prefix_without_word_boundary_returns_false(self) -> None:
        # "myapp.validators_evil" shares the prefix but has no boundary separator
        evaluator = InvariantEvaluator(allowed_import_paths=["myapp.validators"])
        assert evaluator._is_import_path_allowed("myapp.validators_evil.fn") is False

    def test_malformed_callable_path_returns_false(self) -> None:
        # Path starting with a digit is not a valid Python identifier
        evaluator = InvariantEvaluator(allowed_import_paths=["myapp"])
        assert evaluator._is_import_path_allowed("1invalid.path") is False

    def test_empty_prefix_entry_is_skipped(self) -> None:
        # An empty string in the allow-list must not cause a match
        evaluator = InvariantEvaluator(allowed_import_paths=["", "myapp.validators"])
        # The empty prefix should be skipped; myapp.foo should still NOT match
        # because "myapp.foo" doesn't match "myapp.validators" exactly/prefix
        assert evaluator._is_import_path_allowed("myapp.foo.bar") is False
        # But the valid prefix still works
        assert evaluator._is_import_path_allowed("myapp.validators.fn") is True

    def test_path_not_in_list_returns_false(self) -> None:
        evaluator = InvariantEvaluator(allowed_import_paths=["myapp.validators"])
        assert evaluator._is_import_path_allowed("os.system") is False


# ---------------------------------------------------------------------------
# _safe_regex_search  (called _apply_regex_safe in the task spec)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSafeRegexSearch:
    def test_valid_pattern_with_match_returns_true_match_no_error(self) -> None:
        evaluator = InvariantEvaluator()
        success, match, error = evaluator._safe_regex_search(r"\d+", "hello 42 world")
        assert success is True
        assert match is not None
        assert error == ""

    def test_valid_pattern_no_match_returns_true_none_no_error(self) -> None:
        evaluator = InvariantEvaluator()
        success, match, error = evaluator._safe_regex_search(r"\d+", "no digits here")
        assert success is True
        assert match is None
        assert error == ""

    def test_invalid_regex_returns_false_none_error_str(self) -> None:
        evaluator = InvariantEvaluator()
        success, match, error = evaluator._safe_regex_search(r"[unclosed", "text")
        assert success is False
        assert match is None
        assert len(error) > 0

    def test_input_exceeding_max_length_returns_false_none_error(self) -> None:
        evaluator = InvariantEvaluator()
        long_text = "a" * (InvariantEvaluator.MAX_REGEX_INPUT_LENGTH + 1)
        success, match, error = evaluator._safe_regex_search(r"a+", long_text)
        assert success is False
        assert match is None
        assert "too long" in error.lower() or "max" in error.lower()


# ---------------------------------------------------------------------------
# _is_regex_safe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsRegexSafe:
    def test_simple_pattern_is_safe(self) -> None:
        evaluator = InvariantEvaluator()
        is_safe, error = evaluator._is_regex_safe(r"\d+")
        assert is_safe is True
        assert error == ""

    def test_nested_quantifier_pattern_is_rejected(self) -> None:
        # (a+)+ triggers catastrophic backtracking
        evaluator = InvariantEvaluator()
        is_safe, error = evaluator._is_regex_safe(r"(a+)+")
        assert is_safe is False
        assert len(error) > 0

    def test_multiple_dotstar_pattern_is_rejected(self) -> None:
        # .*.*  — multiple .* sequences
        evaluator = InvariantEvaluator()
        is_safe, error = evaluator._is_regex_safe(r".*.*")
        assert is_safe is False
        assert len(error) > 0

    def test_alternation_with_quantifier_is_rejected(self) -> None:
        # (a|b)+ has alternation with quantifier
        evaluator = InvariantEvaluator()
        is_safe, error = evaluator._is_regex_safe(r"(a|b)+")
        assert is_safe is False
        assert len(error) > 0

    def test_pattern_exceeding_max_length_is_rejected(self) -> None:
        evaluator = InvariantEvaluator()
        long_pattern = "a" * (InvariantEvaluator.MAX_REGEX_PATTERN_LENGTH + 1)
        is_safe, error = evaluator._is_regex_safe(long_pattern)
        assert is_safe is False
        assert "too long" in error.lower() or "max" in error.lower()


# ---------------------------------------------------------------------------
# InvariantEvaluator.evaluate — LATENCY
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateLatency:
    def test_latency_passes_when_value_at_or_below_max_ms(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.LATENCY,
            {"max_ms": 500},
            name="latency_ok",
        )
        result = evaluator.evaluate(inv, {"latency_ms": 250})
        assert result.passed is True

    def test_latency_passes_at_exact_boundary(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.LATENCY,
            {"max_ms": 500},
            name="latency_boundary",
        )
        result = evaluator.evaluate(inv, {"latency_ms": 500})
        assert result.passed is True

    def test_latency_fails_when_value_exceeds_max_ms(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.LATENCY,
            {"max_ms": 500},
            name="latency_fail",
        )
        result = evaluator.evaluate(inv, {"latency_ms": 750})
        assert result.passed is False
        assert "750" in result.message or "exceeds" in result.message.lower()

    def test_latency_uses_duration_ms_fallback_field(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.LATENCY,
            {"max_ms": 1000},
            name="latency_duration",
        )
        result = evaluator.evaluate(inv, {"duration_ms": 200})
        assert result.passed is True


# ---------------------------------------------------------------------------
# InvariantEvaluator.evaluate — THRESHOLD
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateThreshold:
    def test_threshold_passes_when_metric_meets_min_value(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.THRESHOLD,
            {"metric_name": "score", "min_value": 0.5},
            name="threshold_ok",
        )
        result = evaluator.evaluate(inv, {"score": 0.9})
        assert result.passed is True

    def test_threshold_passes_at_exact_min_boundary(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.THRESHOLD,
            {"metric_name": "score", "min_value": 0.5},
            name="threshold_exact",
        )
        result = evaluator.evaluate(inv, {"score": 0.5})
        assert result.passed is True

    def test_threshold_fails_when_metric_below_min_value(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.THRESHOLD,
            {"metric_name": "score", "min_value": 0.5},
            name="threshold_fail",
        )
        result = evaluator.evaluate(inv, {"score": 0.1})
        assert result.passed is False
        assert "below" in result.message.lower() or "minimum" in result.message.lower()

    def test_threshold_fails_when_metric_not_found(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.THRESHOLD,
            {"metric_name": "score", "min_value": 0.5},
            name="threshold_missing",
        )
        result = evaluator.evaluate(inv, {})
        assert result.passed is False
        assert "not found" in result.message.lower()


# ---------------------------------------------------------------------------
# InvariantEvaluator.evaluate — FIELD_VALUE with regex (FORMAT-like)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateFieldValueRegex:
    def test_field_value_passes_on_regex_match(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.FIELD_VALUE,
            {"field_path": "email", "pattern": r"^[\w.+-]+@[\w-]+\.\w+$"},
            name="format_email",
        )
        result = evaluator.evaluate(inv, {"email": "user@example.com"})
        assert result.passed is True

    def test_field_value_fails_on_no_regex_match(self) -> None:
        evaluator = InvariantEvaluator()
        inv = _make_invariant(
            EnumInvariantType.FIELD_VALUE,
            {"field_path": "email", "pattern": r"^[\w.+-]+@[\w-]+\.\w+$"},
            name="format_email_fail",
        )
        result = evaluator.evaluate(inv, {"email": "not-an-email"})
        assert result.passed is False
        assert (
            "pattern" in result.message.lower()
            or "does not match" in result.message.lower()
        )


# ---------------------------------------------------------------------------
# InvariantEvaluator.evaluate — SCHEMA
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateSchema:
    def test_schema_passes_valid_json_against_schema(self) -> None:
        evaluator = InvariantEvaluator()
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["status", "count"],
        }
        inv = _make_invariant(
            EnumInvariantType.SCHEMA,
            {"json_schema": schema},
            name="schema_pass",
        )
        result = evaluator.evaluate(inv, {"status": "ok", "count": 3})
        assert result.passed is True

    def test_schema_fails_when_required_field_missing(self) -> None:
        evaluator = InvariantEvaluator()
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }
        inv = _make_invariant(
            EnumInvariantType.SCHEMA,
            {"json_schema": schema},
            name="schema_missing_field",
        )
        result = evaluator.evaluate(inv, {})
        assert result.passed is False
        assert (
            "failed" in result.message.lower() or "validation" in result.message.lower()
        )

    def test_schema_fails_when_field_has_wrong_type(self) -> None:
        evaluator = InvariantEvaluator()
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        inv = _make_invariant(
            EnumInvariantType.SCHEMA,
            {"json_schema": schema},
            name="schema_wrong_type",
        )
        result = evaluator.evaluate(inv, {"count": "not-an-int"})
        assert result.passed is False

    def test_schema_uses_cached_validator_on_second_call(self) -> None:
        evaluator = InvariantEvaluator()
        schema = {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        }
        inv = _make_invariant(
            EnumInvariantType.SCHEMA,
            {"json_schema": schema},
            name="schema_cache",
        )
        # Call twice — second call should hit the LRU cache
        result1 = evaluator.evaluate(inv, {"value": 1.0})
        assert result1.passed is True
        cache_size_after_first = evaluator.get_validator_cache_size()
        result2 = evaluator.evaluate(inv, {"value": 2.0})
        assert result2.passed is True
        # Cache should not have grown — same schema reuses cached validator
        assert evaluator.get_validator_cache_size() == cache_size_after_first
