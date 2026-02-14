# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmInferenceResponse validation.

Tests cover the model's invariants and edge cases:
    - Text XOR tool_calls mutual exclusivity
    - Empty string policy (None vs "" vs content)
    - Content filter with no output scenario
    - Backend success requirement
    - finish_reason=ERROR rejection
    - Truncated/finish_reason consistency
    - Tool_calls/finish_reason consistency
    - Timestamp timezone awareness
    - Field constraints and serialization

Related:
    - OMN-2110: Phase 10 inference model validation tests
    - OMN-2106: Phase 6 LLM inference response model
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)
from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage

pytestmark = [pytest.mark.unit]


# =============================================================================
# Fixtures
# =============================================================================


def _success_backend() -> ModelBackendResult:
    """Create a successful backend result for test convenience."""
    return ModelBackendResult(success=True, duration_ms=100.0)


def _failure_backend() -> ModelBackendResult:
    """Create a failed backend result for test convenience."""
    return ModelBackendResult(success=False, error="test failure", duration_ms=50.0)


def _usage() -> ModelLlmUsage:
    """Create a minimal usage object for test convenience."""
    return ModelLlmUsage(tokens_input=10, tokens_output=5)


def _tool_call() -> ModelLlmToolCall:
    """Create a single tool call for test convenience."""
    return ModelLlmToolCall(
        id="call_abc123",
        function=ModelLlmFunctionCall(name="search", arguments='{"q": "hello"}'),
    )


def _base_kwargs(**overrides: object) -> dict[str, object]:
    """Return keyword arguments for a valid text-based response, with overrides."""
    defaults: dict[str, object] = {
        "generated_text": "Hello, world!",
        "model_used": "test-model",
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "finish_reason": EnumLlmFinishReason.STOP,
        "usage": _usage(),
        "latency_ms": 150.0,
        "backend_result": _success_backend(),
        "correlation_id": uuid4(),
        "execution_id": uuid4(),
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# Empty String Policy Tests
# =============================================================================


class TestEmptyStringPolicy:
    """Tests for the empty string policy on generated_text.

    The model distinguishes three states:
        - generated_text=None: no text output (content filter, tool-call-only)
        - generated_text="": valid empty completion (STOP with no content)
        - generated_text="hi": normal text output

    Both None and "" with empty tool_calls are valid (edge cases).
    """

    def test_generated_text_none_tool_calls_empty_allowed(self) -> None:
        """generated_text=None, tool_calls=() is valid.

        Represents a content filter with no output or a model returning
        nothing. This is a legitimate empty response state.
        """
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                finish_reason=EnumLlmFinishReason.CONTENT_FILTER,
            ),
        )
        assert resp.generated_text is None
        assert resp.tool_calls == ()

    def test_generated_text_empty_string_tool_calls_empty_allowed(self) -> None:
        """generated_text="", tool_calls=() is valid.

        Represents an empty completion with STOP finish reason.
        """
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(generated_text=""),
        )
        assert resp.generated_text == ""
        assert resp.tool_calls == ()

    def test_generated_text_with_content_valid(self) -> None:
        """generated_text="hi" with no tool_calls is standard valid response."""
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(generated_text="hi"),
        )
        assert resp.generated_text == "hi"

    def test_empty_string_counts_as_text_for_xor(self) -> None:
        """Empty string "" counts as text, so it violates XOR with tool_calls."""
        tc = _tool_call()
        with pytest.raises(
            ValueError, match="cannot have both generated_text and tool_calls"
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    generated_text="",
                    tool_calls=(tc,),
                    finish_reason=EnumLlmFinishReason.TOOL_CALLS,
                ),
            )


# =============================================================================
# Text XOR Tool Calls Tests
# =============================================================================


class TestTextXorToolCalls:
    """Tests for the text XOR tool_calls invariant."""

    def test_text_and_tool_calls_rejected(self) -> None:
        """generated_text="hi" with tool_calls=[...] is rejected."""
        tc = _tool_call()
        with pytest.raises(
            ValueError, match="cannot have both generated_text and tool_calls"
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    generated_text="hi",
                    tool_calls=(tc,),
                    finish_reason=EnumLlmFinishReason.TOOL_CALLS,
                ),
            )

    def test_text_only_valid(self) -> None:
        """generated_text with no tool_calls is valid."""
        resp = ModelLlmInferenceResponse(**_base_kwargs())
        assert resp.generated_text == "Hello, world!"
        assert resp.tool_calls == ()

    def test_tool_calls_only_valid(self) -> None:
        """tool_calls with no generated_text is valid."""
        tc = _tool_call()
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                tool_calls=(tc,),
                finish_reason=EnumLlmFinishReason.TOOL_CALLS,
            ),
        )
        assert resp.generated_text is None
        assert len(resp.tool_calls) == 1

    def test_neither_text_nor_tool_calls_valid(self) -> None:
        """Neither text nor tool_calls is valid (empty response)."""
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                finish_reason=EnumLlmFinishReason.CONTENT_FILTER,
            ),
        )
        assert resp.generated_text is None
        assert resp.tool_calls == ()


# =============================================================================
# Backend Success Tests
# =============================================================================


class TestBackendSuccess:
    """Tests for the backend_result.success requirement."""

    def test_backend_failure_raises(self) -> None:
        """backend_result.success=False raises ValueError."""
        with pytest.raises(ValueError, match=r"backend_result\.success must be True"):
            ModelLlmInferenceResponse(
                **_base_kwargs(backend_result=_failure_backend()),
            )

    def test_backend_success_valid(self) -> None:
        """backend_result.success=True is accepted."""
        resp = ModelLlmInferenceResponse(**_base_kwargs())
        assert resp.backend_result.success is True


# =============================================================================
# finish_reason=ERROR Rejection Tests
# =============================================================================


class TestErrorFinishReasonRejection:
    """Tests for the finish_reason=ERROR rejection."""

    def test_finish_reason_error_raises(self) -> None:
        """finish_reason=ERROR is not permitted on successful responses."""
        with pytest.raises(ValueError, match="finish_reason=ERROR is not permitted"):
            ModelLlmInferenceResponse(
                **_base_kwargs(finish_reason=EnumLlmFinishReason.ERROR),
            )


# =============================================================================
# Truncated / finish_reason Consistency Tests
# =============================================================================


class TestTruncatedConsistency:
    """Tests for truncated/finish_reason consistency."""

    def test_truncated_true_with_stop_raises(self) -> None:
        """truncated=True with finish_reason=STOP is contradictory."""
        with pytest.raises(
            ValueError, match="truncated=True is contradictory with finish_reason=STOP"
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(truncated=True, finish_reason=EnumLlmFinishReason.STOP),
            )

    def test_truncated_false_with_length_raises(self) -> None:
        """truncated=False with finish_reason=LENGTH is contradictory."""
        with pytest.raises(
            ValueError,
            match="truncated=False is contradictory with finish_reason=LENGTH",
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    truncated=False, finish_reason=EnumLlmFinishReason.LENGTH
                ),
            )

    def test_truncated_true_with_length_valid(self) -> None:
        """truncated=True with finish_reason=LENGTH is consistent and valid."""
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(truncated=True, finish_reason=EnumLlmFinishReason.LENGTH),
        )
        assert resp.truncated is True
        assert resp.finish_reason == EnumLlmFinishReason.LENGTH


# =============================================================================
# tool_calls / finish_reason Consistency Tests
# =============================================================================


class TestToolCallsFinishReasonConsistency:
    """Tests for tool_calls/finish_reason consistency."""

    def test_finish_reason_tool_calls_with_no_tools_raises(self) -> None:
        """finish_reason=TOOL_CALLS with no tool_calls raises ValueError."""
        with pytest.raises(
            ValueError,
            match="finish_reason is TOOL_CALLS but no tool_calls are present",
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    generated_text=None,
                    finish_reason=EnumLlmFinishReason.TOOL_CALLS,
                ),
            )

    def test_tool_calls_with_non_tool_calls_finish_reason_raises(self) -> None:
        """tool_calls present with finish_reason != TOOL_CALLS raises ValueError."""
        tc = _tool_call()
        with pytest.raises(
            ValueError,
            match="tool_calls are present but finish_reason is not TOOL_CALLS",
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    generated_text=None,
                    tool_calls=(tc,),
                    finish_reason=EnumLlmFinishReason.STOP,
                ),
            )

    def test_tool_calls_with_correct_finish_reason_valid(self) -> None:
        """tool_calls with finish_reason=TOOL_CALLS is valid."""
        tc = _tool_call()
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                tool_calls=(tc,),
                finish_reason=EnumLlmFinishReason.TOOL_CALLS,
            ),
        )
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS
        assert len(resp.tool_calls) == 1


# =============================================================================
# Timestamp Validation Tests
# =============================================================================


class TestTimestampValidation:
    """Tests for timestamp timezone-awareness validation."""

    def test_naive_datetime_raises(self) -> None:
        """A naive datetime (no tzinfo) must be rejected."""
        with pytest.raises(ValueError, match="timezone-aware"):
            ModelLlmInferenceResponse(
                **_base_kwargs(timestamp=datetime(2025, 1, 1)),  # noqa: DTZ001
            )

    def test_utc_datetime_valid(self) -> None:
        """UTC-aware datetime is accepted."""
        ts = datetime.now(UTC)
        resp = ModelLlmInferenceResponse(**_base_kwargs(timestamp=ts))
        assert resp.timestamp.tzinfo is not None


# =============================================================================
# Field Constraint Tests
# =============================================================================


class TestFieldConstraints:
    """Tests for individual field constraints."""

    def test_empty_model_used_raises(self) -> None:
        """model_used with empty string must be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceResponse(**_base_kwargs(model_used=""))

    def test_negative_latency_raises(self) -> None:
        """Negative latency_ms is rejected (ge=0.0)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceResponse(**_base_kwargs(latency_ms=-1.0))

    def test_negative_retry_count_raises(self) -> None:
        """Negative retry_count is rejected (ge=0)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceResponse(**_base_kwargs(retry_count=-1))

    def test_zero_latency_valid(self) -> None:
        """latency_ms=0.0 is a valid boundary value."""
        resp = ModelLlmInferenceResponse(**_base_kwargs(latency_ms=0.0))
        assert resp.latency_ms == 0.0

    def test_zero_retry_count_valid(self) -> None:
        """retry_count=0 is the default and valid."""
        resp = ModelLlmInferenceResponse(**_base_kwargs(retry_count=0))
        assert resp.retry_count == 0

    def test_status_always_success(self) -> None:
        """Status field is always 'success' (Literal type)."""
        resp = ModelLlmInferenceResponse(**_base_kwargs())
        assert resp.status == "success"

    def test_status_non_success_rejected(self) -> None:
        """Non-'success' status value is rejected by Literal type."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceResponse(**_base_kwargs(status="error"))


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for JSON serialization round-trip correctness."""

    def test_json_round_trip_text_response(self) -> None:
        """Text response serializes and deserializes correctly."""
        original = ModelLlmInferenceResponse(**_base_kwargs())
        data = original.model_dump(mode="json")
        restored = ModelLlmInferenceResponse.model_validate(data)
        assert restored.generated_text == original.generated_text
        assert restored.finish_reason == original.finish_reason
        assert restored.model_used == original.model_used
        assert restored.correlation_id == original.correlation_id

    def test_json_round_trip_tool_calls_response(self) -> None:
        """Tool call response serializes and deserializes correctly."""
        tc = _tool_call()
        original = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                tool_calls=(tc,),
                finish_reason=EnumLlmFinishReason.TOOL_CALLS,
            ),
        )
        data = original.model_dump(mode="json")
        restored = ModelLlmInferenceResponse.model_validate(data)
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "call_abc123"
        assert restored.generated_text is None

    def test_json_round_trip_empty_response(self) -> None:
        """Empty response (no text, no tools) serializes correctly."""
        original = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                finish_reason=EnumLlmFinishReason.CONTENT_FILTER,
            ),
        )
        data = original.model_dump(mode="json")
        restored = ModelLlmInferenceResponse.model_validate(data)
        assert restored.generated_text is None
        assert restored.tool_calls == ()


# =============================================================================
# Immutability Tests
# =============================================================================


class TestImmutability:
    """Tests for frozen=True immutability enforcement."""

    def test_frozen_generated_text(self) -> None:
        """Cannot reassign generated_text after construction."""
        resp = ModelLlmInferenceResponse(**_base_kwargs())
        with pytest.raises(ValidationError):
            resp.generated_text = "modified"  # type: ignore[misc]

    def test_frozen_model_used(self) -> None:
        """Cannot reassign model_used after construction."""
        resp = ModelLlmInferenceResponse(**_base_kwargs())
        with pytest.raises(ValidationError):
            resp.model_used = "other-model"  # type: ignore[misc]


# =============================================================================
# extra="forbid" Tests
# =============================================================================


class TestExtraFieldsRejected:
    """Tests for extra='forbid' enforcement."""

    def test_unknown_field_rejected(self) -> None:
        """Unknown field is rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceResponse(**_base_kwargs(unknown_field="surprise"))


# =============================================================================
# Usage Model Tests
# =============================================================================


class TestUsageModel:
    """Tests for ModelLlmUsage validation within the response."""

    def test_usage_auto_computes_total(self) -> None:
        """tokens_total is auto-computed when omitted."""
        usage = ModelLlmUsage(tokens_input=120, tokens_output=45)
        assert usage.tokens_total == 165

    def test_usage_explicit_total_validated(self) -> None:
        """Explicit tokens_total must match sum."""
        with pytest.raises(ValidationError, match="does not equal"):
            ModelLlmUsage(tokens_input=10, tokens_output=5, tokens_total=20)

    def test_usage_zero_values_valid(self) -> None:
        """All-zero token counts are valid."""
        usage = ModelLlmUsage(tokens_input=0, tokens_output=0)
        assert usage.tokens_total == 0
