# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelLlmInferenceResponse validators.

Tests cover the model's invariants:
    - Text XOR tool_calls mutual exclusivity
    - Backend success requirement
    - finish_reason=ERROR rejection
    - Truncated/finish_reason consistency
    - Tool_calls/finish_reason consistency

Related:
    - OMN-2106: Phase 6 LLM inference response model
"""

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


@pytest.mark.unit
class TestModelLlmInferenceResponseValid:
    """Tests for valid construction scenarios."""

    def test_valid_with_generated_text(self) -> None:
        """Valid response with generated_text and no tool_calls."""
        resp = ModelLlmInferenceResponse(**_base_kwargs())
        assert resp.status == "success"
        assert resp.generated_text == "Hello, world!"
        assert resp.tool_calls == ()

    def test_valid_with_tool_calls(self) -> None:
        """Valid response with tool_calls and no generated_text."""
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
        assert resp.tool_calls[0].id == "call_abc123"

    def test_valid_empty_response(self) -> None:
        """Valid response with neither text nor tool_calls (empty response)."""
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(
                generated_text=None,
                finish_reason=EnumLlmFinishReason.CONTENT_FILTER,
            ),
        )
        assert resp.generated_text is None
        assert resp.tool_calls == ()

    def test_valid_empty_string_text(self) -> None:
        """Valid response with empty string as generated_text."""
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(generated_text=""),
        )
        assert resp.generated_text == ""

    def test_valid_tool_calls_with_finish_reason_tool_calls(self) -> None:
        """finish_reason=TOOL_CALLS with tool_calls present is valid."""
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


@pytest.mark.unit
class TestModelLlmInferenceResponseXorViolation:
    """Tests for the text XOR tool_calls invariant."""

    def test_both_text_and_tool_calls_raises(self) -> None:
        """Having both generated_text and tool_calls raises ValueError."""
        tc = _tool_call()
        with pytest.raises(
            ValueError, match="cannot have both generated_text and tool_calls"
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    generated_text="some text",
                    tool_calls=(tc,),
                    finish_reason=EnumLlmFinishReason.TOOL_CALLS,
                ),
            )

    def test_empty_string_text_with_tool_calls_raises(self) -> None:
        """Empty string counts as text, so it still violates XOR with tool_calls."""
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


@pytest.mark.unit
class TestModelLlmInferenceResponseBackendSuccess:
    """Tests for the backend_result.success requirement."""

    def test_backend_failure_raises(self) -> None:
        """backend_result.success=False raises ValueError."""
        with pytest.raises(ValueError, match=r"backend_result\.success must be True"):
            ModelLlmInferenceResponse(
                **_base_kwargs(backend_result=_failure_backend()),
            )


@pytest.mark.unit
class TestModelLlmInferenceResponseErrorFinishReason:
    """Tests for the finish_reason=ERROR rejection."""

    def test_finish_reason_error_raises(self) -> None:
        """finish_reason=ERROR is not permitted on this model."""
        with pytest.raises(ValueError, match="finish_reason=ERROR is not permitted"):
            ModelLlmInferenceResponse(
                **_base_kwargs(finish_reason=EnumLlmFinishReason.ERROR),
            )


@pytest.mark.unit
class TestModelLlmInferenceResponseTruncatedConsistency:
    """Tests for the truncated/finish_reason consistency."""

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
        """truncated=True with finish_reason=LENGTH is consistent."""
        resp = ModelLlmInferenceResponse(
            **_base_kwargs(truncated=True, finish_reason=EnumLlmFinishReason.LENGTH),
        )
        assert resp.truncated is True
        assert resp.finish_reason == EnumLlmFinishReason.LENGTH


@pytest.mark.unit
class TestModelLlmInferenceResponseToolCallsFinishReasonConsistency:
    """Tests for the tool_calls/finish_reason consistency."""

    def test_finish_reason_tool_calls_with_empty_tool_calls_raises(self) -> None:
        """finish_reason=TOOL_CALLS with no tool_calls present raises ValueError."""
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

    def test_tool_calls_present_with_stop_finish_reason_raises(self) -> None:
        """tool_calls present with finish_reason=STOP raises ValueError."""
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

    def test_tool_calls_present_with_unknown_finish_reason_raises(self) -> None:
        """tool_calls present with finish_reason=UNKNOWN raises ValueError."""
        tc = _tool_call()
        with pytest.raises(
            ValueError,
            match="tool_calls are present but finish_reason is not TOOL_CALLS",
        ):
            ModelLlmInferenceResponse(
                **_base_kwargs(
                    generated_text=None,
                    tool_calls=(tc,),
                    finish_reason=EnumLlmFinishReason.UNKNOWN,
                ),
            )

    def test_tool_calls_present_with_tool_calls_finish_reason_valid(self) -> None:
        """tool_calls present with finish_reason=TOOL_CALLS is valid."""
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


@pytest.mark.unit
class TestModelLlmInferenceResponseTimestamp:
    """Tests for timestamp timezone-awareness validation."""

    def test_naive_datetime_raises(self) -> None:
        """A naive datetime (no tzinfo) must be rejected by the field validator."""
        with pytest.raises(ValueError, match="timezone-aware"):
            ModelLlmInferenceResponse(
                **_base_kwargs(timestamp=datetime(2025, 1, 1)),
            )


@pytest.mark.unit
class TestModelLlmInferenceResponseFieldConstraints:
    """Tests for individual field constraints."""

    def test_empty_model_used_raises(self) -> None:
        """model_used with empty string must be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceResponse(
                **_base_kwargs(model_used=""),
            )


@pytest.mark.unit
class TestModelLlmInferenceResponseSerialization:
    """Tests for JSON serialization round-trip correctness."""

    def test_json_round_trip(self) -> None:
        """Serialize to JSON dict and deserialize back; key fields must match."""
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

        assert len(restored.tool_calls) == len(original.tool_calls)
        assert restored.generated_text == original.generated_text
        assert restored.finish_reason == original.finish_reason
        assert restored.model_used == original.model_used
        assert restored.correlation_id == original.correlation_id
