# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmInferenceRequest validation.

Tests cover the model's invariants and edge cases:
    - Operation type consistency (CHAT_COMPLETION, COMPLETION, EMBEDDING)
    - Stream guard (Literal[False] rejects stream=True)
    - messages vs prompt mutual exclusivity per operation type
    - ToolChoice validator consistency
    - max_tokens and temperature boundary conditions
    - base_url validation
    - Frozen immutability
    - extra="forbid" enforcement

Related:
    - OMN-2110: Phase 10 inference model validation tests
    - OMN-2105: Phase 5 LLM inference request model
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.nodes.effects.models.model_llm_function_def import (
    ModelLlmFunctionDef,
)
from omnibase_infra.nodes.effects.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.nodes.effects.models.model_llm_message import ModelLlmMessage
from omnibase_infra.nodes.effects.models.model_llm_tool_choice import (
    ModelLlmToolChoice,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_definition import (
    ModelLlmToolDefinition,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Fixtures
# =============================================================================


def _user_message(content: str = "Hello") -> ModelLlmMessage:
    """Create a user message for test convenience."""
    return ModelLlmMessage(role="user", content=content)


def _tool_def() -> ModelLlmToolDefinition:
    """Create a tool definition for test convenience."""
    return ModelLlmToolDefinition(
        function=ModelLlmFunctionDef(
            name="search",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
    )


def _chat_kwargs(**overrides: object) -> dict[str, object]:
    """Return minimal valid kwargs for a CHAT_COMPLETION request."""
    defaults: dict[str, object] = {
        "base_url": "http://192.168.86.201:8000",
        "model": "qwen2.5-coder-14b",
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "messages": (_user_message(),),
    }
    defaults.update(overrides)
    return defaults


def _completion_kwargs(**overrides: object) -> dict[str, object]:
    """Return minimal valid kwargs for a COMPLETION request."""
    defaults: dict[str, object] = {
        "base_url": "http://192.168.86.201:8000",
        "model": "qwen2.5-coder-14b",
        "operation_type": EnumLlmOperationType.COMPLETION,
        "prompt": "Write a hello world program",
    }
    defaults.update(overrides)
    return defaults


def _embedding_kwargs(**overrides: object) -> dict[str, object]:
    """Return minimal valid kwargs for an EMBEDDING request."""
    defaults: dict[str, object] = {
        "base_url": "http://192.168.86.201:8002",
        "model": "gte-qwen2-1.5b",
        "operation_type": EnumLlmOperationType.EMBEDDING,
        "prompt": "embed this text",
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestModelLlmInferenceRequestConstruction:
    """Tests for valid construction of ModelLlmInferenceRequest."""

    def test_minimal_chat_completion(self) -> None:
        """Minimal CHAT_COMPLETION kwargs produce a valid request with defaults."""
        req = ModelLlmInferenceRequest(**_chat_kwargs())
        assert req.base_url == "http://192.168.86.201:8000"
        assert req.model == "qwen2.5-coder-14b"
        assert req.operation_type == EnumLlmOperationType.CHAT_COMPLETION
        assert len(req.messages) == 1
        assert req.prompt is None
        assert req.stream is False
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.top_p is None
        assert req.stop == ()
        assert req.tools == ()
        assert req.tool_choice is None
        assert isinstance(req.correlation_id, UUID)
        assert isinstance(req.execution_id, UUID)

    def test_minimal_completion(self) -> None:
        """Minimal COMPLETION kwargs produce a valid request."""
        req = ModelLlmInferenceRequest(**_completion_kwargs())
        assert req.operation_type == EnumLlmOperationType.COMPLETION
        assert req.prompt == "Write a hello world program"
        assert req.messages == ()

    def test_minimal_embedding(self) -> None:
        """Minimal EMBEDDING kwargs produce a valid request."""
        req = ModelLlmInferenceRequest(**_embedding_kwargs())
        assert req.operation_type == EnumLlmOperationType.EMBEDDING
        assert req.prompt == "embed this text"

    def test_chat_with_system_prompt(self) -> None:
        """CHAT_COMPLETION with a system_prompt is valid."""
        req = ModelLlmInferenceRequest(
            **_chat_kwargs(system_prompt="You are a helpful assistant."),
        )
        assert req.system_prompt == "You are a helpful assistant."

    def test_chat_with_tools(self) -> None:
        """CHAT_COMPLETION with tools and tool_choice is valid."""
        tool = _tool_def()
        choice = ModelLlmToolChoice(mode="auto")
        req = ModelLlmInferenceRequest(
            **_chat_kwargs(tools=(tool,), tool_choice=choice),
        )
        assert len(req.tools) == 1
        assert req.tool_choice is not None
        assert req.tool_choice.mode == "auto"

    def test_explicit_correlation_id(self) -> None:
        """Caller-provided correlation_id overrides the default."""
        cid = uuid4()
        req = ModelLlmInferenceRequest(**_chat_kwargs(correlation_id=cid))
        assert req.correlation_id == cid


# =============================================================================
# Stream Guard Tests
# =============================================================================


class TestStreamGuard:
    """Tests for the stream=True type-level guard (Literal[False])."""

    def test_stream_true_rejected(self) -> None:
        """stream=True is rejected by the Literal[False] type guard.

        This catches accidental removal of the stream guard, which would
        allow streaming requests that the v1 handler cannot process.
        """
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(stream=True))

    def test_stream_default_is_false(self) -> None:
        """Default stream value is False."""
        req = ModelLlmInferenceRequest(**_chat_kwargs())
        assert req.stream is False

    def test_stream_explicit_false_accepted(self) -> None:
        """Explicitly passing stream=False is valid."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(stream=False))
        assert req.stream is False


# =============================================================================
# CHAT_COMPLETION Operation Type Tests
# =============================================================================


class TestChatCompletionValidation:
    """Tests for CHAT_COMPLETION operation type constraints."""

    def test_empty_messages_raises(self) -> None:
        """CHAT_COMPLETION with empty messages raises ValueError."""
        with pytest.raises(ValidationError, match="messages must be non-empty"):
            ModelLlmInferenceRequest(**_chat_kwargs(messages=()))

    def test_prompt_with_chat_completion_raises(self) -> None:
        """CHAT_COMPLETION with prompt set raises ValueError."""
        with pytest.raises(ValidationError, match="prompt must be None"):
            ModelLlmInferenceRequest(
                **_chat_kwargs(prompt="should not be here"),
            )

    def test_tool_choice_without_tools_raises(self) -> None:
        """CHAT_COMPLETION with tool_choice but no tools raises ValueError."""
        choice = ModelLlmToolChoice(mode="auto")
        with pytest.raises(
            ValidationError, match="tools must be non-empty when tool_choice is set"
        ):
            ModelLlmInferenceRequest(**_chat_kwargs(tool_choice=choice))


# =============================================================================
# COMPLETION Operation Type Tests
# =============================================================================


class TestCompletionValidation:
    """Tests for COMPLETION operation type constraints."""

    def test_none_prompt_raises(self) -> None:
        """COMPLETION with None prompt raises ValueError."""
        with pytest.raises(
            ValidationError, match="prompt must be non-None and non-empty"
        ):
            ModelLlmInferenceRequest(**_completion_kwargs(prompt=None))

    def test_empty_prompt_raises(self) -> None:
        """COMPLETION with empty string prompt raises ValueError."""
        with pytest.raises(
            ValidationError, match="prompt must be non-None and non-empty"
        ):
            ModelLlmInferenceRequest(**_completion_kwargs(prompt=""))

    def test_whitespace_only_prompt_raises(self) -> None:
        """COMPLETION with whitespace-only prompt raises ValueError."""
        with pytest.raises(
            ValidationError, match="prompt must be non-None and non-empty"
        ):
            ModelLlmInferenceRequest(**_completion_kwargs(prompt="   "))

    def test_messages_with_completion_raises(self) -> None:
        """COMPLETION with messages set raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="messages must be empty when operation_type is COMPLETION",
        ):
            ModelLlmInferenceRequest(
                **_completion_kwargs(messages=(_user_message(),)),
            )

    def test_system_prompt_with_completion_raises(self) -> None:
        """COMPLETION with system_prompt raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="system_prompt must be None when operation_type is COMPLETION",
        ):
            ModelLlmInferenceRequest(
                **_completion_kwargs(system_prompt="Be helpful"),
            )

    def test_tools_with_completion_raises(self) -> None:
        """COMPLETION with tools raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="tools must be empty when operation_type is COMPLETION",
        ):
            ModelLlmInferenceRequest(
                **_completion_kwargs(tools=(_tool_def(),)),
            )

    def test_tool_choice_with_completion_raises(self) -> None:
        """COMPLETION with tool_choice raises ValueError."""
        choice = ModelLlmToolChoice(mode="auto")
        with pytest.raises(
            ValidationError,
            match="tool_choice must be None when operation_type is COMPLETION",
        ):
            ModelLlmInferenceRequest(
                **_completion_kwargs(tool_choice=choice),
            )


# =============================================================================
# EMBEDDING Operation Type Tests
# =============================================================================


class TestEmbeddingValidation:
    """Tests for EMBEDDING operation type constraints."""

    def test_none_prompt_raises(self) -> None:
        """EMBEDDING with None prompt raises ValueError."""
        with pytest.raises(
            ValidationError, match="prompt must be non-None and non-empty"
        ):
            ModelLlmInferenceRequest(**_embedding_kwargs(prompt=None))

    def test_messages_with_embedding_raises(self) -> None:
        """EMBEDDING with messages raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="messages must be empty when operation_type is EMBEDDING",
        ):
            ModelLlmInferenceRequest(
                **_embedding_kwargs(messages=(_user_message(),)),
            )

    def test_max_tokens_with_embedding_raises(self) -> None:
        """EMBEDDING with max_tokens raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="max_tokens must be None when operation_type is EMBEDDING",
        ):
            ModelLlmInferenceRequest(**_embedding_kwargs(max_tokens=100))

    def test_temperature_with_embedding_raises(self) -> None:
        """EMBEDDING with temperature raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="temperature must be None when operation_type is EMBEDDING",
        ):
            ModelLlmInferenceRequest(**_embedding_kwargs(temperature=0.5))

    def test_tools_with_embedding_raises(self) -> None:
        """EMBEDDING with tools raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="tools must be empty when operation_type is EMBEDDING",
        ):
            ModelLlmInferenceRequest(**_embedding_kwargs(tools=(_tool_def(),)))


# =============================================================================
# max_tokens Boundary Tests
# =============================================================================


class TestMaxTokensBoundary:
    """Tests for max_tokens field boundary conditions."""

    def test_max_tokens_one_valid(self) -> None:
        """max_tokens=1 is the minimum valid value."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(max_tokens=1))
        assert req.max_tokens == 1

    def test_max_tokens_128000_valid(self) -> None:
        """max_tokens=128000 is the maximum valid value."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(max_tokens=128_000))
        assert req.max_tokens == 128_000

    def test_max_tokens_zero_rejected(self) -> None:
        """max_tokens=0 is below minimum (ge=1)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_tokens=0))

    def test_max_tokens_negative_rejected(self) -> None:
        """Negative max_tokens is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_tokens=-1))

    def test_max_tokens_128001_rejected(self) -> None:
        """max_tokens=128001 exceeds the maximum (le=128000)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_tokens=128_001))

    def test_max_tokens_none_valid(self) -> None:
        """max_tokens=None is valid (optional field)."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(max_tokens=None))
        assert req.max_tokens is None


# =============================================================================
# temperature Boundary Tests
# =============================================================================


class TestTemperatureBoundary:
    """Tests for temperature field boundary conditions."""

    def test_temperature_zero_valid(self) -> None:
        """temperature=0.0 is the minimum valid value (deterministic)."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(temperature=0.0))
        assert req.temperature == 0.0

    def test_temperature_two_valid(self) -> None:
        """temperature=2.0 is the maximum valid value."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(temperature=2.0))
        assert req.temperature == 2.0

    def test_temperature_negative_rejected(self) -> None:
        """Negative temperature is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(temperature=-0.1))

    def test_temperature_above_two_rejected(self) -> None:
        """temperature > 2.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(temperature=2.1))

    def test_temperature_none_valid(self) -> None:
        """temperature=None is valid (optional field)."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(temperature=None))
        assert req.temperature is None


# =============================================================================
# base_url Validation Tests
# =============================================================================


class TestBaseUrlValidation:
    """Tests for base_url field validator."""

    def test_missing_scheme_rejected(self) -> None:
        """base_url without http:// or https:// is rejected."""
        with pytest.raises(ValidationError, match="base_url must start with http"):
            ModelLlmInferenceRequest(**_chat_kwargs(base_url="192.168.86.201:8000"))

    def test_ftp_scheme_rejected(self) -> None:
        """Non-HTTP scheme is rejected."""
        with pytest.raises(ValidationError, match="base_url must start with http"):
            ModelLlmInferenceRequest(
                **_chat_kwargs(base_url="ftp://192.168.86.201:8000")
            )

    def test_scheme_only_rejected(self) -> None:
        """Scheme without host is rejected."""
        with pytest.raises(ValidationError, match="host"):
            ModelLlmInferenceRequest(**_chat_kwargs(base_url="http://"))

    def test_empty_base_url_rejected(self) -> None:
        """Empty string base_url is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(base_url=""))

    def test_https_accepted(self) -> None:
        """HTTPS scheme is accepted."""
        req = ModelLlmInferenceRequest(
            **_chat_kwargs(base_url="https://api.openai.com"),
        )
        assert req.base_url == "https://api.openai.com"


# =============================================================================
# system_prompt Validation Tests
# =============================================================================


class TestSystemPromptValidation:
    """Tests for system_prompt validator."""

    def test_whitespace_only_system_prompt_rejected(self) -> None:
        """Whitespace-only system_prompt is rejected."""
        with pytest.raises(ValidationError, match="system_prompt must be non-empty"):
            ModelLlmInferenceRequest(**_chat_kwargs(system_prompt="   "))

    def test_empty_string_system_prompt_rejected(self) -> None:
        """Empty string system_prompt is rejected."""
        with pytest.raises(ValidationError, match="system_prompt must be non-empty"):
            ModelLlmInferenceRequest(**_chat_kwargs(system_prompt=""))

    def test_none_system_prompt_valid(self) -> None:
        """None system_prompt is valid (optional)."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(system_prompt=None))
        assert req.system_prompt is None


# =============================================================================
# stop Sequence Validation Tests
# =============================================================================


class TestStopSequenceValidation:
    """Tests for stop sequence validator."""

    def test_empty_stop_sequence_rejected(self) -> None:
        """Empty string in stop sequences is rejected."""
        with pytest.raises(
            ValidationError, match="stop sequence at index 0 must be non-empty"
        ):
            ModelLlmInferenceRequest(**_chat_kwargs(stop=("",)))

    def test_whitespace_only_stop_sequence_rejected(self) -> None:
        """Whitespace-only stop sequence is rejected."""
        with pytest.raises(
            ValidationError, match="stop sequence at index 0 must be non-empty"
        ):
            ModelLlmInferenceRequest(**_chat_kwargs(stop=("   ",)))

    def test_valid_stop_sequences(self) -> None:
        """Valid stop sequences are accepted."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(stop=("###", "<|end|>")))
        assert req.stop == ("###", "<|end|>")


# =============================================================================
# Immutability Tests
# =============================================================================


class TestModelLlmInferenceRequestImmutability:
    """Tests for frozen=True immutability enforcement."""

    def test_frozen_base_url(self) -> None:
        """Cannot reassign base_url after construction."""
        req = ModelLlmInferenceRequest(**_chat_kwargs())
        with pytest.raises(ValidationError):
            req.base_url = "http://other:8000"  # type: ignore[misc]

    def test_frozen_model(self) -> None:
        """Cannot reassign model after construction."""
        req = ModelLlmInferenceRequest(**_chat_kwargs())
        with pytest.raises(ValidationError):
            req.model = "other-model"  # type: ignore[misc]


# =============================================================================
# extra="forbid" Tests
# =============================================================================


class TestExtraFieldsRejected:
    """Tests for extra='forbid' enforcement."""

    def test_unknown_field_rejected(self) -> None:
        """Unknown field is rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(unknown_field="surprise"))


# =============================================================================
# Required Field Tests
# =============================================================================


class TestRequiredFields:
    """Tests for required field validation."""

    def test_missing_base_url_rejected(self) -> None:
        """Missing base_url raises ValidationError."""
        kwargs = _chat_kwargs()
        del kwargs["base_url"]
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**kwargs)

    def test_missing_model_rejected(self) -> None:
        """Missing model raises ValidationError."""
        kwargs = _chat_kwargs()
        del kwargs["model"]
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**kwargs)

    def test_empty_model_rejected(self) -> None:
        """Empty string model is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(model=""))


# =============================================================================
# Resilience Parameter Tests
# =============================================================================


class TestResilienceParameters:
    """Tests for timeout_seconds and max_retries bounds."""

    def test_timeout_below_minimum_rejected(self) -> None:
        """timeout_seconds below 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=0.5))

    def test_timeout_above_maximum_rejected(self) -> None:
        """timeout_seconds above 600.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=601.0))

    def test_max_retries_negative_rejected(self) -> None:
        """Negative max_retries is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_retries=-1))

    def test_max_retries_above_ten_rejected(self) -> None:
        """max_retries > 10 is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_retries=11))


# =============================================================================
# ToolChoice Validator Tests
# =============================================================================


class TestToolChoiceValidation:
    """Tests for ModelLlmToolChoice validator consistency."""

    def test_mode_function_without_function_name_raises(self) -> None:
        """mode='function' without function_name raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="function_name is required when mode is 'function'",
        ):
            ModelLlmToolChoice(mode="function")

    def test_mode_auto_with_function_name_raises(self) -> None:
        """mode='auto' with function_name raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="function_name must be None when mode is not 'function'",
        ):
            ModelLlmToolChoice(mode="auto", function_name="search")

    def test_mode_none_with_function_name_raises(self) -> None:
        """mode='none' with function_name raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="function_name must be None when mode is not 'function'",
        ):
            ModelLlmToolChoice(mode="none", function_name="search")

    def test_mode_required_with_function_name_raises(self) -> None:
        """mode='required' with function_name raises ValueError."""
        with pytest.raises(
            ValidationError,
            match="function_name must be None when mode is not 'function'",
        ):
            ModelLlmToolChoice(mode="required", function_name="search")

    def test_mode_function_with_function_name_valid(self) -> None:
        """mode='function' with function_name is valid."""
        choice = ModelLlmToolChoice(mode="function", function_name="search")
        assert choice.mode == "function"
        assert choice.function_name == "search"

    def test_mode_auto_valid(self) -> None:
        """mode='auto' without function_name is valid."""
        choice = ModelLlmToolChoice(mode="auto")
        assert choice.mode == "auto"
        assert choice.function_name is None

    def test_mode_none_valid(self) -> None:
        """mode='none' without function_name is valid."""
        choice = ModelLlmToolChoice(mode="none")
        assert choice.mode == "none"

    def test_mode_required_valid(self) -> None:
        """mode='required' without function_name is valid."""
        choice = ModelLlmToolChoice(mode="required")
        assert choice.mode == "required"

    def test_function_name_empty_string_rejected(self) -> None:
        """Empty string function_name is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            ModelLlmToolChoice(mode="function", function_name="")
