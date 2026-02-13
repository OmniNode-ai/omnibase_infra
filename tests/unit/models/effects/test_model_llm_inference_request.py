# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelLlmInferenceRequest.

Tests validate:
- Minimal and full construction for each operation mode
- base_url field-level validation (scheme, host)
- system_prompt field-level validation (empty/whitespace rejection)
- stop sequence field-level validation (empty/whitespace entry rejection)
- CHAT_COMPLETION cross-field invariants
- COMPLETION cross-field invariants
- EMBEDDING cross-field invariants
- Generation parameter boundary constraints
- Resilience parameter bounds
- Stream Literal[False] type guard
- Frozen immutability and extra field rejection
- Serialization round-trip for all three modes

Test Organization:
    - TestConstruction: Basic instantiation (5 tests)
    - TestBaseUrlValidator: URL scheme/host validation (7 tests)
    - TestSystemPromptValidator: Empty/whitespace rejection (3 tests)
    - TestStopSequenceValidator: Entry-level validation (4 tests)
    - TestChatCompletionInvariants: CHAT_COMPLETION cross-field rules (7 tests)
    - TestCompletionInvariants: COMPLETION cross-field rules (7 tests)
    - TestEmbeddingInvariants: EMBEDDING cross-field rules (11 tests)
    - TestGenerationParamBounds: Boundary constraints (8 tests)
    - TestResilienceParams: Timeout and retry bounds (4 tests)
    - TestStreamGuard: Literal[False] enforcement (2 tests)
    - TestImmutability: Frozen / extra forbid (2 tests)
    - TestSerialization: Dump / JSON round-trips (4 tests)
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

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

# ==============================================================================
# Helpers
# ==============================================================================


def _make_message(
    role: str = "user",
    content: str | None = "Hello",
    **kwargs: Any,
) -> ModelLlmMessage:
    """Build a valid ModelLlmMessage with sensible defaults."""
    return ModelLlmMessage(role=role, content=content, **kwargs)


def _make_tool_def(name: str = "get_weather") -> ModelLlmToolDefinition:
    """Build a valid ModelLlmToolDefinition with a minimal function def."""
    return ModelLlmToolDefinition(
        function=ModelLlmFunctionDef(name=name),
    )


def _chat_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid CHAT_COMPLETION request with sensible defaults."""
    defaults: dict[str, Any] = {
        "base_url": "http://192.168.86.201:8000",
        "model": "qwen2.5-coder-14b",
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "messages": (_make_message(),),
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _completion_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid COMPLETION request with sensible defaults."""
    defaults: dict[str, Any] = {
        "base_url": "http://192.168.86.201:8000",
        "model": "qwen2.5-coder-14b",
        "operation_type": EnumLlmOperationType.COMPLETION,
        "prompt": "Once upon a time",
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _embedding_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid EMBEDDING request with sensible defaults."""
    defaults: dict[str, Any] = {
        "base_url": "http://192.168.86.201:8002",
        "model": "gte-qwen2",
        "operation_type": EnumLlmOperationType.EMBEDDING,
        "prompt": "Embed this text",
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


# ==============================================================================
# Construction
# ==============================================================================


class TestConstruction:
    """Tests for basic model instantiation across operation modes."""

    def test_minimal_chat_completion(self) -> None:
        """Test minimal CHAT_COMPLETION construction with required fields only."""
        req = _chat_request()

        assert req.base_url == "http://192.168.86.201:8000"
        assert req.model == "qwen2.5-coder-14b"
        assert req.operation_type == EnumLlmOperationType.CHAT_COMPLETION
        assert len(req.messages) == 1
        assert req.prompt is None
        assert req.tools == ()
        assert req.tool_choice is None
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.top_p is None
        assert req.stop == ()
        assert req.stream is False
        assert req.timeout_seconds == 60.0
        assert req.max_retries == 3
        assert req.provider_label == ""
        assert req.metadata == {}

    def test_full_chat_completion_with_all_optional_fields(self) -> None:
        """Test CHAT_COMPLETION with all optional fields populated."""
        tool = _make_tool_def("search")
        choice = ModelLlmToolChoice(mode="auto")
        cid = UUID("12345678-1234-5678-1234-567812345678")
        eid = UUID("abcdefab-cdef-abcd-efab-cdefabcdefab")

        req = _chat_request(
            provider_label="local-vllm",
            system_prompt="You are helpful.",
            tools=(tool,),
            tool_choice=choice,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=("END",),
            timeout_seconds=120.0,
            max_retries=5,
            correlation_id=cid,
            execution_id=eid,
            metadata={"env": "test"},
        )

        assert req.provider_label == "local-vllm"
        assert req.system_prompt == "You are helpful."
        assert len(req.tools) == 1
        assert req.tool_choice is choice
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stop == ("END",)
        assert req.timeout_seconds == 120.0
        assert req.max_retries == 5
        assert req.correlation_id == cid
        assert req.execution_id == eid
        assert req.metadata == {"env": "test"}

    def test_completion_mode(self) -> None:
        """Test COMPLETION mode construction with required prompt."""
        req = _completion_request()

        assert req.operation_type == EnumLlmOperationType.COMPLETION
        assert req.prompt == "Once upon a time"
        assert req.messages == ()

    def test_embedding_mode(self) -> None:
        """Test EMBEDDING mode construction with required prompt."""
        req = _embedding_request()

        assert req.operation_type == EnumLlmOperationType.EMBEDDING
        assert req.prompt == "Embed this text"
        assert req.messages == ()

    def test_auto_generated_uuids(self) -> None:
        """Test that correlation_id and execution_id are auto-generated."""
        req = _chat_request()

        assert isinstance(req.correlation_id, UUID)
        assert isinstance(req.execution_id, UUID)
        # Two separate constructions should yield distinct IDs
        req2 = _chat_request()
        assert req.correlation_id != req2.correlation_id
        assert req.execution_id != req2.execution_id


# ==============================================================================
# Base URL Validator
# ==============================================================================


class TestBaseUrlValidator:
    """Tests for base_url field validation (scheme and host checks)."""

    def test_base_url_required(self) -> None:
        """Test that omitting base_url raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(
                model="m",
                messages=(_make_message(),),
            )  # type: ignore[call-arg]

    def test_base_url_empty_rejected(self) -> None:
        """Test that empty string base_url is rejected."""
        with pytest.raises(ValidationError, match="base_url"):
            _chat_request(base_url="")

    def test_base_url_no_scheme_rejected(self) -> None:
        """Test that a URL without http/https scheme is rejected."""
        with pytest.raises(ValidationError, match="http://"):
            _chat_request(base_url="192.168.86.201:8000")

    def test_base_url_ftp_rejected(self) -> None:
        """Test that ftp:// scheme is rejected."""
        with pytest.raises(ValidationError, match="http://"):
            _chat_request(base_url="ftp://host:8000")

    def test_base_url_scheme_only_rejected(self) -> None:
        """Test that scheme-only URL without host is rejected."""
        with pytest.raises(ValidationError, match="host"):
            _chat_request(base_url="http://")

    def test_base_url_http_valid(self) -> None:
        """Test that http:// with host is accepted."""
        req = _chat_request(base_url="http://localhost:8000")
        assert req.base_url == "http://localhost:8000"

    def test_base_url_https_valid(self) -> None:
        """Test that https:// with host is accepted."""
        req = _chat_request(base_url="https://api.openai.com/v1")
        assert req.base_url == "https://api.openai.com/v1"


# ==============================================================================
# System Prompt Validator
# ==============================================================================


class TestSystemPromptValidator:
    """Tests for system_prompt field validation."""

    def test_system_prompt_none_valid(self) -> None:
        """Test that None system_prompt passes validation."""
        req = _chat_request(system_prompt=None)
        assert req.system_prompt is None

    def test_system_prompt_empty_rejected(self) -> None:
        """Test that empty string system_prompt is rejected."""
        with pytest.raises(ValidationError, match="system_prompt"):
            _chat_request(system_prompt="")

    def test_system_prompt_whitespace_rejected(self) -> None:
        """Test that whitespace-only system_prompt is rejected."""
        with pytest.raises(ValidationError, match="system_prompt"):
            _chat_request(system_prompt="   \t\n  ")


# ==============================================================================
# Stop Sequence Validator
# ==============================================================================


class TestStopSequenceValidator:
    """Tests for stop sequence entry-level validation."""

    def test_empty_tuple_valid(self) -> None:
        """Test that an empty stop tuple passes validation."""
        req = _chat_request(stop=())
        assert req.stop == ()

    def test_valid_sequences(self) -> None:
        """Test that non-empty stop sequences pass validation."""
        req = _chat_request(stop=("STOP", "END", "###"))
        assert req.stop == ("STOP", "END", "###")

    def test_empty_entry_rejected(self) -> None:
        """Test that an empty string entry in stop is rejected."""
        with pytest.raises(ValidationError, match="stop sequence at index"):
            _chat_request(stop=("valid", ""))

    def test_whitespace_entry_rejected(self) -> None:
        """Test that a whitespace-only entry in stop is rejected."""
        with pytest.raises(ValidationError, match="stop sequence at index"):
            _chat_request(stop=("  \t",))


# ==============================================================================
# Chat Completion Invariants
# ==============================================================================


class TestChatCompletionInvariants:
    """Tests for CHAT_COMPLETION cross-field invariants."""

    def test_empty_messages_rejected(self) -> None:
        """Test that CHAT_COMPLETION with empty messages is rejected."""
        with pytest.raises(ValidationError, match="messages must be non-empty"):
            _chat_request(messages=())

    def test_prompt_set_rejected(self) -> None:
        """Test that CHAT_COMPLETION with prompt set is rejected."""
        with pytest.raises(ValidationError, match="prompt must be None"):
            _chat_request(prompt="some text")

    def test_tool_choice_without_tools_rejected(self) -> None:
        """Test that tool_choice without tools is rejected."""
        with pytest.raises(ValidationError, match="tools must be non-empty"):
            _chat_request(
                tool_choice=ModelLlmToolChoice(mode="auto"),
                tools=(),
            )

    def test_tools_with_tool_choice_valid(self) -> None:
        """Test that tools + tool_choice together are accepted."""
        tool = _make_tool_def("search")
        choice = ModelLlmToolChoice(mode="auto")
        req = _chat_request(tools=(tool,), tool_choice=choice)

        assert len(req.tools) == 1
        assert req.tool_choice is choice

    def test_tools_without_tool_choice_valid(self) -> None:
        """Test that tools without tool_choice is accepted."""
        tool = _make_tool_def("search")
        req = _chat_request(tools=(tool,))

        assert len(req.tools) == 1
        assert req.tool_choice is None

    def test_system_prompt_valid(self) -> None:
        """Test that system_prompt is allowed with CHAT_COMPLETION."""
        req = _chat_request(system_prompt="Be concise.")
        assert req.system_prompt == "Be concise."

    def test_generation_params_valid(self) -> None:
        """Test that generation params are accepted with CHAT_COMPLETION."""
        req = _chat_request(
            max_tokens=512,
            temperature=0.5,
            top_p=0.8,
            stop=("END",),
        )
        assert req.max_tokens == 512
        assert req.temperature == 0.5
        assert req.top_p == 0.8
        assert req.stop == ("END",)


# ==============================================================================
# Completion Invariants
# ==============================================================================


class TestCompletionInvariants:
    """Tests for COMPLETION cross-field invariants."""

    def test_prompt_none_rejected(self) -> None:
        """Test that COMPLETION with None prompt is rejected."""
        with pytest.raises(ValidationError, match="prompt must be non-None"):
            _completion_request(prompt=None)

    def test_prompt_empty_rejected(self) -> None:
        """Test that COMPLETION with empty prompt is rejected."""
        with pytest.raises(ValidationError, match="prompt must be non-None"):
            _completion_request(prompt="")

    def test_prompt_whitespace_rejected(self) -> None:
        """Test that COMPLETION with whitespace-only prompt is rejected."""
        with pytest.raises(ValidationError, match="prompt must be non-None"):
            _completion_request(prompt="   ")

    def test_messages_rejected(self) -> None:
        """Test that COMPLETION with messages is rejected."""
        with pytest.raises(ValidationError, match="messages must be empty"):
            _completion_request(messages=(_make_message(),))

    def test_system_prompt_rejected(self) -> None:
        """Test that COMPLETION with system_prompt is rejected."""
        with pytest.raises(ValidationError, match="system_prompt must be None"):
            _completion_request(system_prompt="Be helpful.")

    def test_tools_rejected(self) -> None:
        """Test that COMPLETION with tools is rejected."""
        with pytest.raises(ValidationError, match="tools must be empty"):
            _completion_request(tools=(_make_tool_def(),))

    def test_tool_choice_rejected(self) -> None:
        """Test that COMPLETION with tool_choice is rejected."""
        with pytest.raises(ValidationError, match="tool_choice must be None"):
            _completion_request(tool_choice=ModelLlmToolChoice(mode="auto"))


# ==============================================================================
# Embedding Invariants
# ==============================================================================


class TestEmbeddingInvariants:
    """Tests for EMBEDDING cross-field invariants."""

    def test_prompt_none_rejected(self) -> None:
        """Test that EMBEDDING with None prompt is rejected."""
        with pytest.raises(ValidationError, match="prompt must be non-None"):
            _embedding_request(prompt=None)

    def test_prompt_empty_rejected(self) -> None:
        """Test that EMBEDDING with empty prompt is rejected."""
        with pytest.raises(ValidationError, match="prompt must be non-None"):
            _embedding_request(prompt="")

    def test_prompt_whitespace_rejected(self) -> None:
        """Test that EMBEDDING with whitespace-only prompt is rejected."""
        with pytest.raises(ValidationError, match="prompt must be non-None"):
            _embedding_request(prompt="   ")

    def test_messages_rejected(self) -> None:
        """Test that EMBEDDING with messages is rejected."""
        with pytest.raises(ValidationError, match="messages must be empty"):
            _embedding_request(messages=(_make_message(),))

    def test_tools_rejected(self) -> None:
        """Test that EMBEDDING with tools is rejected."""
        with pytest.raises(ValidationError, match="tools must be empty"):
            _embedding_request(tools=(_make_tool_def(),))

    def test_tool_choice_rejected(self) -> None:
        """Test that EMBEDDING with tool_choice is rejected."""
        with pytest.raises(ValidationError, match="tool_choice must be None"):
            _embedding_request(tool_choice=ModelLlmToolChoice(mode="auto"))

    def test_system_prompt_rejected(self) -> None:
        """Test that EMBEDDING with system_prompt is rejected."""
        with pytest.raises(ValidationError, match="system_prompt must be None"):
            _embedding_request(system_prompt="Embed as code.")

    def test_max_tokens_rejected(self) -> None:
        """Test that EMBEDDING with max_tokens set is rejected."""
        with pytest.raises(ValidationError, match="max_tokens must be None"):
            _embedding_request(max_tokens=100)

    def test_temperature_rejected(self) -> None:
        """Test that EMBEDDING with temperature set is rejected."""
        with pytest.raises(ValidationError, match="temperature must be None"):
            _embedding_request(temperature=0.5)

    def test_top_p_rejected(self) -> None:
        """Test that EMBEDDING with top_p set is rejected."""
        with pytest.raises(ValidationError, match="top_p must be None"):
            _embedding_request(top_p=0.9)

    def test_stop_rejected(self) -> None:
        """Test that EMBEDDING with non-empty stop is rejected."""
        with pytest.raises(ValidationError, match="stop must be empty"):
            _embedding_request(stop=("END",))


# ==============================================================================
# Generation Parameter Bounds
# ==============================================================================


class TestGenerationParamBounds:
    """Tests for generation parameter boundary constraints."""

    def test_max_tokens_zero_rejected(self) -> None:
        """Test that max_tokens=0 is rejected (ge=1)."""
        with pytest.raises(ValidationError, match="max_tokens"):
            _chat_request(max_tokens=0)

    def test_max_tokens_above_limit_rejected(self) -> None:
        """Test that max_tokens=128001 is rejected (le=128000)."""
        with pytest.raises(ValidationError, match="max_tokens"):
            _chat_request(max_tokens=128_001)

    def test_max_tokens_boundary_values_valid(self) -> None:
        """Test that boundary values 1 and 128000 are accepted."""
        req_low = _chat_request(max_tokens=1)
        assert req_low.max_tokens == 1

        req_high = _chat_request(max_tokens=128_000)
        assert req_high.max_tokens == 128_000

    def test_temperature_below_zero_rejected(self) -> None:
        """Test that temperature < 0.0 is rejected."""
        with pytest.raises(ValidationError, match="temperature"):
            _chat_request(temperature=-0.1)

    def test_temperature_above_limit_rejected(self) -> None:
        """Test that temperature > 2.0 is rejected."""
        with pytest.raises(ValidationError, match="temperature"):
            _chat_request(temperature=2.1)

    def test_temperature_boundary_values_valid(self) -> None:
        """Test that temperature 0.0 and 2.0 are accepted."""
        req_low = _chat_request(temperature=0.0)
        assert req_low.temperature == 0.0

        req_high = _chat_request(temperature=2.0)
        assert req_high.temperature == 2.0

    def test_top_p_below_zero_rejected(self) -> None:
        """Test that top_p < 0.0 is rejected."""
        with pytest.raises(ValidationError, match="top_p"):
            _chat_request(top_p=-0.01)

    def test_top_p_above_one_rejected(self) -> None:
        """Test that top_p > 1.0 is rejected."""
        with pytest.raises(ValidationError, match="top_p"):
            _chat_request(top_p=1.01)


# ==============================================================================
# Resilience Parameters
# ==============================================================================


class TestResilienceParams:
    """Tests for timeout_seconds and max_retries bounds."""

    def test_timeout_below_minimum_rejected(self) -> None:
        """Test that timeout_seconds < 1.0 is rejected."""
        with pytest.raises(ValidationError, match="timeout_seconds"):
            _chat_request(timeout_seconds=0.5)

    def test_timeout_above_maximum_rejected(self) -> None:
        """Test that timeout_seconds > 600.0 is rejected."""
        with pytest.raises(ValidationError, match="timeout_seconds"):
            _chat_request(timeout_seconds=601.0)

    def test_max_retries_below_minimum_rejected(self) -> None:
        """Test that max_retries < 0 is rejected."""
        with pytest.raises(ValidationError, match="max_retries"):
            _chat_request(max_retries=-1)

    def test_max_retries_above_maximum_rejected(self) -> None:
        """Test that max_retries > 10 is rejected."""
        with pytest.raises(ValidationError, match="max_retries"):
            _chat_request(max_retries=11)


# ==============================================================================
# Stream Guard
# ==============================================================================


class TestStreamGuard:
    """Tests for Literal[False] stream enforcement."""

    def test_stream_false_valid(self) -> None:
        """Test that stream=False is accepted."""
        req = _chat_request(stream=False)
        assert req.stream is False

    def test_stream_true_rejected(self) -> None:
        """Test that stream=True is rejected by Literal[False]."""
        with pytest.raises(ValidationError, match="stream"):
            _chat_request(stream=True)


# ==============================================================================
# Immutability
# ==============================================================================


class TestImmutability:
    """Tests for frozen model and extra field rejection."""

    def test_frozen_immutability(self) -> None:
        """Test that assigning to fields raises ValidationError."""
        req = _chat_request()

        with pytest.raises(ValidationError):
            req.model = "other-model"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            req.base_url = "http://other:8000"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that supplying an extra field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _chat_request(unknown_field="value")

        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)


# ==============================================================================
# Metadata Validation
# ==============================================================================


class TestMetadataValidation:
    """Tests for metadata field type enforcement."""

    def test_metadata_non_string_values_rejected(self) -> None:
        """Test that metadata with non-string values raises ValidationError."""
        with pytest.raises(ValidationError, match="metadata"):
            _chat_request(metadata={"key": 123})

    def test_metadata_non_string_keys_rejected(self) -> None:
        """Test that metadata with non-string keys raises ValidationError."""
        with pytest.raises(ValidationError, match="metadata"):
            _chat_request(metadata={42: "value"})  # type: ignore[dict-item]


# ==============================================================================
# model_copy Regression
# ==============================================================================


class TestModelCopy:
    """Tests for model_copy behaviour, guarding against metadata repr regressions.

    See commit ec045c42 which reverted MappingProxyType metadata because it
    broke model_copy.
    """

    def test_model_copy_preserves_metadata(self) -> None:
        """Test that model_copy(update=...) succeeds and preserves metadata."""
        original = _chat_request(metadata={"env": "staging", "team": "infra"})
        copied = original.model_copy(update={"model": "new-model"})

        assert copied.model == "new-model"
        assert copied.metadata == {"env": "staging", "team": "infra"}
        # Other fields should carry over
        assert copied.base_url == original.base_url
        assert copied.correlation_id == original.correlation_id

    def test_model_copy_with_empty_metadata(self) -> None:
        """Test that model_copy works when metadata is the default empty dict."""
        original = _chat_request()
        copied = original.model_copy(update={"model": "another-model"})

        assert copied.model == "another-model"
        assert copied.metadata == {}


# ==============================================================================
# Serialization
# ==============================================================================


class TestSerialization:
    """Tests for model_dump, JSON output, and round-trip reconstruction."""

    def test_roundtrip_chat_completion(self) -> None:
        """Test model_dump round-trip for CHAT_COMPLETION."""
        tool = _make_tool_def("search")
        choice = ModelLlmToolChoice(mode="auto")
        req = _chat_request(
            tools=(tool,),
            tool_choice=choice,
            max_tokens=256,
            system_prompt="Be brief.",
        )
        dumped = req.model_dump()
        reconstructed = ModelLlmInferenceRequest(**dumped)

        assert reconstructed.base_url == req.base_url
        assert reconstructed.model == req.model
        assert reconstructed.operation_type == req.operation_type
        assert len(reconstructed.messages) == len(req.messages)
        assert len(reconstructed.tools) == len(req.tools)
        assert reconstructed.max_tokens == req.max_tokens
        assert reconstructed.system_prompt == req.system_prompt
        assert reconstructed.correlation_id == req.correlation_id
        assert reconstructed.execution_id == req.execution_id

    def test_roundtrip_completion(self) -> None:
        """Test model_dump round-trip for COMPLETION."""
        req = _completion_request(max_tokens=100, temperature=0.8)
        dumped = req.model_dump()
        reconstructed = ModelLlmInferenceRequest(**dumped)

        assert reconstructed.operation_type == EnumLlmOperationType.COMPLETION
        assert reconstructed.prompt == req.prompt
        assert reconstructed.max_tokens == 100
        assert reconstructed.temperature == 0.8

    def test_roundtrip_embedding(self) -> None:
        """Test model_dump round-trip for EMBEDDING."""
        req = _embedding_request()
        dumped = req.model_dump()
        reconstructed = ModelLlmInferenceRequest(**dumped)

        assert reconstructed.operation_type == EnumLlmOperationType.EMBEDDING
        assert reconstructed.prompt == req.prompt
        assert reconstructed.max_tokens is None
        assert reconstructed.temperature is None

    def test_model_dump_json_valid(self) -> None:
        """Test that model_dump_json produces valid parseable JSON."""
        req = _chat_request(max_tokens=512, metadata={"key": "value"})
        raw_json = req.model_dump_json()
        parsed = json.loads(raw_json)

        assert parsed["base_url"] == "http://192.168.86.201:8000"
        assert parsed["model"] == "qwen2.5-coder-14b"
        assert parsed["operation_type"] == "chat_completion"
        assert parsed["max_tokens"] == 512
        assert parsed["metadata"] == {"key": "value"}
        assert len(parsed["messages"]) == 1
