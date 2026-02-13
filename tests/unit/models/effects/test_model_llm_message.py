# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive tests for ModelLlmMessage.

Tests cover:
- Valid construction for each role (system, user, assistant, tool)
- System role validation (content required, tool_calls/tool_call_id forbidden)
- User role validation (same constraints as system)
- Assistant role validation (content or tool_calls required, tool_call_id forbidden)
- Tool role validation (content + tool_call_id required, tool_calls forbidden)
- Field validation (role required, invalid role rejected)
- Frozen immutability (frozen=True) and extra field rejection (extra='forbid')
- Serialization roundtrip (model_dump, model_validate, equality)

OMN-2105: Phase 5 LLM inference request model
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)
from omnibase_infra.nodes.effects.models.model_llm_message import ModelLlmMessage
from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall


def _make_tool_call(
    call_id: str = "call_abc123",
    fn_name: str = "search",
    fn_args: str = '{"q": "hello"}',
) -> ModelLlmToolCall:
    """Build a valid ModelLlmToolCall instance for reuse in tests."""
    return ModelLlmToolCall(
        id=call_id,
        function=ModelLlmFunctionCall(name=fn_name, arguments=fn_args),
    )


# ============================================================================
# Construction Tests
# ============================================================================


class TestModelLlmMessageConstruction:
    """Tests for valid construction of ModelLlmMessage across all roles."""

    def test_system_message_with_content(self) -> None:
        """Test that a system message with content constructs successfully."""
        msg = ModelLlmMessage(role="system", content="You are a helpful assistant.")

        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."
        assert msg.tool_calls == ()
        assert msg.tool_call_id is None

    def test_user_message_with_content(self) -> None:
        """Test that a user message with content constructs successfully."""
        msg = ModelLlmMessage(role="user", content="Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.tool_calls == ()
        assert msg.tool_call_id is None

    def test_assistant_message_with_content(self) -> None:
        """Test that an assistant message with content only constructs successfully."""
        msg = ModelLlmMessage(role="assistant", content="Sure, I can help.")

        assert msg.role == "assistant"
        assert msg.content == "Sure, I can help."
        assert msg.tool_calls == ()
        assert msg.tool_call_id is None

    def test_assistant_message_with_tool_calls(self) -> None:
        """Test that an assistant message with tool_calls (no content) constructs."""
        tc = _make_tool_call()
        msg = ModelLlmMessage(role="assistant", tool_calls=(tc,))

        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0] is tc

    def test_assistant_message_with_content_and_tool_calls(self) -> None:
        """Test that an assistant message with both content and tool_calls constructs."""
        tc = _make_tool_call()
        msg = ModelLlmMessage(
            role="assistant",
            content="Let me search for that.",
            tool_calls=(tc,),
        )

        assert msg.role == "assistant"
        assert msg.content == "Let me search for that."
        assert len(msg.tool_calls) == 1

    def test_tool_message_with_content_and_tool_call_id(self) -> None:
        """Test that a tool message with content and tool_call_id constructs."""
        msg = ModelLlmMessage(
            role="tool",
            content='{"results": [1, 2, 3]}',
            tool_call_id="call_abc123",
        )

        assert msg.role == "tool"
        assert msg.content == '{"results": [1, 2, 3]}'
        assert msg.tool_call_id == "call_abc123"
        assert msg.tool_calls == ()


# ============================================================================
# System Role Validation Tests
# ============================================================================


class TestModelLlmMessageSystemRoleValidation:
    """Tests for system role field constraints."""

    def test_system_content_none_rejected(self) -> None:
        """Test that system messages with content=None are rejected."""
        with pytest.raises(ValidationError, match="content must be non-empty"):
            ModelLlmMessage(role="system", content=None)

    def test_system_content_empty_string_rejected(self) -> None:
        """Test that system messages with content='' are rejected."""
        with pytest.raises(ValidationError, match="content must be non-empty"):
            ModelLlmMessage(role="system", content="")

    def test_system_content_whitespace_rejected(self) -> None:
        """Test that system messages with whitespace-only content are rejected."""
        with pytest.raises(ValidationError, match="content must be non-empty"):
            ModelLlmMessage(role="system", content="   ")

    def test_system_tool_calls_non_empty_rejected(self) -> None:
        """Test that system messages with non-empty tool_calls are rejected."""
        tc = _make_tool_call()
        with pytest.raises(ValidationError, match="tool_calls must be empty"):
            ModelLlmMessage(role="system", content="Hello", tool_calls=(tc,))

    def test_system_tool_call_id_set_rejected(self) -> None:
        """Test that system messages with tool_call_id set are rejected."""
        with pytest.raises(ValidationError, match="tool_call_id must be None"):
            ModelLlmMessage(
                role="system",
                content="Hello",
                tool_call_id="call_123",
            )


# ============================================================================
# User Role Validation Tests
# ============================================================================


class TestModelLlmMessageUserRoleValidation:
    """Tests for user role field constraints."""

    def test_user_content_none_rejected(self) -> None:
        """Test that user messages with content=None are rejected."""
        with pytest.raises(ValidationError, match="content must be non-empty"):
            ModelLlmMessage(role="user", content=None)

    def test_user_content_empty_string_rejected(self) -> None:
        """Test that user messages with content='' are rejected."""
        with pytest.raises(ValidationError, match="content must be non-empty"):
            ModelLlmMessage(role="user", content="")

    def test_user_content_whitespace_rejected(self) -> None:
        """Test that user messages with whitespace-only content are rejected."""
        with pytest.raises(ValidationError, match="content must be non-empty"):
            ModelLlmMessage(role="user", content="   \t\n  ")

    def test_user_tool_calls_non_empty_rejected(self) -> None:
        """Test that user messages with non-empty tool_calls are rejected."""
        tc = _make_tool_call()
        with pytest.raises(ValidationError, match="tool_calls must be empty"):
            ModelLlmMessage(role="user", content="Hello", tool_calls=(tc,))

    def test_user_tool_call_id_set_rejected(self) -> None:
        """Test that user messages with tool_call_id set are rejected."""
        with pytest.raises(ValidationError, match="tool_call_id must be None"):
            ModelLlmMessage(
                role="user",
                content="Hello",
                tool_call_id="call_123",
            )


# ============================================================================
# Assistant Role Validation Tests
# ============================================================================


class TestModelLlmMessageAssistantRoleValidation:
    """Tests for assistant role field constraints."""

    def test_assistant_no_content_no_tool_calls_rejected(self) -> None:
        """Test that assistant messages with neither content nor tool_calls are rejected."""
        with pytest.raises(ValidationError, match="must have content or tool_calls"):
            ModelLlmMessage(role="assistant")

    def test_assistant_empty_content_no_tool_calls_rejected(self) -> None:
        """Test that assistant messages with empty content and no tool_calls are rejected."""
        with pytest.raises(ValidationError, match="must have content or tool_calls"):
            ModelLlmMessage(role="assistant", content="")

    def test_assistant_whitespace_content_no_tool_calls_rejected(self) -> None:
        """Test that whitespace-only content with no tool_calls is rejected."""
        with pytest.raises(ValidationError, match="must have content or tool_calls"):
            ModelLlmMessage(role="assistant", content="   ")

    def test_assistant_tool_call_id_set_rejected(self) -> None:
        """Test that assistant messages with tool_call_id set are rejected."""
        with pytest.raises(ValidationError, match="tool_call_id must be None"):
            ModelLlmMessage(
                role="assistant",
                content="Hi",
                tool_call_id="call_123",
            )

    def test_assistant_none_content_with_tool_calls_valid(self) -> None:
        """Test that None content with tool_calls present is valid for assistant."""
        tc = _make_tool_call()
        msg = ModelLlmMessage(role="assistant", content=None, tool_calls=(tc,))

        assert msg.content is None
        assert len(msg.tool_calls) == 1


# ============================================================================
# Tool Role Validation Tests
# ============================================================================


class TestModelLlmMessageToolRoleValidation:
    """Tests for tool role field constraints."""

    def test_tool_tool_call_id_none_rejected(self) -> None:
        """Test that tool messages with tool_call_id=None are rejected."""
        with pytest.raises(ValidationError, match="tool_call_id is required"):
            ModelLlmMessage(role="tool", content="result data")

    def test_tool_content_none_rejected(self) -> None:
        """Test that tool messages with content=None are rejected."""
        with pytest.raises(ValidationError, match="tool messages must include content"):
            ModelLlmMessage(role="tool", content=None, tool_call_id="call_123")

    def test_tool_content_empty_string_rejected(self) -> None:
        """Test that tool messages with content='' are rejected."""
        with pytest.raises(ValidationError, match="tool messages must include content"):
            ModelLlmMessage(role="tool", content="", tool_call_id="call_123")

    def test_tool_content_whitespace_rejected(self) -> None:
        """Test that tool messages with whitespace-only content are rejected."""
        with pytest.raises(ValidationError, match="tool messages must include content"):
            ModelLlmMessage(role="tool", content="   ", tool_call_id="call_123")

    def test_tool_tool_calls_non_empty_rejected(self) -> None:
        """Test that tool messages with non-empty tool_calls are rejected."""
        tc = _make_tool_call()
        with pytest.raises(ValidationError, match="tool_calls must be empty"):
            ModelLlmMessage(
                role="tool",
                content="result",
                tool_call_id="call_123",
                tool_calls=(tc,),
            )


# ============================================================================
# Field Validation Tests
# ============================================================================


class TestModelLlmMessageFieldValidation:
    """Tests for general field validation rules on ModelLlmMessage."""

    def test_role_is_required(self) -> None:
        """Test that omitting role raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmMessage(content="Hello")  # type: ignore[call-arg]

    def test_invalid_role_value_rejected(self) -> None:
        """Test that an invalid role value is rejected by the Literal constraint."""
        with pytest.raises(ValidationError):
            ModelLlmMessage(role="moderator", content="Hello")


# ============================================================================
# Immutability Tests
# ============================================================================


class TestModelLlmMessageImmutability:
    """Tests for frozen immutability of ModelLlmMessage."""

    def test_frozen_prevents_assignment(self) -> None:
        """Test that assigning to fields on a frozen instance raises ValidationError."""
        msg = ModelLlmMessage(role="user", content="Hello")

        with pytest.raises(ValidationError):
            msg.content = "New content"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            msg.role = "system"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmMessage(
                role="user",
                content="Hello",
                metadata="extra",  # type: ignore[call-arg]
            )

        assert "metadata" in str(exc_info.value)


# ============================================================================
# Serialization Tests
# ============================================================================


class TestModelLlmMessageSerialization:
    """Tests for serialization and deserialization of ModelLlmMessage."""

    def test_user_message_roundtrip(self) -> None:
        """Test model_dump -> model_validate roundtrip for a user message."""
        original = ModelLlmMessage(role="user", content="What is 2+2?")

        dumped = original.model_dump()
        restored = ModelLlmMessage.model_validate(dumped)

        assert restored == original
        assert restored.role == "user"
        assert restored.content == "What is 2+2?"

    def test_assistant_with_tool_calls_roundtrip(self) -> None:
        """Test roundtrip for an assistant message containing tool_calls."""
        tc = _make_tool_call(call_id="call_rt", fn_name="get_data", fn_args='{"id": 1}')
        original = ModelLlmMessage(role="assistant", tool_calls=(tc,))

        dumped = original.model_dump()
        restored = ModelLlmMessage.model_validate(dumped)

        assert restored == original
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "call_rt"
        assert restored.tool_calls[0].function.name == "get_data"

    def test_tool_message_roundtrip(self) -> None:
        """Test roundtrip for a tool message with content and tool_call_id."""
        original = ModelLlmMessage(
            role="tool",
            content='{"answer": 4}',
            tool_call_id="call_rt",
        )

        dumped = original.model_dump()
        restored = ModelLlmMessage.model_validate(dumped)

        assert restored == original
        assert restored.tool_call_id == "call_rt"
        assert restored.content == '{"answer": 4}'

    def test_model_dump_json_structure(self) -> None:
        """Test that model_dump_json produces correct JSON structure."""
        tc = _make_tool_call(call_id="call_json", fn_name="calc", fn_args='{"x": 1}')
        msg = ModelLlmMessage(
            role="assistant",
            content="Calling tool.",
            tool_calls=(tc,),
        )

        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["role"] == "assistant"
        assert parsed["content"] == "Calling tool."
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["id"] == "call_json"
        assert parsed["tool_calls"][0]["function"]["name"] == "calc"
        assert parsed["tool_call_id"] is None
