# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for LLM response redaction utilities.

Tests cover:
    - Message content redaction for system/user/tool roles
    - Assistant message preservation
    - Tool call argument stripping (names + keys preserved)
    - Size cap enforcement with truncation metadata
    - Edge cases (empty responses, non-dict inputs, missing fields)

Related:
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - util_llm_response_redaction.py: Module under test
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from omnibase_infra.utils.util_llm_response_redaction import (
    MAX_RAW_BLOB_BYTES,
    _enforce_size_cap,
    _redact_messages,
    _redact_tool_calls,
    _sha256_of,
    redact_llm_response,
)

# ---------------------------------------------------------------------------
# SHA-256 Helper Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSha256Of:
    """Tests for the _sha256_of helper."""

    def test_deterministic(self) -> None:
        """Same input produces same hash."""
        assert _sha256_of("hello") == _sha256_of("hello")

    def test_different_inputs_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        assert _sha256_of("hello") != _sha256_of("world")

    def test_prefix(self) -> None:
        """Hash is prefixed with sha256:."""
        result = _sha256_of("test")
        assert result.startswith("sha256:")
        # SHA-256 hex digest is 64 characters.
        assert len(result) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# Message Redaction Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRedactMessages:
    """Tests for _redact_messages."""

    def test_system_message_content_hashed(self) -> None:
        """System message content is replaced with SHA-256 hash."""
        messages = [{"role": "system", "content": "You are a secret agent"}]
        result = _redact_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"].startswith("sha256:")
        assert "secret agent" not in result[0]["content"]

    def test_user_message_content_hashed(self) -> None:
        """User message content is replaced with SHA-256 hash."""
        messages = [{"role": "user", "content": "My password is hunter2"}]
        result = _redact_messages(messages)
        assert result[0]["content"].startswith("sha256:")
        assert "hunter2" not in result[0]["content"]

    def test_assistant_message_preserved(self) -> None:
        """Assistant message content is preserved as-is."""
        messages = [{"role": "assistant", "content": "Hello, how can I help?"}]
        result = _redact_messages(messages)
        assert result[0]["content"] == "Hello, how can I help?"

    def test_tool_message_content_hashed(self) -> None:
        """Tool role message content is hashed (may contain sensitive output)."""
        messages = [{"role": "tool", "content": '{"api_key": "sk-secret"}'}]
        result = _redact_messages(messages)
        assert result[0]["content"].startswith("sha256:")

    def test_none_content_preserved(self) -> None:
        """None content is not modified."""
        messages = [{"role": "user", "content": None}]
        result = _redact_messages(messages)
        assert result[0]["content"] is None

    def test_mixed_roles(self) -> None:
        """Multiple roles are handled correctly in order."""
        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user query"},
            {"role": "assistant", "content": "model response"},
            {"role": "tool", "content": "tool output"},
        ]
        result = _redact_messages(messages)
        assert len(result) == 4
        assert result[0]["content"].startswith("sha256:")  # system
        assert result[1]["content"].startswith("sha256:")  # user
        assert result[2]["content"] == "model response"  # assistant
        assert result[3]["content"].startswith("sha256:")  # tool

    def test_non_dict_entries_skipped(self) -> None:
        """Non-dict entries in the message list are skipped."""
        messages: list[Any] = [
            "not a dict",
            {"role": "user", "content": "valid"},
        ]
        result = _redact_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_list(self) -> None:
        """Empty message list returns empty list."""
        assert _redact_messages([]) == []


# ---------------------------------------------------------------------------
# Tool Call Redaction Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRedactToolCalls:
    """Tests for _redact_tool_calls."""

    def test_argument_values_stripped(self) -> None:
        """Tool call argument values are replaced with type placeholders."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London", "units": "metric"}',
                },
            },
        ]
        result = _redact_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        args = json.loads(result[0]["function"]["arguments"])
        assert "city" in args
        assert args["city"] == "<str>"
        assert args["units"] == "<str>"
        assert "London" not in str(result)

    def test_function_name_preserved(self) -> None:
        """Function name is always preserved."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "search_database",
                    "arguments": '{"query": "secret data"}',
                },
            },
        ]
        result = _redact_tool_calls(tool_calls)
        assert result[0]["function"]["name"] == "search_database"

    def test_invalid_json_arguments_redacted(self) -> None:
        """Non-JSON arguments string is replaced with <redacted>."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "fn",
                    "arguments": "not valid json",
                },
            },
        ]
        result = _redact_tool_calls(tool_calls)
        assert result[0]["function"]["arguments"] == "<redacted>"

    def test_empty_arguments(self) -> None:
        """Empty arguments string is not modified."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "fn",
                    "arguments": "",
                },
            },
        ]
        result = _redact_tool_calls(tool_calls)
        assert result[0]["function"]["arguments"] == ""

    def test_non_dict_entries_skipped(self) -> None:
        """Non-dict entries in tool_calls are skipped."""
        tool_calls: list[Any] = ["not_a_dict", 42]
        result = _redact_tool_calls(tool_calls)
        assert result == []

    def test_numeric_argument_values(self) -> None:
        """Numeric argument values show type placeholder."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "set_temp",
                    "arguments": '{"temperature": 72, "threshold": 0.5}',
                },
            },
        ]
        result = _redact_tool_calls(tool_calls)
        args = json.loads(result[0]["function"]["arguments"])
        assert args["temperature"] == "<int>"
        assert args["threshold"] == "<float>"


# ---------------------------------------------------------------------------
# Full Response Redaction Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRedactLlmResponse:
    """Tests for redact_llm_response."""

    def test_basic_chat_response_redacted(self) -> None:
        """Standard chat response has choices preserved, safe fields kept."""
        response = {
            "id": "chatcmpl-abc",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = redact_llm_response(response)
        assert result["id"] == "chatcmpl-abc"
        assert result["model"] == "gpt-4"
        assert result["usage"]["prompt_tokens"] == 10

    def test_user_content_in_choices_redacted(self) -> None:
        """User content in choice messages is hashed."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "user",
                        "content": "What is my SSN?",
                    },
                },
            ],
        }
        result = redact_llm_response(response)
        msg = result["choices"][0]["message"]
        assert msg["content"].startswith("sha256:")
        assert "SSN" not in str(result)

    def test_tool_calls_in_choices_redacted(self) -> None:
        """Tool call arguments in choice messages are redacted."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "confidential"}',
                                },
                            },
                        ],
                    },
                },
            ],
        }
        result = redact_llm_response(response)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert "confidential" not in str(result)

    def test_top_level_messages_redacted(self) -> None:
        """Top-level messages array (request echo) is redacted."""
        response = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Secret prompt"},
            ],
            "choices": [],
        }
        result = redact_llm_response(response)
        assert result["messages"][0]["content"].startswith("sha256:")
        assert result["messages"][1]["content"].startswith("sha256:")
        assert "Secret prompt" not in str(result)

    def test_non_dict_input_returns_empty(self) -> None:
        """Non-dict input returns empty dict."""
        assert redact_llm_response("not a dict") == {}  # type: ignore[arg-type]
        assert redact_llm_response(None) == {}  # type: ignore[arg-type]

    def test_empty_dict_returns_empty(self) -> None:
        """Empty dict is returned as-is."""
        assert redact_llm_response({}) == {}


# ---------------------------------------------------------------------------
# Size Cap Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnforceSizeCap:
    """Tests for _enforce_size_cap."""

    def test_small_response_unchanged(self) -> None:
        """Response under size limit is returned unchanged."""
        data = {"model": "gpt-4", "usage": {"prompt_tokens": 10}}
        result = _enforce_size_cap(data)
        assert result == data

    def test_oversized_response_truncated(self) -> None:
        """Response over size limit produces truncation marker."""
        # Create a response that exceeds 64KB.
        large_data = {"content": "x" * 100_000}
        result = _enforce_size_cap(large_data, max_bytes=1000)
        assert result["truncated"] is True
        assert "original_size_bytes" in result
        assert result["original_size_bytes"] > 1000
        assert result["content_hash"].startswith("sha256:")

    def test_truncated_preserves_usage(self) -> None:
        """Truncation marker preserves the usage block."""
        data = {
            "content": "x" * 100_000,
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }
        result = _enforce_size_cap(data, max_bytes=1000)
        assert result["truncated"] is True
        assert result["usage"]["prompt_tokens"] == 42

    def test_truncated_preserves_model_and_id(self) -> None:
        """Truncation marker preserves model and id fields."""
        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "content": "x" * 100_000,
        }
        result = _enforce_size_cap(data, max_bytes=1000)
        assert result["model"] == "gpt-4"
        assert result["id"] == "chatcmpl-123"

    def test_exact_limit_not_truncated(self) -> None:
        """Response exactly at the size limit is not truncated."""
        data = {"a": "b"}
        serialized = json.dumps(data)
        result = _enforce_size_cap(data, max_bytes=len(serialized.encode("utf-8")))
        assert "truncated" not in result
