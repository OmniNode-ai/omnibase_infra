# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for LLM usage normalization service.

Tests cover the 5 fallback cases defined in OMN-2238:
    1. Complete -- all token counts present from API
    2. Partial -- some token counts missing, estimated
    3. Absent -- no usage block in response, estimated from text
    4. Streaming -- accumulated chunk deltas
    5. Missing -- no usage data and no text to estimate from

Related:
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - service_llm_usage_normalizer.py: Module under test
"""

from __future__ import annotations

from typing import Any

import pytest

from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_usage_normalizer import (
    _estimate_tokens_from_text,
    normalize_llm_usage,
    normalize_streaming_usage,
)
from omnibase_spi.contracts.measurement.enum_usage_source import (
    ContractEnumUsageSource,
)

# ---------------------------------------------------------------------------
# Token Estimation Helper Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateTokensFromText:
    """Tests for the _estimate_tokens_from_text helper."""

    def test_none_returns_zero(self) -> None:
        """None text returns 0 tokens."""
        assert _estimate_tokens_from_text(None) == 0

    def test_empty_string_returns_zero(self) -> None:
        """Empty string returns 0 tokens."""
        assert _estimate_tokens_from_text("") == 0

    def test_short_text(self) -> None:
        """Short text is estimated to at least 1 token."""
        assert _estimate_tokens_from_text("hi") >= 1

    def test_longer_text_proportional(self) -> None:
        """Longer text produces proportionally more tokens."""
        short = _estimate_tokens_from_text("hello")
        long_text = _estimate_tokens_from_text("hello " * 100)
        assert long_text > short


# ---------------------------------------------------------------------------
# Case 1: Complete Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeCompleteUsage:
    """Case 1: API response has complete usage data."""

    def test_complete_usage_from_api(self) -> None:
        """All token counts present -> source=API, is_estimated=false."""
        response: dict[str, Any] = {
            "id": "chatcmpl-abc",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        _raw, normalized = normalize_llm_usage(response)

        assert normalized.prompt_tokens == 100
        assert normalized.completion_tokens == 50
        assert normalized.total_tokens == 150
        assert normalized.source == ContractEnumUsageSource.API
        assert normalized.usage_is_estimated is False

    def test_raw_usage_provider_set(self) -> None:
        """Raw usage has provider field set."""
        response: dict[str, Any] = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        raw, _ = normalize_llm_usage(response, provider="openai_compatible")
        assert raw.provider == "openai_compatible"

    def test_raw_usage_data_redacted(self) -> None:
        """Raw usage data is the redacted response, not the original."""
        response: dict[str, Any] = {
            "messages": [{"role": "user", "content": "secret"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        raw, _ = normalize_llm_usage(response)
        # The raw_data should be redacted (no "secret" in it).
        raw_str = str(raw.raw_data)
        assert "secret" not in raw_str

    def test_zero_tokens_valid(self) -> None:
        """Zero token counts are valid API-reported values."""
        response: dict[str, Any] = {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        _, normalized = normalize_llm_usage(response)
        assert normalized.prompt_tokens == 0
        assert normalized.completion_tokens == 0
        assert normalized.total_tokens == 0
        assert normalized.source == ContractEnumUsageSource.API


# ---------------------------------------------------------------------------
# Case 2: Partial Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizePartialUsage:
    """Case 2: API response has partial usage data."""

    def test_missing_completion_tokens_estimated(self) -> None:
        """Missing completion_tokens -> estimated from generated text."""
        response: dict[str, Any] = {
            "usage": {"prompt_tokens": 50},
        }
        _, normalized = normalize_llm_usage(
            response,
            generated_text="This is a generated response with some words.",
        )

        assert normalized.prompt_tokens == 50
        assert normalized.completion_tokens > 0  # estimated
        assert normalized.source == ContractEnumUsageSource.ESTIMATED
        assert normalized.usage_is_estimated is True

    def test_missing_prompt_tokens_estimated(self) -> None:
        """Missing prompt_tokens -> estimated from prompt text."""
        response: dict[str, Any] = {
            "usage": {"completion_tokens": 25},
        }
        _, normalized = normalize_llm_usage(
            response,
            prompt_text="What is the weather in London today?",
        )

        assert normalized.prompt_tokens > 0  # estimated
        assert normalized.completion_tokens == 25
        assert normalized.source == ContractEnumUsageSource.ESTIMATED
        assert normalized.usage_is_estimated is True

    def test_partial_with_no_text_fallback(self) -> None:
        """Missing tokens with no text defaults to 0 for missing field."""
        response: dict[str, Any] = {
            "usage": {"prompt_tokens": 42},
        }
        _, normalized = normalize_llm_usage(response)

        assert normalized.prompt_tokens == 42
        assert normalized.completion_tokens == 0
        assert normalized.source == ContractEnumUsageSource.ESTIMATED
        assert normalized.usage_is_estimated is True


# ---------------------------------------------------------------------------
# Case 3: Absent Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeAbsentUsage:
    """Case 3: No usage block in the API response."""

    def test_absent_usage_estimated_from_text(self) -> None:
        """No usage block -> estimate from prompt and generated text."""
        response: dict[str, Any] = {
            "id": "chatcmpl-nousage",
            "choices": [{"message": {"content": "Hello!"}}],
        }
        _, normalized = normalize_llm_usage(
            response,
            generated_text="Hello!",
            prompt_text="Say hello",
        )

        assert normalized.prompt_tokens > 0
        assert normalized.completion_tokens > 0
        assert normalized.source == ContractEnumUsageSource.ESTIMATED
        assert normalized.usage_is_estimated is True

    def test_absent_usage_non_dict_usage_field(self) -> None:
        """Non-dict usage field treated as absent."""
        response: dict[str, Any] = {
            "usage": "not a dict",
        }
        _, normalized = normalize_llm_usage(
            response,
            generated_text="Some text",
        )

        assert normalized.source == ContractEnumUsageSource.ESTIMATED
        assert normalized.usage_is_estimated is True


# ---------------------------------------------------------------------------
# Case 4: Streaming Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeStreamingUsage:
    """Case 4: Streaming response with chunk deltas."""

    def test_streaming_with_final_usage_chunk(self) -> None:
        """Streaming with usage in final chunk -> uses API data."""
        chunks: list[dict[str, Any]] = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 3,
                    "total_tokens": 18,
                },
            },
        ]
        _, normalized = normalize_streaming_usage(
            chunks,
            generated_text="Hello world",
        )

        assert normalized.prompt_tokens == 15
        assert normalized.completion_tokens == 3
        assert normalized.source == ContractEnumUsageSource.API
        assert normalized.usage_is_estimated is False

    def test_streaming_without_usage_estimated(self) -> None:
        """Streaming without usage in any chunk -> estimated from text."""
        chunks: list[dict[str, Any]] = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        raw, normalized = normalize_streaming_usage(
            chunks,
            generated_text="Hello world",
            prompt_text="Say hello",
        )

        assert normalized.source == ContractEnumUsageSource.ESTIMATED
        assert normalized.usage_is_estimated is True
        assert normalized.prompt_tokens > 0
        assert normalized.completion_tokens > 0

        # Raw data should indicate streaming mode.
        assert raw.raw_data["streaming"] is True
        assert raw.raw_data["chunk_count"] == 2

    def test_streaming_without_usage_or_text(self) -> None:
        """Streaming without usage or text -> MISSING source."""
        chunks: list[dict[str, Any]] = [
            {"choices": [{"delta": {}}]},
        ]
        _, normalized = normalize_streaming_usage(chunks)

        assert normalized.source == ContractEnumUsageSource.MISSING
        assert normalized.prompt_tokens == 0
        assert normalized.completion_tokens == 0


# ---------------------------------------------------------------------------
# Case 5: Missing Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeMissingUsage:
    """Case 5: No usage data and no text for estimation."""

    def test_missing_everything(self) -> None:
        """No usage block, no generated text, no prompt -> MISSING."""
        response: dict[str, Any] = {
            "id": "chatcmpl-nothing",
        }
        _, normalized = normalize_llm_usage(response)

        assert normalized.prompt_tokens == 0
        assert normalized.completion_tokens == 0
        assert normalized.total_tokens == 0
        assert normalized.source == ContractEnumUsageSource.MISSING
        assert normalized.usage_is_estimated is False

    def test_missing_with_empty_text(self) -> None:
        """No usage block with empty text -> MISSING."""
        response: dict[str, Any] = {}
        _, normalized = normalize_llm_usage(
            response,
            generated_text="",
            prompt_text="",
        )

        assert normalized.source == ContractEnumUsageSource.MISSING

    def test_non_dict_response(self) -> None:
        """Non-dict response input -> MISSING."""
        _, normalized = normalize_llm_usage(
            "not a dict",  # type: ignore[arg-type]
        )
        assert normalized.source == ContractEnumUsageSource.MISSING


# ---------------------------------------------------------------------------
# Tool Call Token Counting Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolCallTokenCounting:
    """Test that tool call tokens are counted as completion_tokens."""

    def test_tool_calls_counted_as_completion(self) -> None:
        """Tool call tokens from usage block go to completion_tokens."""
        response: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
            },
        }
        _, normalized = normalize_llm_usage(response)

        # Tool call tokens are in completion_tokens per OpenAI convention.
        assert normalized.completion_tokens == 15
        assert normalized.prompt_tokens == 20
        assert normalized.source == ContractEnumUsageSource.API
