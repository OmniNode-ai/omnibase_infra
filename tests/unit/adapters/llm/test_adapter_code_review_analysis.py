# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterCodeReviewAnalysis.

Covers:
- handler_type / handler_category properties
- review() for each valid review_type (naming, docstrings, types)
- review() with empty/None LLM response (graceful fallback)
- review() rejects invalid review_type
- code_diff truncation for oversized input
- attribution fields (model_name, endpoint_url, prompt_version, etc.)
- structured_json contains review_type key
- constructor validation (max_tokens, temperature bounds)
- close() propagates to transport
- curly brace safety in code_diff (no str.format() crash)
- rendered_text is non-empty for all paths
- correlation_id parameter accepted / defaults to None
- truncation boundary: code_diff at _MAX_DIFF_CHARS + 50 fits within limit
- RuntimeHostError from LLM handler propagates out of review()
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.adapters.llm.adapter_code_review_analysis import (
    _DEFAULT_MODEL,
    _DELEGATION_CONFIDENCE,
    _MAX_DIFF_CHARS,
    _PROMPT_VERSION,
    REVIEW_TYPE_DOCSTRINGS,
    REVIEW_TYPE_NAMING,
    REVIEW_TYPE_TYPES,
    AdapterCodeReviewAnalysis,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import ProtocolConfigurationError, RuntimeHostError
from omnibase_spi.contracts.delegation.contract_delegated_response import (
    ContractDelegatedResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**kwargs: object) -> AdapterCodeReviewAnalysis:
    """Build an AdapterCodeReviewAnalysis with a mocked transport."""
    adapter = AdapterCodeReviewAnalysis(
        base_url="http://localhost:8001",
        **kwargs,  # type: ignore[arg-type]
    )
    # Replace transport and handler with mocks to avoid HTTP calls.
    adapter._transport = MagicMock()
    adapter._transport.close = AsyncMock()
    adapter._handler = AsyncMock()
    return adapter


def _make_llm_response(generated_text: str | None) -> MagicMock:
    """Build a minimal ModelLlmInferenceResponse mock."""
    resp = MagicMock()
    resp.generated_text = generated_text
    return resp


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisProperties:
    """Tests for classification properties."""

    def test_handler_type(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_type is EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_category is EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# review() -- each review type succeeds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisReviewTypes:
    """Tests for review() with each valid review_type."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "review_type",
        [REVIEW_TYPE_NAMING, REVIEW_TYPE_DOCSTRINGS, REVIEW_TYPE_TYPES],
    )
    async def test_valid_review_type_returns_contract(self, review_type: str) -> None:
        """review() with valid review_type returns ContractDelegatedResponse."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("## Review\n\nNo issues found.")
        )

        result = await adapter.review(
            review_type=review_type, code_diff="- def Foo():\n+ def foo():\n"
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "review_type",
        [REVIEW_TYPE_NAMING, REVIEW_TYPE_DOCSTRINGS, REVIEW_TYPE_TYPES],
    )
    async def test_valid_review_type_calls_llm(self, review_type: str) -> None:
        """review() with valid review_type calls the LLM handler exactly once."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        await adapter.review(review_type=review_type, code_diff="def foo(): pass")

        adapter._handler.handle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_naming_rendered_text_is_llm_output(self) -> None:
        """For naming review, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "## Naming Issues\n\nNo naming issues found."
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(expected))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING,
            code_diff="def compute(x: int) -> int: ...",
        )

        assert result.rendered_text == expected

    @pytest.mark.asyncio
    async def test_docstrings_rendered_text_is_llm_output(self) -> None:
        """For docstrings review, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "## Docstring Issues\n\nMissing docstring on `foo`."
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(expected))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_DOCSTRINGS,
            code_diff="def foo(x: int) -> int:\n    return x",
        )

        assert result.rendered_text == expected

    @pytest.mark.asyncio
    async def test_types_rendered_text_is_llm_output(self) -> None:
        """For types review, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "## Type Annotation Issues\n\nNo type annotation issues found."
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(expected))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_TYPES,
            code_diff="def api_func(x: int) -> str: ...",
        )

        assert result.rendered_text == expected


# ---------------------------------------------------------------------------
# review() -- invalid review_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisInvalidReviewType:
    """Tests for review() with invalid review_type."""

    @pytest.mark.asyncio
    async def test_invalid_review_type_raises_protocol_configuration_error(
        self,
    ) -> None:
        """review() with an unknown review_type raises ProtocolConfigurationError."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="review_type"):
            await adapter.review(
                review_type="invalid_type", code_diff="def foo(): pass"
            )

    @pytest.mark.asyncio
    async def test_invalid_review_type_does_not_call_llm(self) -> None:
        """review() raises before calling the LLM handler for invalid review_type."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        with pytest.raises(ProtocolConfigurationError):
            await adapter.review(review_type="not_valid", code_diff="def foo(): pass")

        adapter._handler.handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_review_type_raises_protocol_configuration_error(self) -> None:
        """Empty string review_type raises ProtocolConfigurationError."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError):
            await adapter.review(review_type="", code_diff="def foo(): pass")


# ---------------------------------------------------------------------------
# review() -- invalid code_diff
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisInvalidCodeDiff:
    """Tests for review() with empty or whitespace-only code_diff."""

    @pytest.mark.asyncio
    async def test_empty_code_diff_raises_protocol_configuration_error(self) -> None:
        """review() raises ProtocolConfigurationError when code_diff is an empty string."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="code_diff"):
            await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff="")

    @pytest.mark.asyncio
    async def test_whitespace_only_code_diff_raises_protocol_configuration_error(
        self,
    ) -> None:
        """review() raises ProtocolConfigurationError when code_diff is whitespace-only."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="code_diff"):
            await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff="   ")

    @pytest.mark.asyncio
    async def test_newline_only_code_diff_raises_protocol_configuration_error(
        self,
    ) -> None:
        """review() raises ProtocolConfigurationError when code_diff contains only newlines."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="code_diff"):
            await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff="\n\n\t\n")

    @pytest.mark.asyncio
    async def test_empty_code_diff_does_not_call_llm(self) -> None:
        """review() raises before calling the LLM handler for empty code_diff."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        with pytest.raises(ProtocolConfigurationError):
            await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff="")

        adapter._handler.handle.assert_not_awaited()


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisAttribution:
    """Tests for attribution fields in the response."""

    @pytest.mark.asyncio
    async def test_attribution_model_name(self) -> None:
        """attribution.model_name matches the configured model."""
        adapter = _make_adapter(model="qwen2.5-coder-7b")
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Review content.")
        )

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.model_name == "qwen2.5-coder-7b"

    @pytest.mark.asyncio
    async def test_attribution_model_name_default(self) -> None:
        """attribution.model_name defaults to _DEFAULT_MODEL when not overridden."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.model_name == _DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_attribution_endpoint_url(self) -> None:
        """attribution.endpoint_url matches the base_url passed at construction."""
        adapter = _make_adapter()
        adapter._base_url = "http://localhost:8001"
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.endpoint_url == "http://localhost:8001"

    @pytest.mark.asyncio
    async def test_attribution_prompt_version(self) -> None:
        """attribution.prompt_version matches _PROMPT_VERSION."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.prompt_version == _PROMPT_VERSION

    @pytest.mark.asyncio
    async def test_attribution_delegation_confidence(self) -> None:
        """attribution.delegation_confidence matches _DELEGATION_CONFIDENCE."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.delegation_confidence == pytest.approx(
            _DELEGATION_CONFIDENCE
        )

    @pytest.mark.asyncio
    async def test_attribution_latency_ms_is_nonnegative(self) -> None:
        """attribution.latency_ms is always >= 0."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# structured_json
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisStructuredJson:
    """Tests for structured_json in the response."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "review_type",
        [REVIEW_TYPE_NAMING, REVIEW_TYPE_DOCSTRINGS, REVIEW_TYPE_TYPES],
    )
    async def test_structured_json_contains_review_type(self, review_type: str) -> None:
        """structured_json contains the review_type key."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=review_type, code_diff="def foo(): pass"
        )

        assert result.structured_json is not None
        assert result.structured_json["review_type"] == review_type


# ---------------------------------------------------------------------------
# Empty/None LLM response (graceful fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisEmptyLlmResponse:
    """Tests for behavior when LLM returns empty or None text."""

    @pytest.mark.asyncio
    async def test_empty_llm_response_rendered_text_is_non_empty(self) -> None:
        """When LLM returns empty string, rendered_text is a non-empty fallback."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(""))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert len(result.rendered_text) > 0

    @pytest.mark.asyncio
    async def test_none_llm_response_rendered_text_is_non_empty(self) -> None:
        """When LLM returns None, rendered_text is a non-empty fallback."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert len(result.rendered_text) > 0

    @pytest.mark.asyncio
    async def test_none_llm_response_returns_contract(self) -> None:
        """When LLM returns None, result is still ContractDelegatedResponse."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_DOCSTRINGS, code_diff="class Foo: pass"
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_none_llm_response_attribution_is_set(self) -> None:
        """When LLM returns None, attribution is still fully populated."""
        adapter = _make_adapter(model="qwen2.5-coder-7b")
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert result.attribution.model_name == "qwen2.5-coder-7b"
        assert result.attribution.prompt_version == _PROMPT_VERSION


# ---------------------------------------------------------------------------
# Code diff truncation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisCodeDiffTruncation:
    """Tests for code_diff truncation when diff exceeds _MAX_DIFF_CHARS."""

    @pytest.mark.asyncio
    async def test_large_diff_truncated_in_request(self) -> None:
        """code_diff larger than _MAX_DIFF_CHARS is truncated before being sent."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        large_diff = "x" * (_MAX_DIFF_CHARS + 5_000)
        await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff=large_diff)

        assert len(captured_requests) == 1
        request = captured_requests[0]
        messages = cast("list[dict[str, Any]]", getattr(request, "messages", []))
        assert len(messages) == 1
        user_content: str = messages[0]["content"]
        # The user message must contain the truncation sentinel.
        assert "[diff truncated]" in user_content

    @pytest.mark.asyncio
    async def test_diff_within_limit_not_truncated(self) -> None:
        """code_diff within _MAX_DIFF_CHARS is not truncated."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        small_diff = "def foo(): pass"
        await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff=small_diff)

        assert len(captured_requests) == 1
        messages = cast(
            "list[dict[str, Any]]", getattr(captured_requests[0], "messages", [])
        )
        user_content: str = messages[0]["content"]
        assert "[diff truncated]" not in user_content
        # The original diff must appear verbatim.
        assert small_diff in user_content


# ---------------------------------------------------------------------------
# Curly brace safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisCurlyBraceSafety:
    """Regression tests: code_diff with curly braces must not crash review()."""

    @pytest.mark.asyncio
    async def test_json_diff_does_not_raise(self) -> None:
        """review() must not raise when code_diff contains JSON-like curly braces."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        json_diff = '+ data = {"key": "value", "nested": {"x": [1, 2, 3]}}'

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff=json_diff
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_python_dict_diff_does_not_raise(self) -> None:
        """review() must not raise when code_diff contains Python dict syntax."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        python_diff = "+ data = {'key': {1: 'one', 2: 'two'}}"

        result = await adapter.review(
            review_type=REVIEW_TYPE_TYPES, code_diff=python_diff
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_diff_placeholder_literal_does_not_raise(self) -> None:
        """review() must not raise when code_diff contains literal '{code_diff}'."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        diff_with_placeholder = "+ x = '{code_diff} is the parameter name'"

        result = await adapter.review(
            review_type=REVIEW_TYPE_DOCSTRINGS, code_diff=diff_with_placeholder
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_diff_appears_verbatim_in_user_message(self) -> None:
        """The code_diff text is forwarded literally into the user message."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Brief.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        diff = "def foo(x: dict[str, int]) -> list[str]: ..."
        await adapter.review(review_type=REVIEW_TYPE_TYPES, code_diff=diff)

        assert len(captured_requests) == 1
        messages = cast(
            "list[dict[str, Any]]", getattr(captured_requests[0], "messages", [])
        )
        user_content: str = messages[0]["content"]
        assert diff in user_content


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisConstructorValidation:
    """Tests for constructor parameter validation."""

    def test_zero_max_tokens_raises_protocol_configuration_error(self) -> None:
        """max_tokens=0 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="max_tokens"):
            AdapterCodeReviewAnalysis(
                base_url="http://localhost:8001",
                max_tokens=0,
            )

    def test_negative_max_tokens_raises_protocol_configuration_error(self) -> None:
        """Negative max_tokens raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="max_tokens"):
            AdapterCodeReviewAnalysis(
                base_url="http://localhost:8001",
                max_tokens=-1,
            )

    def test_oversized_max_tokens_raises_protocol_configuration_error(self) -> None:
        """max_tokens > 32768 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="max_tokens"):
            AdapterCodeReviewAnalysis(
                base_url="http://localhost:8001",
                max_tokens=32_769,
            )

    def test_valid_max_tokens_does_not_raise(self) -> None:
        """max_tokens=1024 is valid."""
        adapter = AdapterCodeReviewAnalysis(
            base_url="http://localhost:8001",
            max_tokens=1024,
        )
        assert adapter._max_tokens == 1024

    def test_negative_temperature_raises_protocol_configuration_error(self) -> None:
        """temperature < 0.0 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="temperature"):
            AdapterCodeReviewAnalysis(
                base_url="http://localhost:8001",
                temperature=-0.1,
            )

    def test_oversized_temperature_raises_protocol_configuration_error(self) -> None:
        """temperature > 2.0 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="temperature"):
            AdapterCodeReviewAnalysis(
                base_url="http://localhost:8001",
                temperature=2.1,
            )

    def test_valid_temperature_boundary_does_not_raise(self) -> None:
        """temperature=0.0 and temperature=2.0 are valid boundaries."""
        for t in (0.0, 2.0):
            adapter = AdapterCodeReviewAnalysis(
                base_url="http://localhost:8001",
                temperature=t,
            )
            assert adapter._temperature == t

    def test_empty_base_url_raises_protocol_configuration_error(self) -> None:
        """Explicit base_url='' raises ProtocolConfigurationError at construction time."""
        with pytest.raises(ProtocolConfigurationError, match="base_url"):
            AdapterCodeReviewAnalysis(base_url="")

    def test_empty_base_url_error_message_is_clear(self) -> None:
        """ProtocolConfigurationError for empty base_url contains a helpful message."""
        with pytest.raises(ProtocolConfigurationError, match="non-empty string"):
            AdapterCodeReviewAnalysis(base_url="")

    def test_whitespace_only_base_url_raises_protocol_configuration_error(self) -> None:
        """Whitespace-only base_url raises ProtocolConfigurationError at construction time."""
        with pytest.raises(ProtocolConfigurationError, match="base_url"):
            AdapterCodeReviewAnalysis(base_url="   ")


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    async def test_close_calls_transport_close(self) -> None:
        """close() delegates to the transport's close() method."""
        adapter = _make_adapter()
        await adapter.close()
        adapter._transport.close.assert_awaited_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Request construction verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisRequestConstruction:
    """Tests for the LLM request fields sent by review()."""

    @pytest.mark.asyncio
    async def test_request_uses_chat_completion(self) -> None:
        """The LLM request uses CHAT_COMPLETION operation type."""
        from omnibase_infra.enums import EnumLlmOperationType

        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert (
            getattr(req, "operation_type", None) is EnumLlmOperationType.CHAT_COMPLETION
        )

    @pytest.mark.asyncio
    async def test_request_model_matches_configured(self) -> None:
        """The LLM request model matches the configured model identifier."""
        adapter = _make_adapter(model="qwen2.5-coder-7b")
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert len(captured_requests) == 1
        assert getattr(captured_requests[0], "model", None) == "qwen2.5-coder-7b"

    @pytest.mark.asyncio
    async def test_request_has_system_prompt(self) -> None:
        """The LLM request includes a non-empty system prompt."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.review(
            review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
        )

        assert len(captured_requests) == 1
        system_prompt = getattr(captured_requests[0], "system_prompt", None)
        assert system_prompt and len(system_prompt) > 0


# ---------------------------------------------------------------------------
# correlation_id parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisCorrelationId:
    """Tests for the correlation_id parameter on review()."""

    @pytest.mark.asyncio
    async def test_review_accepts_correlation_id_uuid(self) -> None:
        """review() accepts a UUID correlation_id without raising."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]
        cid: UUID = uuid4()

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING,
            code_diff="def foo(): pass",
            correlation_id=cid,
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_review_works_with_correlation_id_none(self) -> None:
        """review() works when correlation_id=None (the default)."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))  # type: ignore[method-assign]

        result = await adapter.review(
            review_type=REVIEW_TYPE_NAMING,
            code_diff="def foo(): pass",
            correlation_id=None,
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_bad_review_type_with_correlation_id_raises_protocol_configuration_error(
        self,
    ) -> None:
        """Bad review_type with a correlation_id still raises ProtocolConfigurationError."""
        adapter = _make_adapter()
        cid: UUID = uuid4()

        with pytest.raises(ProtocolConfigurationError):
            await adapter.review(
                review_type="bad_type",
                code_diff="def foo(): pass",
                correlation_id=cid,
            )


# ---------------------------------------------------------------------------
# Truncation boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisTruncationBoundary:
    """Tests for the exact truncation boundary behaviour."""

    @pytest.mark.asyncio
    async def test_diff_at_limit_plus_50_fits_within_max_chars(self) -> None:
        """When code_diff length == _MAX_DIFF_CHARS + 50, the diff portion in the
        user message is <= _MAX_DIFF_CHARS characters (sentinel included).

        The adapter truncates diff_stripped to exactly _MAX_DIFF_CHARS chars
        (truncated_body + sentinel == _MAX_DIFF_CHARS). The naming template
        wraps the diff inside a markdown code fence; we extract that portion by
        splitting on the known fence delimiter ``\\n```diff\\n`` to isolate the
        substituted diff text and verify its length is within budget.
        """
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        # Build a diff that is exactly _MAX_DIFF_CHARS + 50 characters long.
        oversized_diff = "a" * (_MAX_DIFF_CHARS + 50)
        await adapter.review(review_type=REVIEW_TYPE_NAMING, code_diff=oversized_diff)

        assert len(captured_requests) == 1
        messages = cast(
            "list[dict[str, Any]]", getattr(captured_requests[0], "messages", [])
        )
        assert len(messages) == 1
        user_content: str = messages[0]["content"]

        # The sentinel must appear (truncation did happen).
        sentinel = "[diff truncated]"
        assert sentinel in user_content

        # The naming template embeds the diff inside ```diff\n...\n```.
        # Split on the opening fence to find the injected diff text.
        fence_open = "```diff\n"
        fence_close = "\n```"
        fence_start = user_content.find(fence_open)
        assert fence_start != -1, "Code fence must appear in user content"
        diff_start = fence_start + len(fence_open)
        fence_end = user_content.find(fence_close, diff_start)
        assert fence_end != -1, "Closing code fence must appear in user content"
        injected_diff = user_content[diff_start:fence_end]

        # The injected diff (truncated body + sentinel) must fit within budget.
        assert len(injected_diff) <= _MAX_DIFF_CHARS


# ---------------------------------------------------------------------------
# LLM exception propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeReviewAnalysisLlmException:
    """Tests that RuntimeHostError from the LLM handler propagates out of review()."""

    @pytest.mark.asyncio
    async def test_runtime_host_error_propagates(self) -> None:
        """When HandlerLlmOpenaiCompatible raises RuntimeHostError, it propagates."""
        adapter = _make_adapter()

        async def failing_handle(request: object) -> MagicMock:
            raise RuntimeHostError("simulated LLM failure")

        adapter._handler.handle = failing_handle  # type: ignore[method-assign,assignment]

        with pytest.raises(RuntimeHostError, match="simulated LLM failure"):
            await adapter.review(
                review_type=REVIEW_TYPE_NAMING, code_diff="def foo(): pass"
            )

    @pytest.mark.asyncio
    async def test_runtime_host_error_not_swallowed(self) -> None:
        """review() does not swallow RuntimeHostError into another exception type."""
        adapter = _make_adapter()

        async def failing_handle(request: object) -> MagicMock:
            raise RuntimeHostError("connection timeout")

        adapter._handler.handle = failing_handle  # type: ignore[method-assign,assignment]

        exc_info: pytest.ExceptionInfo[RuntimeHostError]
        with pytest.raises(RuntimeHostError) as exc_info:
            await adapter.review(
                review_type=REVIEW_TYPE_DOCSTRINGS, code_diff="class Foo: pass"
            )

        assert isinstance(exc_info.value, RuntimeHostError)

    @pytest.mark.asyncio
    async def test_runtime_host_error_type_is_exact(self) -> None:
        """The propagated exception is a RuntimeHostError instance."""
        adapter = _make_adapter()
        original_error = RuntimeHostError("network error")

        async def failing_handle(request: object) -> MagicMock:
            raise original_error

        adapter._handler.handle = failing_handle  # type: ignore[method-assign,assignment]

        with pytest.raises(RuntimeHostError) as exc_info:
            await adapter.review(
                review_type=REVIEW_TYPE_TYPES, code_diff="def api(): ..."
            )

        assert exc_info.value is original_error
