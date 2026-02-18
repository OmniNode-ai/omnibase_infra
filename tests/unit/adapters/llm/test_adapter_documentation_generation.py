# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterDocumentationGeneration.

Covers:
- handler_type / handler_category properties
- generate() for each valid task_type (docstring, readme, api_doc)
- generate() with empty/None LLM response (graceful fallback)
- generate() rejects invalid task_type
- source truncation for oversized input
- attribution fields (model_name, endpoint_url, prompt_version, etc.)
- structured_json contains task_type key
- constructor validation (max_tokens, temperature bounds)
- close() propagates to transport
- curly brace safety in source text (no str.format() crash)
- rendered_text is non-empty for all paths
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.adapters.llm.adapter_documentation_generation import (
    _DEFAULT_MODEL,
    _DELEGATION_CONFIDENCE,
    _MAX_SOURCE_CHARS,
    _PROMPT_VERSION,
    TASK_TYPE_API_DOC,
    TASK_TYPE_DOCSTRING,
    TASK_TYPE_README,
    AdapterDocumentationGeneration,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_spi.contracts.delegation.contract_delegated_response import (
    ContractDelegatedResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**kwargs: object) -> AdapterDocumentationGeneration:
    """Build an AdapterDocumentationGeneration with a mocked transport."""
    adapter = AdapterDocumentationGeneration(
        base_url="http://localhost:8100",
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
class TestAdapterDocumentationGenerationProperties:
    """Tests for classification properties."""

    def test_handler_type(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_type is EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_category is EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# generate() -- each task type succeeds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationTaskTypes:
    """Tests for generate() with each valid task_type."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_type",
        [TASK_TYPE_DOCSTRING, TASK_TYPE_README, TASK_TYPE_API_DOC],
    )
    async def test_valid_task_type_returns_contract(self, task_type: str) -> None:
        """generate() with valid task_type returns ContractDelegatedResponse."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("## Generated Doc\n\nSome content.")
        )

        result = await adapter.generate(task_type=task_type, source="def foo(): pass")

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_type",
        [TASK_TYPE_DOCSTRING, TASK_TYPE_README, TASK_TYPE_API_DOC],
    )
    async def test_valid_task_type_calls_llm(self, task_type: str) -> None:
        """generate() with valid task_type calls the LLM handler exactly once."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        await adapter.generate(task_type=task_type, source="class Foo: pass")

        adapter._handler.handle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_docstring_rendered_text_is_llm_output(self) -> None:
        """For docstring task, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "Args:\n    x: The input.\n\nReturns:\n    int."
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(expected))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING,
            source="def compute(x: int) -> int: ...",
        )

        assert result.rendered_text == expected

    @pytest.mark.asyncio
    async def test_readme_rendered_text_is_llm_output(self) -> None:
        """For readme task, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "## Overview\n\nThis module does X."
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(expected))

        result = await adapter.generate(
            task_type=TASK_TYPE_README,
            source="Module description here.",
        )

        assert result.rendered_text == expected

    @pytest.mark.asyncio
    async def test_api_doc_rendered_text_is_llm_output(self) -> None:
        """For api_doc task, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "## Parameters\n\n- `x` (int): Input value."
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(expected))

        result = await adapter.generate(
            task_type=TASK_TYPE_API_DOC,
            source="def api_func(x: int) -> str: ...",
        )

        assert result.rendered_text == expected


# ---------------------------------------------------------------------------
# generate() -- invalid task_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationInvalidTaskType:
    """Tests for generate() with invalid task_type."""

    @pytest.mark.asyncio
    async def test_invalid_task_type_raises_value_error(self) -> None:
        """generate() with an unknown task_type raises ValueError."""
        adapter = _make_adapter()

        with pytest.raises(ValueError, match="task_type"):
            await adapter.generate(task_type="invalid_type", source="def foo(): pass")

    @pytest.mark.asyncio
    async def test_invalid_task_type_does_not_call_llm(self) -> None:
        """generate() raises before calling the LLM handler for invalid task_type."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        with pytest.raises(ValueError):
            await adapter.generate(task_type="not_valid", source="def foo(): pass")

        adapter._handler.handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_task_type_raises_value_error(self) -> None:
        """Empty string task_type raises ValueError."""
        adapter = _make_adapter()

        with pytest.raises(ValueError):
            await adapter.generate(task_type="", source="def foo(): pass")


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationAttribution:
    """Tests for attribution fields in the response."""

    @pytest.mark.asyncio
    async def test_attribution_model_name(self) -> None:
        """attribution.model_name matches the configured model."""
        adapter = _make_adapter(model="qwen2.5-72b")
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("Docstring content.")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.model_name == "qwen2.5-72b"

    @pytest.mark.asyncio
    async def test_attribution_model_name_default(self) -> None:
        """attribution.model_name defaults to _DEFAULT_MODEL when not overridden."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.model_name == _DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_attribution_endpoint_url(self) -> None:
        """attribution.endpoint_url matches the base_url passed at construction."""
        adapter = _make_adapter()
        adapter._base_url = "http://localhost:8100"
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.endpoint_url == "http://localhost:8100"

    @pytest.mark.asyncio
    async def test_attribution_prompt_version(self) -> None:
        """attribution.prompt_version matches _PROMPT_VERSION."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.prompt_version == _PROMPT_VERSION

    @pytest.mark.asyncio
    async def test_attribution_delegation_confidence(self) -> None:
        """attribution.delegation_confidence matches _DELEGATION_CONFIDENCE."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.delegation_confidence == pytest.approx(
            _DELEGATION_CONFIDENCE
        )

    @pytest.mark.asyncio
    async def test_attribution_latency_ms_is_nonnegative(self) -> None:
        """attribution.latency_ms is always >= 0."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# structured_json
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationStructuredJson:
    """Tests for structured_json in the response."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_type",
        [TASK_TYPE_DOCSTRING, TASK_TYPE_README, TASK_TYPE_API_DOC],
    )
    async def test_structured_json_contains_task_type(self, task_type: str) -> None:
        """structured_json contains the task_type key."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        result = await adapter.generate(task_type=task_type, source="def foo(): pass")

        assert result.structured_json is not None
        assert result.structured_json["task_type"] == task_type


# ---------------------------------------------------------------------------
# Empty/None LLM response (graceful fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationEmptyLlmResponse:
    """Tests for behavior when LLM returns empty or None text."""

    @pytest.mark.asyncio
    async def test_empty_llm_response_rendered_text_is_non_empty(self) -> None:
        """When LLM returns empty string, rendered_text is a non-empty fallback."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(""))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert len(result.rendered_text) > 0

    @pytest.mark.asyncio
    async def test_none_llm_response_rendered_text_is_non_empty(self) -> None:
        """When LLM returns None, rendered_text is a non-empty fallback."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert len(result.rendered_text) > 0

    @pytest.mark.asyncio
    async def test_none_llm_response_returns_contract(self) -> None:
        """When LLM returns None, result is still ContractDelegatedResponse."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))

        result = await adapter.generate(
            task_type=TASK_TYPE_README, source="Module description."
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_none_llm_response_attribution_is_set(self) -> None:
        """When LLM returns None, attribution is still fully populated."""
        adapter = _make_adapter(model="qwen2.5-72b")
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass"
        )

        assert result.attribution.model_name == "qwen2.5-72b"
        assert result.attribution.prompt_version == _PROMPT_VERSION


# ---------------------------------------------------------------------------
# Source truncation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationSourceTruncation:
    """Tests for source truncation when source exceeds _MAX_SOURCE_CHARS."""

    @pytest.mark.asyncio
    async def test_large_source_truncated_in_request(self) -> None:
        """Source larger than _MAX_SOURCE_CHARS is truncated before being sent."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign]

        large_source = "x" * (_MAX_SOURCE_CHARS + 5_000)
        await adapter.generate(task_type=TASK_TYPE_DOCSTRING, source=large_source)

        assert len(captured_requests) == 1
        request = captured_requests[0]
        messages = getattr(request, "messages", ())
        assert len(messages) == 1
        user_content: str = messages[0]["content"]
        # The user message must contain the truncation sentinel.
        assert "[source truncated]" in user_content

    @pytest.mark.asyncio
    async def test_source_within_limit_not_truncated(self) -> None:
        """Source within _MAX_SOURCE_CHARS is not truncated."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign]

        small_source = "def foo(): pass"
        await adapter.generate(task_type=TASK_TYPE_DOCSTRING, source=small_source)

        assert len(captured_requests) == 1
        messages = getattr(captured_requests[0], "messages", ())
        user_content: str = messages[0]["content"]
        assert "[source truncated]" not in user_content
        # The original source must appear verbatim.
        assert small_source in user_content


# ---------------------------------------------------------------------------
# Curly brace safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationCurlyBraceSafety:
    """Regression tests: source with curly braces must not crash generate()."""

    @pytest.mark.asyncio
    async def test_json_source_does_not_raise(self) -> None:
        """generate() must not raise when source contains JSON-like curly braces."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        json_source = '{"key": "value", "nested": {"x": [1, 2, 3]}}'

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source=json_source
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_python_dict_source_does_not_raise(self) -> None:
        """generate() must not raise when source contains Python dict syntax."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        python_source = "data = {'key': {1: 'one', 2: 'two'}}"

        result = await adapter.generate(
            task_type=TASK_TYPE_README, source=python_source
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_source_placeholder_literal_does_not_raise(self) -> None:
        """generate() must not raise when source contains literal '{source}'."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response("Content."))

        source_with_placeholder = "The {source} parameter is used here."

        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING, source=source_with_placeholder
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_source_appears_verbatim_in_user_message(self) -> None:
        """The source text is forwarded literally into the user message."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Brief.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign]

        source = "def foo(x: dict[str, int]) -> list[str]: ..."
        await adapter.generate(task_type=TASK_TYPE_DOCSTRING, source=source)

        assert len(captured_requests) == 1
        messages = getattr(captured_requests[0], "messages", ())
        user_content: str = messages[0]["content"]
        assert source in user_content


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationConstructorValidation:
    """Tests for constructor parameter validation."""

    def test_zero_max_tokens_raises_value_error(self) -> None:
        """max_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens"):
            AdapterDocumentationGeneration(
                base_url="http://localhost:8100",
                max_tokens=0,
            )

    def test_negative_max_tokens_raises_value_error(self) -> None:
        """Negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens"):
            AdapterDocumentationGeneration(
                base_url="http://localhost:8100",
                max_tokens=-1,
            )

    def test_oversized_max_tokens_raises_value_error(self) -> None:
        """max_tokens > 32768 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens"):
            AdapterDocumentationGeneration(
                base_url="http://localhost:8100",
                max_tokens=32_769,
            )

    def test_valid_max_tokens_does_not_raise(self) -> None:
        """max_tokens=1024 is valid."""
        adapter = AdapterDocumentationGeneration(
            base_url="http://localhost:8100",
            max_tokens=1024,
        )
        assert adapter._max_tokens == 1024

    def test_negative_temperature_raises_value_error(self) -> None:
        """temperature < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            AdapterDocumentationGeneration(
                base_url="http://localhost:8100",
                temperature=-0.1,
            )

    def test_oversized_temperature_raises_value_error(self) -> None:
        """temperature > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            AdapterDocumentationGeneration(
                base_url="http://localhost:8100",
                temperature=2.1,
            )

    def test_valid_temperature_boundary_does_not_raise(self) -> None:
        """temperature=0.0 and temperature=2.0 are valid boundaries."""
        for t in (0.0, 2.0):
            adapter = AdapterDocumentationGeneration(
                base_url="http://localhost:8100",
                temperature=t,
            )
            assert adapter._temperature == t


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterDocumentationGenerationClose:
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
class TestAdapterDocumentationGenerationRequestConstruction:
    """Tests for the LLM request fields sent by generate()."""

    @pytest.mark.asyncio
    async def test_request_uses_chat_completion(self) -> None:
        """The LLM request uses CHAT_COMPLETION operation type."""
        from omnibase_infra.enums import EnumLlmOperationType

        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign]

        await adapter.generate(task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass")

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert (
            getattr(req, "operation_type", None) is EnumLlmOperationType.CHAT_COMPLETION
        )

    @pytest.mark.asyncio
    async def test_request_model_matches_configured(self) -> None:
        """The LLM request model matches the configured model identifier."""
        adapter = _make_adapter(model="qwen2.5-72b")
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign]

        await adapter.generate(task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass")

        assert len(captured_requests) == 1
        assert getattr(captured_requests[0], "model", None) == "qwen2.5-72b"

    @pytest.mark.asyncio
    async def test_request_has_system_prompt(self) -> None:
        """The LLM request includes a non-empty system prompt."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Content.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign]

        await adapter.generate(task_type=TASK_TYPE_DOCSTRING, source="def foo(): pass")

        assert len(captured_requests) == 1
        system_prompt = getattr(captured_requests[0], "system_prompt", None)
        assert system_prompt and len(system_prompt) > 0
