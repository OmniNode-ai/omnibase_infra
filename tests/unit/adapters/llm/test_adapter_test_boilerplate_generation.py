# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterTestBoilerplateGeneration.

Covers:
- handler_type / handler_category properties
- generate() for each valid task_type (test_module, test_class, test_function)
- generate() with empty/None LLM response (graceful fallback)
- generate() rejects invalid task_type
- generate() rejects empty/whitespace-only source
- source truncation for oversized input
- attribution fields (model_name, endpoint_url, prompt_version, etc.)
- structured_json contains task_type key
- constructor validation (max_tokens, temperature, base_url bounds)
- close() propagates to transport
- curly brace safety in source text (no str.format() crash)
- rendered_text is non-empty for all paths
- correlation_id parameter accepted / defaults to None
- truncation boundary: source at _MAX_SOURCE_CHARS + 50 fits within limit
- RuntimeHostError from LLM handler propagates out of generate()
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.adapters.llm.adapter_test_boilerplate_generation import (
    _DEFAULT_MODEL,
    _DELEGATION_CONFIDENCE,
    _MAX_SOURCE_CHARS,
    _PROMPT_VERSION,
    TASK_TYPE_TEST_CLASS,
    TASK_TYPE_TEST_FUNCTION,
    TASK_TYPE_TEST_MODULE,
    AdapterTestBoilerplateGeneration,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLlmOperationType,
)
from omnibase_infra.errors import ProtocolConfigurationError, RuntimeHostError
from omnibase_spi.contracts.delegation.contract_delegated_response import (
    ContractDelegatedResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 2_048,
    temperature: float = 0.1,
    api_key: str | None = None,
) -> AdapterTestBoilerplateGeneration:
    """Build an AdapterTestBoilerplateGeneration with a mocked transport."""
    adapter = AdapterTestBoilerplateGeneration(
        base_url="http://localhost:8001",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
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
class TestAdapterTestBoilerplateGenerationProperties:
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
class TestAdapterTestBoilerplateGenerationTaskTypes:
    """Tests for generate() with each valid task_type."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_type",
        [TASK_TYPE_TEST_MODULE, TASK_TYPE_TEST_CLASS, TASK_TYPE_TEST_FUNCTION],
    )
    async def test_valid_task_type_returns_contract(self, task_type: str) -> None:
        """generate() with valid task_type returns ContractDelegatedResponse."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): assert True")
        )

        result = await adapter.generate(task_type=task_type, source="def foo(): pass")

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_type",
        [TASK_TYPE_TEST_MODULE, TASK_TYPE_TEST_CLASS, TASK_TYPE_TEST_FUNCTION],
    )
    async def test_valid_task_type_calls_llm(self, task_type: str) -> None:
        """generate() with valid task_type calls the LLM handler exactly once."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_bar(): pass")
        )

        await adapter.generate(task_type=task_type, source="class Foo: pass")

        adapter._handler.handle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_test_module_rendered_text_is_llm_output(self) -> None:
        """For test_module task, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = (
            "from __future__ import annotations\n\nimport pytest\n\n"
            "@pytest.mark.unit\nclass TestFoo:\n    def test_run(self): assert True"
        )
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(expected)
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE,
            source="class Foo:\n    def run(self) -> bool: ...",
        )

        assert result.rendered_text == expected

    @pytest.mark.asyncio
    async def test_test_class_rendered_text_is_llm_output(self) -> None:
        """For test_class task, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = "@pytest.mark.unit\nclass TestBar:\n    def test_init(self): ..."
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(expected)
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_CLASS,
            source="class Bar:\n    def __init__(self, x: int) -> None: ...",
        )

        assert result.rendered_text == expected

    @pytest.mark.asyncio
    async def test_test_function_rendered_text_is_llm_output(self) -> None:
        """For test_function task, rendered_text equals the LLM generated_text."""
        adapter = _make_adapter()
        expected = (
            "@pytest.mark.unit\ndef test_compute_returns_int(): assert compute(1) == 1"
        )
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(expected)
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_FUNCTION,
            source="def compute(x: int) -> int: return x",
        )

        assert result.rendered_text == expected


# ---------------------------------------------------------------------------
# generate() -- invalid task_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationInvalidTaskType:
    """Tests for generate() with invalid task_type."""

    @pytest.mark.asyncio
    async def test_invalid_task_type_raises_protocol_configuration_error(
        self,
    ) -> None:
        """generate() with an unknown task_type raises ProtocolConfigurationError."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="task_type"):
            await adapter.generate(task_type="invalid_type", source="def foo(): pass")

    @pytest.mark.asyncio
    async def test_invalid_task_type_does_not_call_llm(self) -> None:
        """generate() raises before calling the LLM handler for invalid task_type."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Content.")
        )

        with pytest.raises(ProtocolConfigurationError):
            await adapter.generate(task_type="not_valid", source="def foo(): pass")

        adapter._handler.handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_task_type_raises_protocol_configuration_error(self) -> None:
        """Empty string task_type raises ProtocolConfigurationError."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError):
            await adapter.generate(task_type="", source="def foo(): pass")

    @pytest.mark.asyncio
    async def test_docstring_task_type_is_invalid(self) -> None:
        """'docstring' is not a valid task_type for test boilerplate generation."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="task_type"):
            await adapter.generate(task_type="docstring", source="def foo(): pass")


# ---------------------------------------------------------------------------
# generate() -- invalid source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationInvalidSource:
    """Tests for generate() with empty or whitespace-only source."""

    @pytest.mark.asyncio
    async def test_empty_source_raises_protocol_configuration_error(self) -> None:
        """generate() raises ProtocolConfigurationError when source is an empty string."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="source"):
            await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source="")

    @pytest.mark.asyncio
    async def test_whitespace_only_source_raises_protocol_configuration_error(
        self,
    ) -> None:
        """generate() raises ProtocolConfigurationError when source is whitespace-only."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="source"):
            await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source="   ")

    @pytest.mark.asyncio
    async def test_newline_only_source_raises_protocol_configuration_error(
        self,
    ) -> None:
        """generate() raises ProtocolConfigurationError when source contains only newlines."""
        adapter = _make_adapter()

        with pytest.raises(ProtocolConfigurationError, match="source"):
            await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source="\n\n\t\n")

    @pytest.mark.asyncio
    async def test_empty_source_does_not_call_llm(self) -> None:
        """generate() raises before calling the LLM handler for empty source."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Content.")
        )

        with pytest.raises(ProtocolConfigurationError):
            await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source="")

        adapter._handler.handle.assert_not_awaited()


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationAttribution:
    """Tests for attribution fields in the response."""

    @pytest.mark.asyncio
    async def test_attribution_model_name(self) -> None:
        """attribution.model_name matches the configured model."""
        adapter = _make_adapter(model="qwen3-14b-custom")
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.model_name == "qwen3-14b-custom"

    @pytest.mark.asyncio
    async def test_attribution_model_name_default(self) -> None:
        """attribution.model_name defaults to _DEFAULT_MODEL when not overridden."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.model_name == _DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_attribution_endpoint_url(self) -> None:
        """attribution.endpoint_url matches the base_url passed at construction."""
        adapter = _make_adapter()
        adapter._base_url = "http://localhost:8001"
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.endpoint_url == "http://localhost:8001"

    @pytest.mark.asyncio
    async def test_attribution_prompt_version(self) -> None:
        """attribution.prompt_version matches _PROMPT_VERSION."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.prompt_version == _PROMPT_VERSION

    @pytest.mark.asyncio
    async def test_attribution_delegation_confidence(self) -> None:
        """attribution.delegation_confidence matches _DELEGATION_CONFIDENCE."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.delegation_confidence == pytest.approx(
            _DELEGATION_CONFIDENCE
        )

    @pytest.mark.asyncio
    async def test_attribution_latency_ms_is_nonnegative(self) -> None:
        """attribution.latency_ms is always >= 0."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# structured_json
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationStructuredJson:
    """Tests for structured_json in the response."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task_type",
        [TASK_TYPE_TEST_MODULE, TASK_TYPE_TEST_CLASS, TASK_TYPE_TEST_FUNCTION],
    )
    async def test_structured_json_contains_task_type(self, task_type: str) -> None:
        """structured_json contains the task_type key."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(task_type=task_type, source="def foo(): pass")

        assert result.structured_json is not None
        assert result.structured_json["task_type"] == task_type


# ---------------------------------------------------------------------------
# Empty/None LLM response (graceful fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationEmptyLlmResponse:
    """Tests for behavior when LLM returns empty or None text."""

    @pytest.mark.asyncio
    async def test_empty_llm_response_rendered_text_is_non_empty(self) -> None:
        """When LLM returns empty string, rendered_text is a non-empty fallback."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert len(result.rendered_text) > 0

    @pytest.mark.asyncio
    async def test_none_llm_response_rendered_text_is_non_empty(self) -> None:
        """When LLM returns None, rendered_text is a non-empty fallback."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(None)
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert len(result.rendered_text) > 0

    @pytest.mark.asyncio
    async def test_none_llm_response_returns_contract(self) -> None:
        """When LLM returns None, result is still ContractDelegatedResponse."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(None)
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_CLASS, source="class Foo: pass"
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_none_llm_response_attribution_is_set(self) -> None:
        """When LLM returns None, attribution is still fully populated."""
        adapter = _make_adapter(model="qwen3-14b-custom")
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(None)
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert result.attribution.model_name == "qwen3-14b-custom"
        assert result.attribution.prompt_version == _PROMPT_VERSION

    @pytest.mark.asyncio
    async def test_empty_response_fallback_contains_comment(self) -> None:
        """Fallback rendered_text for empty LLM response contains a Python comment."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_FUNCTION, source="def compute(x: int) -> int: ..."
        )

        # Fallback starts with a comment character.
        assert result.rendered_text.startswith("#")


# ---------------------------------------------------------------------------
# Source truncation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationSourceTruncation:
    """Tests for source truncation when source exceeds _MAX_SOURCE_CHARS."""

    @pytest.mark.asyncio
    async def test_large_source_truncated_in_request(self) -> None:
        """Source larger than _MAX_SOURCE_CHARS is truncated before being sent."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        large_source = "x" * (_MAX_SOURCE_CHARS + 5_000)
        await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source=large_source)

        assert len(captured_requests) == 1
        request = captured_requests[0]
        messages = cast("list[dict[str, Any]]", getattr(request, "messages", []))
        assert len(messages) == 1
        user_content: str = messages[0]["content"]
        assert "[source truncated]" in user_content

    @pytest.mark.asyncio
    async def test_source_within_limit_not_truncated(self) -> None:
        """Source within _MAX_SOURCE_CHARS is not truncated."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        small_source = "def foo(): pass"
        await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source=small_source)

        assert len(captured_requests) == 1
        messages = cast(
            "list[dict[str, Any]]", getattr(captured_requests[0], "messages", [])
        )
        user_content: str = messages[0]["content"]
        assert "[source truncated]" not in user_content
        assert small_source in user_content


# ---------------------------------------------------------------------------
# Curly brace safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationCurlyBraceSafety:
    """Regression tests: source with curly braces must not crash generate()."""

    @pytest.mark.asyncio
    async def test_json_source_does_not_raise(self) -> None:
        """generate() must not raise when source contains JSON-like curly braces."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        json_source = '{"key": "value", "nested": {"x": [1, 2, 3]}}'

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source=json_source
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_python_dict_source_does_not_raise(self) -> None:
        """generate() must not raise when source contains Python dict syntax."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        python_source = "data = {'key': {1: 'one', 2: 'two'}}"

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_CLASS, source=python_source
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_source_placeholder_literal_does_not_raise(self) -> None:
        """generate() must not raise when source contains literal '{source}'."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        source_with_placeholder = "The {source} parameter is injected here."

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_FUNCTION, source=source_with_placeholder
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_source_appears_verbatim_in_user_message(self) -> None:
        """The source text is forwarded literally into the user message."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("Brief test stub.")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        source = "def foo(x: dict[str, int]) -> list[str]: ..."
        await adapter.generate(task_type=TASK_TYPE_TEST_FUNCTION, source=source)

        assert len(captured_requests) == 1
        messages = cast(
            "list[dict[str, Any]]", getattr(captured_requests[0], "messages", [])
        )
        user_content: str = messages[0]["content"]
        assert source in user_content


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationConstructorValidation:
    """Tests for constructor parameter validation."""

    def test_zero_max_tokens_raises_protocol_configuration_error(self) -> None:
        """max_tokens=0 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="max_tokens"):
            AdapterTestBoilerplateGeneration(
                base_url="http://localhost:8001",
                max_tokens=0,
            )

    def test_negative_max_tokens_raises_protocol_configuration_error(self) -> None:
        """Negative max_tokens raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="max_tokens"):
            AdapterTestBoilerplateGeneration(
                base_url="http://localhost:8001",
                max_tokens=-1,
            )

    def test_oversized_max_tokens_raises_protocol_configuration_error(self) -> None:
        """max_tokens > 32768 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="max_tokens"):
            AdapterTestBoilerplateGeneration(
                base_url="http://localhost:8001",
                max_tokens=32_769,
            )

    @pytest.mark.asyncio
    async def test_valid_max_tokens_does_not_raise(self) -> None:
        """max_tokens=1024 is valid."""
        async with AdapterTestBoilerplateGeneration(
            base_url="http://localhost:8001",
            max_tokens=1024,
        ) as adapter:
            assert adapter._max_tokens == 1024

    @pytest.mark.asyncio
    async def test_min_max_tokens_boundary_does_not_raise(self) -> None:
        """max_tokens=1 is the minimum valid value and must not raise."""
        async with AdapterTestBoilerplateGeneration(
            base_url="http://localhost:8001",
            max_tokens=1,
        ) as adapter:
            assert adapter._max_tokens == 1

    def test_negative_temperature_raises_protocol_configuration_error(self) -> None:
        """temperature < 0.0 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="temperature"):
            AdapterTestBoilerplateGeneration(
                base_url="http://localhost:8001",
                temperature=-0.1,
            )

    def test_oversized_temperature_raises_protocol_configuration_error(self) -> None:
        """temperature > 2.0 raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="temperature"):
            AdapterTestBoilerplateGeneration(
                base_url="http://localhost:8001",
                temperature=2.1,
            )

    @pytest.mark.asyncio
    async def test_valid_temperature_boundary_does_not_raise(self) -> None:
        """temperature=0.0 and temperature=2.0 are valid boundaries."""
        for t in (0.0, 2.0):
            async with AdapterTestBoilerplateGeneration(
                base_url="http://localhost:8001",
                temperature=t,
            ) as adapter:
                assert adapter._temperature == t

    def test_empty_base_url_raises_protocol_configuration_error(self) -> None:
        """Explicit base_url='' raises ProtocolConfigurationError at construction time."""
        with pytest.raises(ProtocolConfigurationError, match="base_url"):
            AdapterTestBoilerplateGeneration(base_url="")

    def test_empty_base_url_error_message_is_clear(self) -> None:
        """ProtocolConfigurationError for empty base_url contains a helpful message."""
        with pytest.raises(ProtocolConfigurationError, match="non-empty string"):
            AdapterTestBoilerplateGeneration(base_url="")

    def test_whitespace_only_base_url_raises_protocol_configuration_error(
        self,
    ) -> None:
        """Whitespace-only base_url raises ProtocolConfigurationError at construction time."""
        with pytest.raises(ProtocolConfigurationError, match="base_url"):
            AdapterTestBoilerplateGeneration(base_url="   ")

    def test_empty_env_var_base_url_raises_protocol_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string LLM_CODER_FAST_URL env var raises ProtocolConfigurationError."""
        monkeypatch.setenv("LLM_CODER_FAST_URL", "")
        with pytest.raises(ProtocolConfigurationError, match="base_url"):
            AdapterTestBoilerplateGeneration(base_url=None)

    def test_whitespace_env_var_base_url_raises_protocol_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace-only LLM_CODER_FAST_URL env var raises ProtocolConfigurationError."""
        monkeypatch.setenv("LLM_CODER_FAST_URL", "   ")
        with pytest.raises(ProtocolConfigurationError, match="base_url"):
            AdapterTestBoilerplateGeneration(base_url=None)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    async def test_close_calls_transport_close(self) -> None:
        """close() delegates to the transport's close() method."""
        adapter = _make_adapter()
        await adapter.close()
        adapter._transport.close.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_close_twice_does_not_raise(self) -> None:
        """Calling close() twice does not raise (transport mock allows multiple calls)."""
        adapter = _make_adapter()
        await adapter.close()
        await adapter.close()


# ---------------------------------------------------------------------------
# Request construction verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationRequestConstruction:
    """Tests for the LLM request fields sent by generate()."""

    @pytest.mark.asyncio
    async def test_request_uses_chat_completion(self) -> None:
        """The LLM request uses CHAT_COMPLETION operation type."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert (
            getattr(req, "operation_type", None) is EnumLlmOperationType.CHAT_COMPLETION
        )

    @pytest.mark.asyncio
    async def test_request_model_matches_configured(self) -> None:
        """The LLM request model matches the configured model identifier."""
        adapter = _make_adapter(model="qwen3-14b-test")
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert len(captured_requests) == 1
        assert getattr(captured_requests[0], "model", None) == "qwen3-14b-test"

    @pytest.mark.asyncio
    async def test_request_has_system_prompt(self) -> None:
        """The LLM request includes a non-empty system prompt."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
        )

        assert len(captured_requests) == 1
        system_prompt = getattr(captured_requests[0], "system_prompt", None)
        assert system_prompt and len(system_prompt) > 0

    @pytest.mark.asyncio
    async def test_system_prompt_mentions_pytest(self) -> None:
        """The system prompt includes pytest-specific guidance."""
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        await adapter.generate(task_type=TASK_TYPE_TEST_CLASS, source="class Foo: pass")

        assert len(captured_requests) == 1
        system_prompt: str = getattr(captured_requests[0], "system_prompt", "")
        assert "pytest" in system_prompt.lower()


# ---------------------------------------------------------------------------
# correlation_id parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationCorrelationId:
    """Tests for the correlation_id parameter on generate()."""

    @pytest.mark.asyncio
    async def test_generate_accepts_correlation_id_uuid(self) -> None:
        """generate() accepts a UUID correlation_id without raising."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )
        cid: UUID = uuid4()

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE,
            source="def foo(): pass",
            correlation_id=cid,
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_generate_works_with_correlation_id_none(self) -> None:
        """generate() works when correlation_id=None (the default)."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        result = await adapter.generate(
            task_type=TASK_TYPE_TEST_MODULE,
            source="def foo(): pass",
            correlation_id=None,
        )

        assert isinstance(result, ContractDelegatedResponse)

    @pytest.mark.asyncio
    async def test_bad_task_type_with_correlation_id_raises_protocol_configuration_error(
        self,
    ) -> None:
        """Bad task_type with a correlation_id still raises ProtocolConfigurationError."""
        adapter = _make_adapter()
        cid: UUID = uuid4()

        with pytest.raises(ProtocolConfigurationError):
            await adapter.generate(
                task_type="bad_type",
                source="def foo(): pass",
                correlation_id=cid,
            )


# ---------------------------------------------------------------------------
# Truncation boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationTruncationBoundary:
    """Tests for the exact truncation boundary behaviour."""

    @pytest.mark.asyncio
    async def test_source_at_limit_plus_50_fits_within_max_chars(self) -> None:
        """When source length == _MAX_SOURCE_CHARS + 50, the source portion in the
        user message is <= _MAX_SOURCE_CHARS characters (sentinel included).

        The adapter truncates source_stripped to exactly _MAX_SOURCE_CHARS chars
        (truncated_body + sentinel == _MAX_SOURCE_CHARS). The test_module template
        injects the source in a plain section; we verify the truncation sentinel
        appears and the full message fits within a reasonable bound.
        """
        adapter = _make_adapter()
        captured_requests: list[object] = []

        async def capturing_handle(request: object) -> MagicMock:
            captured_requests.append(request)
            return _make_llm_response("def test_foo(): pass")

        adapter._handler.handle = capturing_handle  # type: ignore[method-assign,assignment]

        oversized_source = "a" * (_MAX_SOURCE_CHARS + 50)
        await adapter.generate(task_type=TASK_TYPE_TEST_MODULE, source=oversized_source)

        assert len(captured_requests) == 1
        messages = cast(
            "list[dict[str, Any]]", getattr(captured_requests[0], "messages", [])
        )
        assert len(messages) == 1
        user_content: str = messages[0]["content"]

        # Sentinel must appear (truncation did happen).
        assert "[source truncated]" in user_content

        # The injected source portion (everything after the template prefix up to
        # the first occurrence of the sentinel) must be within budget.
        sentinel = "[source truncated]"
        sentinel_pos = user_content.find(sentinel)
        assert sentinel_pos != -1
        source_end = sentinel_pos + len(sentinel)
        # Find where source injection started by locating "## Source\n\n" prefix.
        section_marker = "## Source\n\n"
        section_pos = user_content.find(section_marker)
        assert section_pos != -1, "Section marker must appear in user content"
        source_start = section_pos + len(section_marker)
        injected_source = user_content[source_start:source_end]
        assert len(injected_source) <= _MAX_SOURCE_CHARS


# ---------------------------------------------------------------------------
# LLM exception propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationLlmException:
    """Tests that RuntimeHostError from the LLM handler propagates out of generate()."""

    @pytest.mark.asyncio
    async def test_runtime_host_error_propagates(self) -> None:
        """When HandlerLlmOpenaiCompatible raises RuntimeHostError, it propagates."""
        adapter = _make_adapter()

        async def failing_handle(request: object) -> MagicMock:
            raise RuntimeHostError("simulated LLM failure")

        adapter._handler.handle = failing_handle  # type: ignore[method-assign,assignment]

        with pytest.raises(RuntimeHostError, match="simulated LLM failure"):
            await adapter.generate(
                task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
            )

    @pytest.mark.asyncio
    async def test_runtime_host_error_not_swallowed(self) -> None:
        """generate() does not swallow RuntimeHostError into another exception type."""
        adapter = _make_adapter()

        async def failing_handle(request: object) -> MagicMock:
            raise RuntimeHostError("connection timeout")

        adapter._handler.handle = failing_handle  # type: ignore[method-assign,assignment]

        with pytest.raises(RuntimeHostError) as exc_info:
            await adapter.generate(
                task_type=TASK_TYPE_TEST_CLASS, source="class Foo: pass"
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
            await adapter.generate(
                task_type=TASK_TYPE_TEST_FUNCTION,
                source="def compute(x: int) -> int: ...",
            )

        assert exc_info.value is original_error


# ---------------------------------------------------------------------------
# Async context manager protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterTestBoilerplateGenerationAsyncContextManager:
    """Tests for the async context manager protocol (__aenter__ / __aexit__)."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        """__aenter__ returns the adapter instance itself."""
        adapter = _make_adapter()
        result = await adapter.__aenter__()
        assert result is adapter

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self) -> None:
        """__aexit__ delegates to close(), which calls transport.close()."""
        adapter = _make_adapter()
        await adapter.__aexit__(None, None, None)
        adapter._transport.close.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_async_with_block_calls_close_on_exit(self) -> None:
        """Using 'async with' calls transport.close() when the block exits normally."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("def test_foo(): pass")
        )

        async with adapter as ctx:
            assert ctx is adapter
            result = await ctx.generate(
                task_type=TASK_TYPE_TEST_MODULE, source="def foo(): pass"
            )
            assert isinstance(result, ContractDelegatedResponse)

        adapter._transport.close.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_async_with_block_calls_close_on_exception(self) -> None:
        """Using 'async with' calls transport.close() even when the block raises."""
        adapter = _make_adapter()

        with pytest.raises(RuntimeError, match="intentional"):
            async with adapter:
                raise RuntimeError("intentional")

        adapter._transport.close.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_aexit_passes_exception_type_through(self) -> None:
        """__aexit__ returns None (does not suppress exceptions)."""
        adapter = _make_adapter()
        result = await adapter.__aexit__(ValueError, ValueError("test"), None)
        assert result is None
