# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerEmbeddingOpenaiCompatible.

Verifies:
- handler_type and handler_category properties
- Successful single-text embedding via mocked _execute_llm_http_call
- Successful batch embedding
- URL construction: POST {base_url}/v1/embeddings
- Request payload structure (model, input, optional dimensions)
- Response parsing: data[].embedding -> ModelEmbedding mapping
- Usage parsing: prompt_tokens -> tokens_input
- Provider ID extraction from response
- Malformed response handling (missing data, empty data, non-dict items)
- Correlation ID propagation through InfraProtocolError

Related:
    - OMN-2113: Phase 13 embedding tests
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import InfraProtocolError
from omnibase_infra.nodes.node_llm_embedding_effect.handlers.handler_embedding_openai_compatible import (
    HandlerEmbeddingOpenaiCompatible,
    _parse_openai_embeddings,
    _parse_openai_usage,
)
from omnibase_infra.nodes.node_llm_embedding_effect.models.model_llm_embedding_request import (
    ModelLlmEmbeddingRequest,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Fixtures
# =============================================================================


def _make_request(**overrides: object) -> ModelLlmEmbeddingRequest:
    """Create a valid embedding request with optional overrides."""
    defaults: dict = {
        "base_url": "http://192.168.86.201:8002",
        "model": "gte-qwen2-1.5b",
        "texts": ("Hello, world!",),
        "max_retries": 0,
        "timeout_seconds": 5.0,
    }
    defaults.update(overrides)
    return ModelLlmEmbeddingRequest(**defaults)


def _openai_response(
    embeddings: list[list[float]],
    prompt_tokens: int = 5,
    model: str = "gte-qwen2-1.5b",
) -> dict:
    """Build a mock OpenAI-compatible embedding response."""
    return {
        "data": [{"index": i, "embedding": vec} for i, vec in enumerate(embeddings)],
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        "model": model,
    }


# =============================================================================
# Property Tests
# =============================================================================


class TestHandlerProperties:
    """Tests for handler_type and handler_category."""

    def test_handler_type_is_infra(self) -> None:
        """handler_type is INFRA_HANDLER."""
        handler = HandlerEmbeddingOpenaiCompatible()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_effect(self) -> None:
        """handler_category is EFFECT."""
        handler = HandlerEmbeddingOpenaiCompatible()
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    def test_custom_target_name(self) -> None:
        """Custom target_name is stored."""
        handler = HandlerEmbeddingOpenaiCompatible(target_name="custom-target")
        assert handler._llm_target_name == "custom-target"

    def test_default_target_name(self) -> None:
        """Default target_name is 'openai-embedding'."""
        handler = HandlerEmbeddingOpenaiCompatible()
        assert handler._llm_target_name == "openai-embedding"


# =============================================================================
# Successful Execution Tests
# =============================================================================


class TestHandlerExecuteSuccess:
    """Tests for successful embedding execution."""

    @pytest.mark.anyio
    async def test_single_text_embedding(self) -> None:
        """Single text returns single embedding."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2, 0.3]])
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert len(resp.embeddings) == 1
        assert resp.embeddings[0].vector == [0.1, 0.2, 0.3]
        assert resp.embeddings[0].id == "0"
        assert resp.dimensions == 3
        assert resp.model_used == "gte-qwen2-1.5b"
        assert resp.status == "success"
        assert resp.backend_result.success is True

    @pytest.mark.anyio
    async def test_batch_embedding(self) -> None:
        """Multiple texts return multiple embeddings in order."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        request = _make_request(texts=("one", "two", "three"))

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert len(resp.embeddings) == 3
        assert resp.embeddings[0].id == "0"
        assert resp.embeddings[1].id == "1"
        assert resp.embeddings[2].id == "2"
        assert resp.dimensions == 3

    @pytest.mark.anyio
    async def test_usage_parsing(self) -> None:
        """Usage prompt_tokens maps to tokens_input."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]], prompt_tokens=42)
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert resp.usage.tokens_input == 42
        assert resp.usage.tokens_output == 0

    @pytest.mark.anyio
    async def test_provider_id_from_response_model(self) -> None:
        """provider_id is extracted from response 'model' field."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]], model="gte-qwen2-1.5b-vllm")
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert resp.provider_id == "gte-qwen2-1.5b-vllm"

    @pytest.mark.anyio
    async def test_provider_id_none_when_not_string(self) -> None:
        """provider_id is None when response 'model' is not a string."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        mock_data["model"] = 12345  # Not a string
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert resp.provider_id is None

    @pytest.mark.anyio
    async def test_correlation_id_propagated(self) -> None:
        """Correlation ID from request propagates to response."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        cid = uuid4()
        request = _make_request(correlation_id=cid)

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert resp.correlation_id == cid

    @pytest.mark.anyio
    async def test_execution_id_propagated(self) -> None:
        """Execution ID from request propagates to response."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        eid = uuid4()
        request = _make_request(execution_id=eid)

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert resp.execution_id == eid


# =============================================================================
# URL and Payload Construction Tests
# =============================================================================


class TestUrlAndPayloadConstruction:
    """Tests for URL construction and payload structure."""

    @pytest.mark.anyio
    async def test_url_construction(self) -> None:
        """URL is {base_url}/v1/embeddings."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        request = _make_request(base_url="http://192.168.86.201:8002")

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["url"] == "http://192.168.86.201:8002/v1/embeddings"

    @pytest.mark.anyio
    async def test_url_strips_trailing_slash(self) -> None:
        """Trailing slash in base_url is stripped before appending path."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        request = _make_request(base_url="http://192.168.86.201:8002/")

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["url"] == "http://192.168.86.201:8002/v1/embeddings"

    @pytest.mark.anyio
    async def test_payload_without_dimensions(self) -> None:
        """Payload excludes 'dimensions' when not set."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        payload = call_kwargs.kwargs["payload"]
        assert "dimensions" not in payload
        assert payload["model"] == "gte-qwen2-1.5b"
        assert payload["input"] == ["Hello, world!"]

    @pytest.mark.anyio
    async def test_payload_with_dimensions(self) -> None:
        """Payload includes 'dimensions' when set on request."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        request = _make_request(dimensions=512)

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        payload = call_kwargs.kwargs["payload"]
        assert payload["dimensions"] == 512

    @pytest.mark.anyio
    async def test_texts_tuple_converted_to_list_in_payload(self) -> None:
        """texts tuple is converted to list for JSON payload."""
        handler = HandlerEmbeddingOpenaiCompatible()
        mock_data = _openai_response([[0.1, 0.2]])
        request = _make_request(texts=("hello", "world"))

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        payload = call_kwargs.kwargs["payload"]
        assert payload["input"] == ["hello", "world"]
        assert isinstance(payload["input"], list)


# =============================================================================
# Malformed Response Tests
# =============================================================================


class TestMalformedResponseHandling:
    """Tests for malformed response parsing raises InfraProtocolError."""

    @pytest.mark.anyio
    async def test_missing_data_array(self) -> None:
        """Response without 'data' raises InfraProtocolError."""
        handler = HandlerEmbeddingOpenaiCompatible()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"usage": {"prompt_tokens": 5}}
            with pytest.raises(InfraProtocolError, match="Malformed OpenAI"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_empty_data_array(self) -> None:
        """Response with empty 'data' array raises InfraProtocolError."""
        handler = HandlerEmbeddingOpenaiCompatible()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"data": []}
            with pytest.raises(InfraProtocolError, match="Malformed OpenAI"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_non_dict_item_in_data(self) -> None:
        """Non-dict item in 'data' raises InfraProtocolError."""
        handler = HandlerEmbeddingOpenaiCompatible()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"data": ["not-a-dict"]}
            with pytest.raises(InfraProtocolError, match="Malformed OpenAI"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_missing_embedding_field(self) -> None:
        """Missing 'embedding' field in data item raises InfraProtocolError."""
        handler = HandlerEmbeddingOpenaiCompatible()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"data": [{"index": 0}]}
            with pytest.raises(InfraProtocolError, match="Malformed OpenAI"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_non_list_embedding_field(self) -> None:
        """Non-list 'embedding' field in data raises InfraProtocolError."""
        handler = HandlerEmbeddingOpenaiCompatible()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"data": [{"index": 0, "embedding": "not-a-list"}]}
            with pytest.raises(InfraProtocolError, match="Malformed OpenAI"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_correlation_id_in_protocol_error(self) -> None:
        """InfraProtocolError carries correlation_id from request."""
        handler = HandlerEmbeddingOpenaiCompatible()
        cid = uuid4()
        request = _make_request(correlation_id=cid)

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"data": []}
            with pytest.raises(InfraProtocolError) as exc_info:
                await handler.execute(request)

        assert exc_info.value.correlation_id == cid


# =============================================================================
# Parser Unit Tests (synchronous, no handler needed)
# =============================================================================


class TestParseOpenaiEmbeddings:
    """Tests for _parse_openai_embeddings helper function."""

    def test_valid_single(self) -> None:
        """Single embedding parses correctly."""
        data = {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
        result = _parse_openai_embeddings(data)
        assert len(result) == 1
        assert result[0].id == "0"
        assert result[0].vector == [1.0, 2.0, 3.0]

    def test_valid_batch(self) -> None:
        """Multiple embeddings parse correctly with correct IDs."""
        data = {
            "data": [
                {"index": 0, "embedding": [1.0, 2.0]},
                {"index": 1, "embedding": [3.0, 4.0]},
            ]
        }
        result = _parse_openai_embeddings(data)
        assert len(result) == 2
        assert result[0].id == "0"
        assert result[1].id == "1"

    def test_missing_index_uses_position(self) -> None:
        """Missing 'index' field defaults to current position."""
        data = {"data": [{"embedding": [1.0, 2.0]}]}
        result = _parse_openai_embeddings(data)
        assert result[0].id == "0"

    def test_missing_data_key_raises(self) -> None:
        """Missing 'data' key raises ValueError."""
        with pytest.raises(ValueError, match="missing or empty 'data'"):
            _parse_openai_embeddings({})

    def test_data_is_none_raises(self) -> None:
        """data=None raises ValueError."""
        with pytest.raises(ValueError, match="missing or empty 'data'"):
            _parse_openai_embeddings({"data": None})

    def test_empty_data_raises(self) -> None:
        """Empty data list raises ValueError."""
        with pytest.raises(ValueError, match="missing or empty 'data'"):
            _parse_openai_embeddings({"data": []})

    def test_non_dict_item_raises(self) -> None:
        """Non-dict item in data raises ValueError."""
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_openai_embeddings({"data": [42]})

    def test_non_list_embedding_raises(self) -> None:
        """Non-list embedding value raises ValueError."""
        with pytest.raises(ValueError, match="Expected list for 'embedding'"):
            _parse_openai_embeddings({"data": [{"index": 0, "embedding": "bad"}]})


class TestParseOpenaiUsage:
    """Tests for _parse_openai_usage helper function."""

    def test_valid_usage(self) -> None:
        """Valid usage block parses correctly."""
        data = {"usage": {"prompt_tokens": 42, "total_tokens": 42}}
        usage = _parse_openai_usage(data)
        assert usage.tokens_input == 42
        assert usage.tokens_output == 0

    def test_missing_usage_block(self) -> None:
        """Missing usage block returns zero-usage."""
        usage = _parse_openai_usage({})
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0

    def test_usage_not_dict(self) -> None:
        """Non-dict usage block returns zero-usage."""
        usage = _parse_openai_usage({"usage": "not-a-dict"})
        assert usage.tokens_input == 0

    def test_non_int_prompt_tokens(self) -> None:
        """Non-int prompt_tokens defaults to 0."""
        usage = _parse_openai_usage({"usage": {"prompt_tokens": "bad"}})
        assert usage.tokens_input == 0

    def test_missing_prompt_tokens(self) -> None:
        """Missing prompt_tokens defaults to 0."""
        usage = _parse_openai_usage({"usage": {}})
        assert usage.tokens_input == 0
