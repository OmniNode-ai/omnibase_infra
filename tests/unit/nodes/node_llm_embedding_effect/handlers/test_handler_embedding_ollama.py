# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerEmbeddingOllama.

Verifies:
- handler_type and handler_category properties
- Successful single-text embedding via mocked _execute_llm_http_call
- Successful batch embedding
- URL construction: POST {base_url}/api/embed
- Request payload structure (model, input -- no dimensions)
- Response parsing: embeddings[][] -> ModelEmbedding mapping
- Usage parsing: prompt_eval_count -> tokens_input
- Provider ID extraction from response
- Malformed response handling (missing embeddings, empty, non-list items)
- Correlation ID propagation through InfraProtocolError

Related:
    - OMN-2113: Phase 13 embedding tests
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import InfraProtocolError
from omnibase_infra.nodes.node_llm_embedding_effect.handlers.handler_embedding_ollama import (
    HandlerEmbeddingOllama,
    _parse_ollama_embeddings,
    _parse_ollama_usage,
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
    defaults: dict[str, object] = {
        "base_url": "http://192.168.86.200:11434",
        "model": "nomic-embed-text",
        "texts": ("Hello, world!",),
        "max_retries": 0,
        "timeout_seconds": 5.0,
    }
    defaults.update(overrides)
    return ModelLlmEmbeddingRequest(**defaults)


def _ollama_response(
    embeddings: list[list[float]],
    prompt_eval_count: int = 5,
    model: str = "nomic-embed-text",
) -> dict[str, object]:
    """Build a mock Ollama embedding response."""
    return {
        "model": model,
        "embeddings": embeddings,
        "total_duration": 123456789,
        "load_duration": 123456,
        "prompt_eval_count": prompt_eval_count,
    }


# =============================================================================
# Property Tests
# =============================================================================


class TestHandlerProperties:
    """Tests for handler_type and handler_category."""

    def test_handler_type_is_infra(self) -> None:
        """handler_type is INFRA_HANDLER."""
        handler = HandlerEmbeddingOllama()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_effect(self) -> None:
        """handler_category is EFFECT."""
        handler = HandlerEmbeddingOllama()
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    def test_custom_target_name(self) -> None:
        """Custom target_name is stored."""
        handler = HandlerEmbeddingOllama(target_name="custom-ollama")
        assert handler._llm_target_name == "custom-ollama"

    def test_default_target_name(self) -> None:
        """Default target_name is 'ollama-embedding'."""
        handler = HandlerEmbeddingOllama()
        assert handler._llm_target_name == "ollama-embedding"


# =============================================================================
# Successful Execution Tests
# =============================================================================


class TestHandlerExecuteSuccess:
    """Tests for successful embedding execution."""

    @pytest.mark.anyio
    async def test_single_text_embedding(self) -> None:
        """Single text returns single embedding."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2, 0.3]])
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
        assert resp.model_used == "nomic-embed-text"
        assert resp.status == "success"
        assert resp.backend_result.success is True

    @pytest.mark.anyio
    async def test_batch_embedding(self) -> None:
        """Multiple texts return multiple embeddings in order."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response(
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
        """Usage prompt_eval_count maps to tokens_input."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]], prompt_eval_count=42)
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
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]], model="nomic-embed-text:latest")
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            resp = await handler.execute(request)

        assert resp.provider_id == "nomic-embed-text:latest"

    @pytest.mark.anyio
    async def test_provider_id_none_when_not_string(self) -> None:
        """provider_id is None when response 'model' is not a string."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
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
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
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
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
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
        """URL is {base_url}/api/embed."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
        request = _make_request(base_url="http://192.168.86.200:11434")

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["url"] == "http://192.168.86.200:11434/api/embed"

    @pytest.mark.anyio
    async def test_url_strips_trailing_slash(self) -> None:
        """Trailing slash in base_url is stripped before appending path."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
        request = _make_request(base_url="http://192.168.86.200:11434/")

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["url"] == "http://192.168.86.200:11434/api/embed"

    @pytest.mark.anyio
    async def test_payload_structure(self) -> None:
        """Payload has 'model' and 'input' but no 'dimensions'."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
        request = _make_request(dimensions=512)

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_data
            await handler.execute(request)

        call_kwargs = mock_call.call_args
        payload = call_kwargs.kwargs["payload"]
        # Ollama handler does NOT send dimensions
        assert "dimensions" not in payload
        assert payload["model"] == "nomic-embed-text"
        assert payload["input"] == ["Hello, world!"]

    @pytest.mark.anyio
    async def test_texts_tuple_converted_to_list_in_payload(self) -> None:
        """texts tuple is converted to list for JSON payload."""
        handler = HandlerEmbeddingOllama()
        mock_data = _ollama_response([[0.1, 0.2]])
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
    async def test_missing_embeddings_key(self) -> None:
        """Response without 'embeddings' raises InfraProtocolError."""
        handler = HandlerEmbeddingOllama()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"model": "test"}
            with pytest.raises(InfraProtocolError, match="Malformed Ollama"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_empty_embeddings_array(self) -> None:
        """Response with empty 'embeddings' raises InfraProtocolError."""
        handler = HandlerEmbeddingOllama()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"embeddings": []}
            with pytest.raises(InfraProtocolError, match="Malformed Ollama"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_non_list_embedding_item(self) -> None:
        """Non-list item in embeddings raises InfraProtocolError."""
        handler = HandlerEmbeddingOllama()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"embeddings": ["not-a-list"]}
            with pytest.raises(InfraProtocolError, match="Malformed Ollama"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_embeddings_is_none(self) -> None:
        """embeddings=None raises InfraProtocolError."""
        handler = HandlerEmbeddingOllama()
        request = _make_request()

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"embeddings": None}
            with pytest.raises(InfraProtocolError, match="Malformed Ollama"):
                await handler.execute(request)

    @pytest.mark.anyio
    async def test_correlation_id_in_protocol_error(self) -> None:
        """InfraProtocolError carries correlation_id from request."""
        handler = HandlerEmbeddingOllama()
        cid = uuid4()
        request = _make_request(correlation_id=cid)

        with patch.object(
            handler, "_execute_llm_http_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"embeddings": []}
            with pytest.raises(InfraProtocolError) as exc_info:
                await handler.execute(request)

        assert exc_info.value.correlation_id == cid


# =============================================================================
# Parser Unit Tests (synchronous, no handler needed)
# =============================================================================


class TestParseOllamaEmbeddings:
    """Tests for _parse_ollama_embeddings helper function."""

    def test_valid_single(self) -> None:
        """Single embedding parses correctly."""
        data: dict[str, JsonType] = {"embeddings": [[1.0, 2.0, 3.0]]}
        result = _parse_ollama_embeddings(data)
        assert len(result) == 1
        assert result[0].id == "0"
        assert result[0].vector == [1.0, 2.0, 3.0]

    def test_valid_batch(self) -> None:
        """Multiple embeddings parse correctly with sequential IDs."""
        data: dict[str, JsonType] = {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        result = _parse_ollama_embeddings(data)
        assert len(result) == 2
        assert result[0].id == "0"
        assert result[1].id == "1"

    def test_missing_embeddings_key_raises(self) -> None:
        """Missing 'embeddings' key raises ValueError."""
        with pytest.raises(ValueError, match="missing or empty 'embeddings'"):
            _parse_ollama_embeddings({})

    def test_embeddings_is_none_raises(self) -> None:
        """embeddings=None raises ValueError."""
        with pytest.raises(ValueError, match="missing or empty 'embeddings'"):
            _parse_ollama_embeddings({"embeddings": None})

    def test_empty_embeddings_raises(self) -> None:
        """Empty embeddings list raises ValueError."""
        with pytest.raises(ValueError, match="missing or empty 'embeddings'"):
            _parse_ollama_embeddings({"embeddings": []})

    def test_non_list_embedding_raises(self) -> None:
        """Non-list embedding entry raises ValueError."""
        with pytest.raises(ValueError, match="Expected list for embedding"):
            _parse_ollama_embeddings({"embeddings": ["not-a-list"]})

    def test_non_list_at_specific_index(self) -> None:
        """Non-list at specific index references the index in error."""
        with pytest.raises(ValueError, match="index 1"):
            _parse_ollama_embeddings({"embeddings": [[1.0, 2.0], "bad"]})


class TestParseOllamaUsage:
    """Tests for _parse_ollama_usage helper function."""

    def test_valid_usage(self) -> None:
        """Valid prompt_eval_count parses correctly."""
        data: dict[str, JsonType] = {"prompt_eval_count": 42}
        usage = _parse_ollama_usage(data)
        assert usage.tokens_input == 42
        assert usage.tokens_output == 0

    def test_missing_prompt_eval_count(self) -> None:
        """Missing prompt_eval_count defaults to 0."""
        usage = _parse_ollama_usage({})
        assert usage.tokens_input == 0

    def test_non_int_prompt_eval_count(self) -> None:
        """Non-int prompt_eval_count defaults to 0."""
        usage = _parse_ollama_usage({"prompt_eval_count": "bad"})
        assert usage.tokens_input == 0

    def test_zero_prompt_eval_count(self) -> None:
        """Zero prompt_eval_count is valid."""
        usage = _parse_ollama_usage({"prompt_eval_count": 0})
        assert usage.tokens_input == 0
