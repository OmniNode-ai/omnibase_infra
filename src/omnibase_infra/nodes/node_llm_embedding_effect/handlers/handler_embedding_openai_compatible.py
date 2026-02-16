# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OpenAI-compatible embedding handler for the embedding effect node.

Calls ``POST {base_url}/v1/embeddings`` with the standard OpenAI embeddings
request format and converts the response into ``ModelEmbedding`` instances.

Supported providers:
    - OpenAI (``api.openai.com``)
    - vLLM (``/v1/embeddings`` endpoint)
    - Any OpenAI-compatible server (e.g. GTE-Qwen2 via vLLM)

Related:
    - MixinLlmHttpTransport: HTTP transport with retry and circuit breaker
    - ModelLlmEmbeddingRequest: Input model
    - ModelLlmEmbeddingResponse: Output model
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from omnibase_core.models.vector import ModelEmbedding
from omnibase_core.types import JsonType
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import InfraProtocolError, ModelInfraErrorContext
from omnibase_infra.mixins import MixinLlmHttpTransport
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.nodes.node_llm_embedding_effect.models.model_llm_embedding_request import (
    ModelLlmEmbeddingRequest,
)
from omnibase_infra.nodes.node_llm_embedding_effect.models.model_llm_embedding_response import (
    ModelLlmEmbeddingResponse,
)
from omnibase_spi.contracts.measurement import ContractEnumUsageSource

logger = logging.getLogger(__name__)


class HandlerEmbeddingOpenaiCompatible(MixinLlmHttpTransport):
    """Handler for OpenAI-compatible embedding endpoints.

    Sends ``POST {base_url}/v1/embeddings`` with::

        {
            "model": "<model>",
            "input": ["text1", "text2", ...],
            "dimensions": <int>  // optional
        }

    And expects a response with::

        {
            "data": [{"index": 0, "embedding": [0.1, ...]}, ...],
            "usage": {"prompt_tokens": N, "total_tokens": M}
        }

    Each returned embedding is converted to a ``ModelEmbedding(id=str(index),
    vector=floats, metadata={})``.

    Attributes:
        handler_type: ``INFRA_HANDLER`` (manages external HTTP connection).
        handler_category: ``EFFECT`` (side-effecting HTTP I/O).
    """

    def __init__(self, target_name: str = "openai-embedding") -> None:
        """Initialize the OpenAI-compatible embedding handler.

        Args:
            target_name: Identifier for the target (used in error context
                and logging). Default: ``"openai-embedding"``.
        """
        self._init_llm_http_transport(target_name=target_name)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting I/O."""
        return EnumHandlerTypeCategory.EFFECT

    async def execute(
        self, request: ModelLlmEmbeddingRequest
    ) -> ModelLlmEmbeddingResponse:
        """Execute an embedding request against an OpenAI-compatible endpoint.

        Args:
            request: The embedding request containing texts, model, and config.

        Returns:
            ModelLlmEmbeddingResponse with embeddings and usage metadata.

        Raises:
            InfraConnectionError: On connection failure after retries.
            InfraTimeoutError: On timeout after retries.
            InfraAuthenticationError: On 401/403 responses.
            InfraRateLimitedError: On 429 when retries exhausted.
            InfraRequestRejectedError: On 400/422 responses.
            ProtocolConfigurationError: On 404 responses.
            InfraUnavailableError: On 5xx or circuit breaker open.
            InfraProtocolError: On malformed embedding response body.
        """
        url = f"{request.base_url.rstrip('/')}/v1/embeddings"

        payload: dict[str, JsonType] = {
            "model": request.model,
            "input": list(request.texts),
        }
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions

        start_time = time.monotonic()

        data = await self._execute_llm_http_call(
            url=url,
            payload=payload,
            correlation_id=request.correlation_id,
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Parse response — wrap ValueError from parsers in typed infra error
        try:
            embeddings = _parse_openai_embeddings(data)
        except ValueError as exc:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=request.correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="parse_openai_embeddings",
                target_name=self._llm_target_name,
            )
            raise InfraProtocolError(
                f"Malformed OpenAI embedding response: {exc}",
                context=ctx,
            ) from exc
        usage = _parse_openai_usage(data)
        if not embeddings:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=request.correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="parse_openai_embeddings",
                target_name=self._llm_target_name,
            )
            raise InfraProtocolError(
                "OpenAI embedding response returned no embeddings",
                context=ctx,
            )
        dimensions = len(embeddings[0].vector)

        return ModelLlmEmbeddingResponse(
            embeddings=tuple(embeddings),
            dimensions=dimensions,
            model_used=request.model,
            provider_id=data.get("model", None)
            if isinstance(data.get("model"), str)
            else None,
            usage=usage,
            latency_ms=elapsed_ms,
            backend_result=ModelBackendResult(success=True, duration_ms=elapsed_ms),
            correlation_id=request.correlation_id,
            execution_id=request.execution_id,
            timestamp=datetime.now(UTC),
        )


def _parse_openai_embeddings(data: dict[str, JsonType]) -> list[ModelEmbedding]:
    """Parse embedding vectors from an OpenAI-compatible response.

    Args:
        data: Parsed JSON response body.

    Returns:
        List of ModelEmbedding instances ordered by index.

    Raises:
        ValueError: If ``data`` is missing or has invalid structure.
    """
    raw_data = data.get("data")
    if not isinstance(raw_data, list) or not raw_data:
        msg = "OpenAI embedding response missing or empty 'data' array"
        raise ValueError(msg)

    embeddings: list[ModelEmbedding] = []
    for item in raw_data:
        if not isinstance(item, dict):
            msg = f"Expected dict in 'data' array, got {type(item).__name__}"
            raise ValueError(msg)
        index = item.get("index", len(embeddings))
        vector = item.get("embedding")
        if not isinstance(vector, list):
            msg = f"Expected list for 'embedding' at index {index}, got {type(vector).__name__}"
            raise ValueError(msg)
        embeddings.append(
            ModelEmbedding(
                id=str(index),
                vector=vector,
                metadata={},
            )
        )

    return embeddings


def _parse_openai_usage(data: dict[str, JsonType]) -> ModelLlmUsage:
    """Parse usage metadata from an OpenAI-compatible response.

    When the provider returns a valid usage dict, ``usage_source`` is set
    to ``API`` and the raw dict is preserved.  When absent, ``MISSING``.

    Args:
        data: Parsed JSON response body.

    Returns:
        ModelLlmUsage with token counts and provenance. Zeros if usage
        block is absent.
    """
    usage_raw = data.get("usage")
    if not isinstance(usage_raw, dict):
        return ModelLlmUsage(
            usage_source=ContractEnumUsageSource.MISSING,
        )

    prompt_tokens = usage_raw.get("prompt_tokens", 0)
    if not isinstance(prompt_tokens, int):
        prompt_tokens = 0

    return ModelLlmUsage(
        tokens_input=prompt_tokens,
        tokens_output=0,
        usage_source=ContractEnumUsageSource.API,
        raw_provider_usage=dict(usage_raw),
    )


__all__: list[str] = ["HandlerEmbeddingOpenaiCompatible"]
