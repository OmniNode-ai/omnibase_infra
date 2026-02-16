# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete ProtocolLLMProvider implementation for OpenAI-compatible endpoints.

Wraps the existing HandlerLlmOpenaiCompatible and MixinLlmHttpTransport
behind the SPI ProtocolLLMProvider interface, bridging structural mismatches
between SPI request/response models and infra-layer models.

Architecture:
    - Owns an MixinLlmHttpTransport instance for HTTP transport
    - Delegates inference calls to HandlerLlmOpenaiCompatible
    - Translates ProtocolLLMRequest -> ModelLlmInferenceRequest
    - Translates ModelLlmInferenceResponse -> ProtocolLLMResponse
    - Provides health check, cost estimation, and capability discovery

Related Tickets:
    - OMN-2319: Implement SPI LLM protocol adapters (Gap 1)
    - OMN-2107: HandlerLlmOpenaiCompatible
    - OMN-2104: MixinLlmHttpTransport
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING

from omnibase_core.types import JsonType
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.adapters.llm.model_llm_health_response import (
    ModelLlmHealthResponse,
)
from omnibase_infra.adapters.llm.model_llm_model_capabilities import (
    ModelLlmModelCapabilities,
)
from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

if TYPE_CHECKING:
    from omnibase_infra.adapters.llm.model_llm_provider_config import (
        ModelLlmProviderConfig,
    )

logger = logging.getLogger(__name__)


class _TransportHolder(MixinLlmHttpTransport):
    """Internal transport holder that mixes in LLM HTTP transport.

    HandlerLlmOpenaiCompatible expects a transport instance with
    ``_execute_llm_http_call``. This holder provides that by extending
    MixinLlmHttpTransport.
    """

    def __init__(
        self,
        target_name: str = "openai-compatible",
        max_timeout_seconds: float = 120.0,
    ) -> None:
        self._init_llm_http_transport(
            target_name=target_name,
            max_timeout_seconds=max_timeout_seconds,
        )

    async def close(self) -> None:
        """Close the HTTP client if owned."""
        await self._close_http_client()


class AdapterLlmProviderOpenai:
    """ProtocolLLMProvider implementation for OpenAI-compatible endpoints.

    Wraps HandlerLlmOpenaiCompatible and MixinLlmHttpTransport behind the
    SPI ProtocolLLMProvider interface. Handles translation between SPI-level
    request/response models and infra-level models.

    This adapter supports all OpenAI-compatible inference servers including
    vLLM, text-generation-inference, and the OpenAI API itself.

    Attributes:
        _provider_name: Provider identifier.
        _provider_type: Deployment type classification.
        _base_url: Base URL of the LLM endpoint.
        _default_model: Default model when not specified per-request.
        _api_key: Optional API key for authentication.
        _is_available: Current availability status.
        _transport: HTTP transport instance.
        _handler: OpenAI-compatible inference handler.
        _capabilities_cache: Cached model capabilities.

    Example:
        >>> adapter = AdapterLlmProviderOpenai(
        ...     base_url="http://192.168.86.201:8000",
        ...     default_model="qwen2.5-coder-14b",
        ... )
        >>> response = await adapter.generate_async(request)
        >>> await adapter.close()
    """

    def __init__(
        self,
        base_url: str = "http://192.168.86.201:8000",
        default_model: str = "qwen2.5-coder-14b",
        api_key: str | None = None,
        provider_name: str = "openai-compatible",
        provider_type: str = "local",
        max_timeout_seconds: float = 120.0,
        model_capabilities: dict[str, ModelLlmModelCapabilities] | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible provider adapter.

        Args:
            base_url: Base URL of the LLM endpoint.
            default_model: Default model identifier.
            api_key: Optional API key for Bearer token auth.
            provider_name: Provider identifier for logging/routing.
            provider_type: Deployment type: 'local', 'external_trusted', 'external'.
            max_timeout_seconds: Maximum timeout for any single request.
            model_capabilities: Pre-configured model capabilities mapping.
        """
        self._provider_name_value = provider_name
        self._provider_type_value = provider_type
        self._base_url = base_url
        self._default_model = default_model
        self._api_key = api_key
        self._is_available = True

        self._transport = _TransportHolder(
            target_name=provider_name,
            max_timeout_seconds=max_timeout_seconds,
        )
        self._handler = HandlerLlmOpenaiCompatible(self._transport)
        self._capabilities_cache: dict[str, ModelLlmModelCapabilities] = (
            model_capabilities or {}
        )

    # ── ProtocolLLMProvider properties ─────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Get the provider name identifier."""
        return self._provider_name_value

    @property
    def provider_type(self) -> str:
        """Get the provider deployment type classification."""
        return self._provider_type_value

    @property
    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        return self._is_available

    # ── Configuration ──────────────────────────────────────────────────

    def configure(self, config: ModelLlmProviderConfig) -> None:
        """Configure the provider with connection and authentication details.

        Args:
            config: Provider configuration with API keys, URLs, timeouts.
        """
        if config.base_url is not None:
            self._base_url = config.base_url
        if config.api_key is not None:
            self._api_key = config.api_key
        if config.default_model:
            self._default_model = config.default_model
        self._provider_type_value = config.provider_type

    # ── Model discovery ────────────────────────────────────────────────

    async def get_available_models(self) -> list[str]:
        """Get list of available models from this provider.

        For OpenAI-compatible endpoints, attempts to call the /v1/models
        endpoint. Falls back to returning the default model if the endpoint
        is unavailable.

        Returns:
            List of model identifiers.
        """
        try:
            client = await self._transport._get_http_client()
            url = f"{self._base_url.rstrip('/')}/v1/models"
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    models = []
                    for item in data["data"]:
                        if isinstance(item, dict) and "id" in item:
                            models.append(str(item["id"]))
                    if models:
                        return models
        except Exception:
            logger.debug(
                "Could not fetch models from %s, returning default",
                self._base_url,
            )

        return [self._default_model] if self._default_model else []

    async def get_model_capabilities(
        self, model_name: str
    ) -> ModelLlmModelCapabilities:
        """Get capabilities for a specific model.

        Returns cached capabilities if available, otherwise returns
        defaults appropriate for an OpenAI-compatible endpoint.

        Args:
            model_name: Model identifier to query.

        Returns:
            Model capabilities description.
        """
        if model_name in self._capabilities_cache:
            return self._capabilities_cache[model_name]

        # Return reasonable defaults for OpenAI-compatible endpoints
        return ModelLlmModelCapabilities(
            model_name=model_name,
            supports_streaming=True,
            supports_function_calling=True,
            max_context_length=32768,
            supported_modalities=["text"],
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
        )

    # ── Request validation ─────────────────────────────────────────────

    def validate_request(self, request: ModelLlmAdapterRequest) -> bool:
        """Validate that the request is compatible with this provider.

        Args:
            request: LLM request to validate.

        Returns:
            True if the request can be handled by this provider.
        """
        if not request.prompt:
            return False
        if not request.model_name:
            return False
        return True

    # ── Generation ─────────────────────────────────────────────────────

    async def generate(
        self, request: ModelLlmAdapterRequest
    ) -> ModelLlmAdapterResponse:
        """Generate a response using this provider.

        Delegates to generate_async since the underlying transport is async.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            Generated response with usage metrics.
        """
        return await self.generate_async(request)

    async def generate_async(
        self, request: ModelLlmAdapterRequest
    ) -> ModelLlmAdapterResponse:
        """Generate a response asynchronously.

        Translates the SPI-level request into a ModelLlmInferenceRequest,
        delegates to HandlerLlmOpenaiCompatible, and translates the
        response back to the SPI-level format.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            Generated response with usage metrics.

        Raises:
            InfraConnectionError: If the provider cannot be reached.
            InfraTimeoutError: If the request times out.
        """
        infra_request = self._translate_request(request)
        infra_response = await self._handler.handle(infra_request)

        return ModelLlmAdapterResponse(
            generated_text=infra_response.generated_text or "",
            model_used=infra_response.model_used,
            usage_statistics={
                "prompt_tokens": infra_response.usage.tokens_input,
                "completion_tokens": infra_response.usage.tokens_output,
                "total_tokens": infra_response.usage.tokens_total or 0,
            },
            finish_reason=infra_response.finish_reason.value,
            response_metadata={
                "latency_ms": infra_response.latency_ms,
                "provider_id": infra_response.provider_id or "",
                "correlation_id": str(infra_response.correlation_id),
            },
        )

    def generate_stream(self, request: ModelLlmAdapterRequest) -> Iterator[str]:
        """Generate a streaming response (synchronous).

        Not supported for OpenAI-compatible adapter in v1. Raises
        NotImplementedError.

        Args:
            request: The LLM request.

        Raises:
            NotImplementedError: Streaming is not supported in v1.
        """
        raise NotImplementedError(
            "Synchronous streaming is not supported in v1. "
            "Use generate_stream_async for async streaming."
        )

    async def generate_stream_async(
        self,
        request: ModelLlmAdapterRequest,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response asynchronously.

        Not supported in v1. Falls back to non-streaming generation
        and yields the full response as a single chunk.

        Args:
            request: The LLM request.

        Yields:
            Generated text as a single chunk.
        """
        response = await self.generate_async(request)
        yield response.generated_text

    # ── Cost estimation ────────────────────────────────────────────────

    def estimate_cost(self, request: ModelLlmAdapterRequest) -> float:
        """Estimate the cost for this request.

        For local providers, returns 0.0. For external providers, estimates
        based on cached model capabilities.

        Args:
            request: The LLM request to estimate.

        Returns:
            Estimated cost in USD.
        """
        if self._provider_type_value == "local":
            return 0.0

        caps = self._capabilities_cache.get(request.model_name)
        if caps is None:
            return 0.0

        # Rough token estimation: ~4 chars per token
        estimated_input_tokens = len(request.prompt) / 4
        estimated_output_tokens = (request.max_tokens or 256) / 2

        input_cost = (estimated_input_tokens / 1000) * caps.cost_per_1k_input_tokens
        output_cost = (estimated_output_tokens / 1000) * caps.cost_per_1k_output_tokens

        return input_cost + output_cost

    # ── Health check ───────────────────────────────────────────────────

    async def health_check(self) -> ModelLlmHealthResponse:
        """Perform a health check on the provider.

        Attempts to reach the /v1/models endpoint. Updates internal
        availability state.

        Returns:
            Health check response with latency and available models.
        """
        start_time = time.perf_counter()
        try:
            models = await self.get_available_models()
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._is_available = True
            return ModelLlmHealthResponse(
                is_healthy=True,
                provider_name=self._provider_name_value,
                response_time_ms=latency_ms,
                available_models=models,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._is_available = False
            return ModelLlmHealthResponse(
                is_healthy=False,
                provider_name=self._provider_name_value,
                response_time_ms=latency_ms,
                error_message=str(exc),
            )

    # ── Provider info ──────────────────────────────────────────────────

    async def get_provider_info(self) -> JsonType:
        """Get comprehensive provider information.

        Returns:
            Dictionary with provider metadata.
        """
        return {
            "name": self._provider_name_value,
            "type": self._provider_type_value,
            "base_url": self._base_url,
            "default_model": self._default_model,
            "is_available": self._is_available,
            "supports_streaming": False,
            "supports_async": True,
        }

    # ── Feature support ────────────────────────────────────────────────

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming. Returns False for v1."""
        return False

    def supports_async(self) -> bool:
        """Check if provider supports async operations. Always True."""
        return True

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the HTTP transport client."""
        await self._transport.close()

    # ── Internal translation ───────────────────────────────────────────

    def _translate_request(
        self, request: ModelLlmAdapterRequest
    ) -> ModelLlmInferenceRequest:
        """Translate SPI request to infra-layer ModelLlmInferenceRequest.

        Bridges the structural mismatch between ProtocolLLMRequest.prompt
        (single string) and ModelLlmInferenceRequest's messages tuple.

        Args:
            request: SPI-level request.

        Returns:
            Infra-layer request ready for handler dispatch.
        """
        model_name = request.model_name or self._default_model

        return ModelLlmInferenceRequest(
            base_url=self._base_url,
            model=model_name,
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            messages=({"role": "user", "content": request.prompt},),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            api_key=self._api_key,
        )


__all__: list[str] = ["AdapterLlmProviderOpenai"]
