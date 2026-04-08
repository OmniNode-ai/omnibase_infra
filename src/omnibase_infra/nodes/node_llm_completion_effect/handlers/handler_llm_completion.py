# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that sends chat completions to an OpenAI-compatible LLM endpoint.

This is an EFFECT handler -- it performs I/O (HTTP POST to LLM API).

Ported from archive: ai-dev/containers/autogen_litellm.
"""

from __future__ import annotations

import logging
import os

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_llm_completion_effect.models.model_llm_completion_request import (
    ModelLLMCompletionRequest,
)
from omnibase_infra.nodes.node_llm_completion_effect.models.model_llm_completion_result import (
    ModelLLMCompletionResult,
)

logger = logging.getLogger(__name__)

# Token threshold for routing to the fast (mid-tier) model vs full model.
_FAST_MODEL_TOKEN_THRESHOLD = 40_000


class HandlerLLMCompletion:
    """Sends a chat completion request to an OpenAI-compatible endpoint.

    Routing logic:
    - If ``endpoint_url`` is set on the request, use it directly.
    - Otherwise, estimate token count from messages and route to
      ``LLM_CODER_FAST_URL`` (<= 24K tokens) or ``LLM_CODER_URL`` (> 24K).
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelLLMCompletionRequest,
    ) -> ModelLLMCompletionResult:
        """Execute a chat completion against an OpenAI-compatible endpoint.

        Args:
            request: The completion request with messages and parameters.

        Returns:
            ModelLLMCompletionResult with the generated text or error.
        """
        endpoint = self._resolve_endpoint(request)
        logger.info(
            "LLM completion | endpoint=%s model=%s correlation_id=%s",
            endpoint,
            request.model,
            request.correlation_id,
        )

        payload: dict[str, object] = {
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.model:
            payload["model"] = request.model

        try:
            response = await self._http_client.post(
                f"{endpoint}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "LLM endpoint returned %s | correlation_id=%s",
                exc.response.status_code,
                request.correlation_id,
            )
            return ModelLLMCompletionResult(
                correlation_id=request.correlation_id,
                success=False,
                error_message=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except Exception as exc:
            logger.exception(
                "LLM completion failed | correlation_id=%s", request.correlation_id
            )
            return ModelLLMCompletionResult(
                correlation_id=request.correlation_id,
                success=False,
                error_message=str(exc)[:500],
            )

        # Parse OpenAI-compatible response
        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage", {})

        return ModelLLMCompletionResult(
            correlation_id=request.correlation_id,
            success=True,
            content=content,
            model=data.get("model", request.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def _resolve_endpoint(self, request: ModelLLMCompletionRequest) -> str:
        """Pick the LLM endpoint URL based on request or env configuration."""
        if request.endpoint_url:
            return request.endpoint_url.rstrip("/")

        # Estimate token count from message character length (rough: 4 chars/token)
        char_count = sum(len(m.content) for m in request.messages)
        estimated_tokens = char_count // 4

        if estimated_tokens <= _FAST_MODEL_TOKEN_THRESHOLD:
            url = os.environ.get(  # ONEX_EXCLUDE: archive port
                "LLM_CODER_FAST_URL", ""
            )
            if url:
                return url.rstrip("/")

        url = os.environ.get(  # ONEX_EXCLUDE: archive port
            "LLM_CODER_URL", ""
        )
        if url:
            return url.rstrip("/")

        # Last resort fallback
        return "http://localhost:8000"

    async def close(self) -> None:
        """Release HTTP resources if owned by this handler."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
