# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ollama LLM inference handler.

Implements the Ollama-specific HTTP transport for LLM inference operations.
Supports chat completions (``/api/chat``) and legacy completions
(``/api/generate``) with tool calling, generation parameters, and
structured response parsing.

Design:
    - Inherits ``MixinLlmHttpTransport`` for resilient HTTP calls with
      retry, circuit breaker, and typed error handling.
    - Enforces the text XOR tool_calls invariant before constructing
      the response model.
    - Maps Ollama's native response format to the platform-standard
      ``ModelLlmInferenceResponse``.
    - Pure serialization/parsing functions are module-level to keep the
      handler class focused on orchestration.

Unsupported Operations:
    - ``EMBEDDING``: Ollama embeddings use a different endpoint and response
      format. A dedicated embedding handler should be used instead.

Related:
    - MixinLlmHttpTransport: HTTP transport mixin with retry and circuit breaker
    - ModelLlmInferenceRequest: Input model for inference requests
    - ModelLlmInferenceResponse: Output model with text XOR tool_calls invariant
    - OMN-2108: Phase 8 Ollama inference handler

.. versionadded:: 0.8.0
    Part of OMN-2108 Ollama inference handler.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import ClassVar, cast
from uuid import UUID, uuid4

import httpx

from omnibase_core.types import JsonType
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumLlmFinishReason,
    EnumLlmOperationType,
)
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.mixins import MixinLlmHttpTransport
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models import (
    ModelLlmFunctionCall,
    ModelLlmInferenceRequest,
    ModelLlmInferenceResponse,
    ModelLlmMessage,
    ModelLlmToolCall,
    ModelLlmToolDefinition,
    ModelLlmUsage,
)
from omnibase_spi.contracts.measurement import ContractEnumUsageSource

logger = logging.getLogger(__name__)

# ── Finish reason mapping ────────────────────────────────────────────────

_OLLAMA_FINISH_REASON_MAP: dict[str, EnumLlmFinishReason] = {
    "stop": EnumLlmFinishReason.STOP,
    "length": EnumLlmFinishReason.LENGTH,
    "tool_calls": EnumLlmFinishReason.TOOL_CALLS,
    "content_filter": EnumLlmFinishReason.CONTENT_FILTER,
}


# ── Module-level pure functions ──────────────────────────────────────────


def _build_ollama_options(request: ModelLlmInferenceRequest) -> dict[str, JsonType]:
    """Build the Ollama options dict from request generation parameters.

    Only includes non-None and non-empty values to avoid overriding
    Ollama's defaults.

    Args:
        request: The LLM inference request.

    Returns:
        Options dictionary (may be empty).
    """
    options: dict[str, JsonType] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.stop:  # non-empty tuple only
        options["stop"] = list(request.stop)
    return options


def _serialize_ollama_messages(
    messages: tuple[ModelLlmMessage, ...],
) -> list[dict[str, JsonType]]:
    """Serialize chat messages to Ollama's expected format.

    Args:
        messages: Tuple of chat messages from the request.

    Returns:
        List of message dictionaries for the Ollama API.
    """
    result: list[dict[str, JsonType]] = []
    for msg in messages:
        m: dict[str, JsonType] = {"role": msg.role}
        if msg.content is not None:
            m["content"] = msg.content
        if msg.tool_calls:
            serialized_tool_calls: list[dict[str, JsonType]] = []
            for tc in msg.tool_calls:
                try:
                    parsed_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ProtocolConfigurationError(
                        f"Malformed JSON in tool call arguments for "
                        f"function '{tc.function.name}': "
                        f"{tc.function.arguments!r}",
                        context=ModelInfraErrorContext.with_correlation(
                            transport_type=EnumInfraTransportType.HTTP,
                            operation="serialize_ollama_messages",
                        ),
                    ) from exc
                serialized_tool_calls.append(
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": cast("JsonType", parsed_args),
                        },
                    }
                )
            m["tool_calls"] = cast("JsonType", serialized_tool_calls)
        if msg.tool_call_id is not None:
            m["tool_call_id"] = msg.tool_call_id
        result.append(m)
    return result


def _serialize_ollama_tools(
    tools: tuple[ModelLlmToolDefinition, ...],
) -> list[dict[str, JsonType]]:
    """Serialize tool definitions to Ollama's expected format.

    Args:
        tools: Tuple of tool definitions from the request.

    Returns:
        List of tool definition dictionaries for the Ollama API.
    """
    return [
        {
            "type": "function",
            "function": cast(
                "JsonType",
                {
                    "name": tool.function.name,
                    "description": tool.function.description or "",
                    "parameters": tool.function.parameters or {},
                },
            ),
        }
        for tool in tools
    ]


def _parse_ollama_tool_calls(
    raw_tool_calls: list[dict[str, JsonType]] | None,
    correlation_id: UUID,
) -> tuple[ModelLlmToolCall, ...]:
    """Parse raw tool calls from the Ollama response.

    Handles Ollama's response format where:
    - Tool call IDs may be missing (generated deterministically).
    - Arguments may be a dict (serialized to compact JSON), None
      (replaced with ``"{}"``), or a string (passed through).

    Args:
        raw_tool_calls: Raw tool call list from the response, or None.
        correlation_id: Correlation ID for generating deterministic
            tool call IDs when Ollama omits them.

    Returns:
        Tuple of parsed tool calls (empty if none).
    """
    if not raw_tool_calls:
        return ()
    result: list[ModelLlmToolCall] = []
    for i, raw_tc in enumerate(raw_tool_calls):
        func_raw = raw_tc.get("function", {})
        func = func_raw if isinstance(func_raw, dict) else {}
        name = str(func.get("name", ""))
        if not name:
            logger.warning("Tool call has empty function name, raw=%r", raw_tc)
            continue
        raw_args = func.get("arguments")
        # Handle arguments: dict->JSON string, None->"{}", string->passthrough
        if raw_args is None:
            arguments = "{}"
        elif isinstance(raw_args, (dict, list)):
            arguments = json.dumps(raw_args, separators=(",", ":"))
        else:
            arguments = str(raw_args)
        # Generate ID if missing (Ollama often omits it)
        raw_id = raw_tc.get("id")
        tc_id = str(raw_id) if raw_id else f"ollama-{correlation_id.hex[:8]}-{i}"
        result.append(
            ModelLlmToolCall(
                id=tc_id,
                function=ModelLlmFunctionCall(name=name, arguments=arguments),
            )
        )
    return tuple(result)


# ── Handler class ────────────────────────────────────────────────────────


class HandlerLlmOllama(MixinLlmHttpTransport):
    """Ollama LLM inference handler.

    Translates ``ModelLlmInferenceRequest`` into Ollama HTTP API calls and
    parses the response into ``ModelLlmInferenceResponse``.

    Supported operation types:
        - ``CHAT_COMPLETION``: Uses ``/api/chat`` endpoint.
        - ``COMPLETION``: Uses ``/api/generate`` endpoint.

    Unsupported:
        - ``EMBEDDING``: Raises ``ProtocolConfigurationError``.

    Example:
        >>> handler = HandlerLlmOllama()
        >>> response = await handler.handle(request)
        >>> await handler.close()
    """

    _FINISH_REASON_MAP: ClassVar[dict[str, EnumLlmFinishReason]] = (
        _OLLAMA_FINISH_REASON_MAP
    )

    def __init__(
        self,
        target_name: str = "ollama",
        max_timeout_seconds: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Ollama handler with HTTP transport.

        Args:
            target_name: Identifier for the Ollama target used in error
                context and logging. Default: ``"ollama"``.
            max_timeout_seconds: Maximum allowed timeout for any single
                request. Per-call timeouts are clamped to this value.
                Default: 120.0.
            http_client: Optional pre-configured ``httpx.AsyncClient``.
                If ``None``, a client is created lazily on first use.
                When provided, the caller retains ownership and must
                close it.
        """
        self._init_llm_http_transport(
            target_name=target_name,
            max_timeout_seconds=max_timeout_seconds,
            http_client=http_client,
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification.

        Returns:
            ``EnumHandlerType.INFRA_HANDLER`` indicating this handler
            provides infrastructure-level LLM transport, not domain
            business logic.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification.

        Returns:
            ``EnumHandlerTypeCategory.EFFECT`` indicating this handler
            performs external I/O (HTTP calls to the Ollama API).
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelLlmInferenceRequest
    ) -> ModelLlmInferenceResponse:
        """Execute an LLM inference request against the Ollama API.

        Routes to the appropriate Ollama endpoint based on operation type,
        builds the provider-specific payload, executes the HTTP call with
        retry and circuit breaker protection, and parses the response into
        a platform-standard ``ModelLlmInferenceResponse``.

        Args:
            request: The LLM inference request containing model, messages,
                generation parameters, and tracing metadata.

        Returns:
            A ``ModelLlmInferenceResponse`` with the inference result.

        Raises:
            ProtocolConfigurationError: If operation_type is EMBEDDING.
            InfraConnectionError: On connection failures after retries.
            InfraTimeoutError: On timeout after retries.
            InfraUnavailableError: On 5xx or circuit breaker open.
        """
        correlation_id: UUID = request.correlation_id or uuid4()

        # Validate operation type
        if request.operation_type == EnumLlmOperationType.EMBEDDING:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="ollama_inference",
            )
            raise ProtocolConfigurationError(
                "Ollama handler does not support EMBEDDING operations; "
                "use a dedicated embedding handler",
                context=context,
            )

        # Build URL and payload based on operation type
        if request.operation_type == EnumLlmOperationType.CHAT_COMPLETION:
            url = f"{request.base_url.rstrip('/')}/api/chat"
            payload = self._build_chat_payload(request)
        elif request.operation_type == EnumLlmOperationType.COMPLETION:
            url = f"{request.base_url.rstrip('/')}/api/generate"
            payload = self._build_generate_payload(request)
        else:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="ollama_inference",
            )
            raise ProtocolConfigurationError(
                f"Unsupported operation type for Ollama handler: "
                f"{request.operation_type!r}",
                context=context,
            )

        # Execute HTTP call — time ONLY the HTTP call
        start_time = time.perf_counter()
        raw_response = await self._execute_llm_http_call(
            url=url,
            payload=payload,
            correlation_id=correlation_id,
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse response based on operation type
        if request.operation_type == EnumLlmOperationType.CHAT_COMPLETION:
            message_raw = raw_response.get("message", {})
            message = message_raw if isinstance(message_raw, dict) else {}
            content = message.get("content")
            if content is not None:
                if not isinstance(content, str):
                    logger.warning(
                        "Unexpected content type from Ollama: %s, converting to string",
                        type(content).__name__,
                    )
                    content = str(content)
            raw_tool_calls_val = message.get("tool_calls")
            raw_tool_calls = (
                cast("list[dict[str, JsonType]]", raw_tool_calls_val)
                if isinstance(raw_tool_calls_val, list)
                else None
            )
        elif request.operation_type == EnumLlmOperationType.COMPLETION:
            raw_content = raw_response.get("response")
            content = str(raw_content) if raw_content is not None else None
            raw_tool_calls = None
        else:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="ollama_response_parsing",
            )
            raise ProtocolConfigurationError(
                f"Unsupported operation type for Ollama response parsing: "
                f"{request.operation_type!r}",
                context=context,
            )

        # Parse usage
        tokens_output = raw_response.get("eval_count", 0)
        tokens_input = raw_response.get("prompt_eval_count", 0)
        if not isinstance(tokens_input, (int, float)):
            logger.debug(
                "Non-numeric usage value for prompt_eval_count, "
                "defaulting to 0: type=%s",
                type(tokens_input).__name__,
            )
        if not isinstance(tokens_output, (int, float)):
            logger.debug(
                "Non-numeric usage value for eval_count, defaulting to 0: type=%s",
                type(tokens_output).__name__,
            )
        resolved_input = (
            round(tokens_input) if isinstance(tokens_input, (int, float)) else 0
        )
        resolved_output = (
            round(tokens_output) if isinstance(tokens_output, (int, float)) else 0
        )
        # Determine provenance: Ollama reports eval_count/prompt_eval_count
        # when the model actually ran; treat as API-reported when present.
        has_usage = isinstance(tokens_input, (int, float)) or isinstance(
            tokens_output, (int, float)
        )
        raw_usage_data: dict[str, object] = {
            "prompt_eval_count": raw_response.get("prompt_eval_count"),
            "eval_count": raw_response.get("eval_count"),
            "total_duration": raw_response.get("total_duration"),
            "load_duration": raw_response.get("load_duration"),
            "eval_duration": raw_response.get("eval_duration"),
        }
        usage = ModelLlmUsage(
            tokens_input=resolved_input,
            tokens_output=resolved_output,
            usage_source=(
                ContractEnumUsageSource.API
                if has_usage
                else ContractEnumUsageSource.MISSING
            ),
            raw_provider_usage=raw_usage_data,
        )

        # Enforce text XOR tool_calls BEFORE constructing response
        tool_calls = _parse_ollama_tool_calls(raw_tool_calls, correlation_id)
        if tool_calls:
            if content:
                logger.warning(
                    "Discarding non-empty text content in favor of tool_calls "
                    "to satisfy text-XOR-tool_calls invariant: "
                    "correlation_id=%s, content_length=%d",
                    correlation_id,
                    len(content),
                )
            generated_text = None  # tool calls present -> no text
            finish_reason = EnumLlmFinishReason.TOOL_CALLS
        else:
            generated_text = content  # may be None or "" -- both valid
            raw_done_reason = raw_response.get("done_reason")
            finish_reason = self._map_finish_reason(
                str(raw_done_reason) if raw_done_reason is not None else None
            )

        # Extract model_used with proper type narrowing
        raw_model = raw_response.get("model")
        model_used = str(raw_model) if isinstance(raw_model, str) else request.model

        return ModelLlmInferenceResponse(
            generated_text=generated_text,
            tool_calls=tool_calls,
            model_used=model_used,
            provider_id="ollama",
            operation_type=request.operation_type,
            finish_reason=finish_reason,
            truncated=(finish_reason == EnumLlmFinishReason.LENGTH),
            usage=usage,
            latency_ms=latency_ms,
            # TODO(OMN-2108): retry_count not exposed by MixinLlmHttpTransport — always 0
            retry_count=0,
            backend_result=ModelBackendResult(success=True, duration_ms=latency_ms),
            correlation_id=correlation_id,
            execution_id=request.execution_id,
            timestamp=datetime.now(UTC),
        )

    async def close(self) -> None:
        """Close the HTTP client if owned by this handler.

        Delegates to ``MixinLlmHttpTransport._close_http_client()``.
        """
        await self._close_http_client()

    # ── Payload builders ─────────────────────────────────────────────────

    def _build_chat_payload(
        self, request: ModelLlmInferenceRequest
    ) -> dict[str, JsonType]:
        """Build the Ollama ``/api/chat`` request payload.

        Args:
            request: The LLM inference request.

        Returns:
            JSON-serializable payload dictionary.
        """
        payload: dict[str, JsonType] = {
            "model": request.model,
            "messages": cast("JsonType", _serialize_ollama_messages(request.messages)),
            "stream": False,
        }
        options = _build_ollama_options(request)
        if options:
            payload["options"] = options
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.tools:
            payload["tools"] = cast("JsonType", _serialize_ollama_tools(request.tools))
        return payload

    def _build_generate_payload(
        self, request: ModelLlmInferenceRequest
    ) -> dict[str, JsonType]:
        """Build the Ollama ``/api/generate`` request payload.

        Note: This method is only called for COMPLETION operations.
        ``ModelLlmInferenceRequest`` validates that ``system_prompt`` is
        ``None`` for COMPLETION, so no system prompt handling is needed here.

        Args:
            request: The LLM inference request.

        Returns:
            JSON-serializable payload dictionary.
        """
        assert request.prompt is not None, (
            "prompt must be non-None for COMPLETION operation type"
        )
        payload: dict[str, JsonType] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
        }
        options = _build_ollama_options(request)
        if options:
            payload["options"] = options
        return payload

    def _map_finish_reason(self, raw: str | None) -> EnumLlmFinishReason:
        """Map an Ollama finish reason string to the platform enum.

        Args:
            raw: Raw finish reason from the Ollama response, or None.

        Returns:
            Mapped ``EnumLlmFinishReason``, defaulting to ``UNKNOWN``.
        """
        if raw is None:
            return EnumLlmFinishReason.UNKNOWN
        return self._FINISH_REASON_MAP.get(raw, EnumLlmFinishReason.UNKNOWN)


__all__: list[str] = ["HandlerLlmOllama"]
