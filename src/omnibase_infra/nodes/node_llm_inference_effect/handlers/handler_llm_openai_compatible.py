# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OpenAI-compatible inference handler for the LLM Inference Effect node.

This handler translates between ONEX models and the OpenAI wire format
for inference calls. It supports both CHAT_COMPLETION and COMPLETION
operation types, tool calling, and Bearer token authentication.

Architecture:
    This handler follows the ONEX handler pattern:
    - Receives typed input (ModelLlmInferenceRequest)
    - Translates to OpenAI wire format
    - Delegates HTTP transport to MixinLlmHttpTransport
    - Parses the response into ModelLlmInferenceResponse
    - Maps provider finish reasons to EnumLlmFinishReason

Handler Responsibilities:
    - Build URL from base_url + operation_type path
    - Serialize request fields to OpenAI JSON payload
    - Translate ModelLlmToolChoice to wire format
    - Serialize ModelLlmToolDefinition to wire format
    - Parse response JSON into ModelLlmInferenceResponse
    - Map unknown finish_reason values to UNKNOWN (no crash)
    - Inject Authorization header when api_key is provided

Auth Strategy:
    When ``api_key`` is provided, the handler temporarily injects a
    dedicated ``httpx.AsyncClient`` with the ``Authorization: Bearer``
    header into the transport before calling ``_execute_llm_http_call``.
    The original client is restored after the call completes, even on
    error. When no ``api_key`` is provided, the transport's default
    client is used as-is.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances. The per-call auth client injection
    is scoped to each invocation.

Related Tickets:
    - OMN-2107: Phase 7 OpenAI-compatible inference handler
    - OMN-2104: MixinLlmHttpTransport (Phase 4)
    - OMN-2106: ModelLlmInferenceResponse (Phase 6)

See Also:
    - MixinLlmHttpTransport for HTTP call execution
    - ModelLlmInferenceResponse for output model
    - EnumLlmFinishReason for finish reason mapping
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)
from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall
from omnibase_infra.nodes.effects.models.model_llm_tool_choice import (
    ModelLlmToolChoice,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_definition import (
    ModelLlmToolDefinition,
)
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

logger = logging.getLogger(__name__)

# Mapping from OpenAI finish_reason strings to canonical enum values.
# Unknown values fall through to UNKNOWN.
_FINISH_REASON_MAP: dict[str, EnumLlmFinishReason] = {
    "stop": EnumLlmFinishReason.STOP,
    "length": EnumLlmFinishReason.LENGTH,
    "content_filter": EnumLlmFinishReason.CONTENT_FILTER,
    "tool_calls": EnumLlmFinishReason.TOOL_CALLS,
    "function_call": EnumLlmFinishReason.TOOL_CALLS,
}

# URL path suffixes for each operation type.
_OPERATION_PATHS: dict[EnumLlmOperationType, str] = {
    EnumLlmOperationType.CHAT_COMPLETION: "/v1/chat/completions",
    EnumLlmOperationType.COMPLETION: "/v1/completions",
}


class HandlerLlmOpenaiCompatible:
    """OpenAI wire-format handler for LLM inference calls.

    Translates between ONEX models (ModelLlmInferenceRequest) and the
    OpenAI-compatible JSON wire format used by OpenAI, vLLM, and other
    compatible inference servers.

    This handler does NOT extend MixinLlmHttpTransport directly. Instead,
    it receives a transport instance (any object that provides
    ``_execute_llm_http_call``) via constructor injection, following the
    ONEX handler pattern where handlers are stateless and transport-agnostic.

    Auth Strategy:
        When a request includes ``api_key``, the handler creates a
        temporary ``httpx.AsyncClient`` with the ``Authorization: Bearer``
        header and injects it into the transport for that single call.
        The original client reference is restored after the call, and
        the temporary client is closed. This avoids mutating shared
        mutable state on the transport's default client.

    Attributes:
        _transport: The LLM HTTP transport mixin instance for making calls.

    Example:
        >>> from unittest.mock import AsyncMock, MagicMock
        >>> transport = MagicMock(spec=MixinLlmHttpTransport)
        >>> handler = HandlerLlmOpenaiCompatible(transport)
    """

    def __init__(self, transport: MixinLlmHttpTransport) -> None:
        """Initialize handler with HTTP transport.

        Args:
            transport: An object providing ``_execute_llm_http_call`` for
                making HTTP POST requests to LLM endpoints. Typically a
                node or adapter that mixes in MixinLlmHttpTransport.
        """
        self._transport = transport

    async def handle(
        self,
        request: ModelLlmInferenceRequest,
        correlation_id: UUID,
    ) -> ModelLlmInferenceResponse:
        """Execute an LLM inference call using the OpenAI wire format.

        Translates the ONEX request model into an OpenAI-compatible JSON
        payload, executes the HTTP call via the transport mixin, and parses
        the response into a ModelLlmInferenceResponse.

        Args:
            request: The inference request with all parameters.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelLlmInferenceResponse with parsed results.

        Raises:
            InfraAuthenticationError: On 401/403 from the provider.
            InfraRateLimitedError: On 429 when retries are exhausted.
            InfraRequestRejectedError: On 400/422 from the provider.
            ProtocolConfigurationError: On 404 (misconfigured endpoint).
            InfraConnectionError: On connection failures after retries.
            InfraTimeoutError: On timeout after retries.
            InfraUnavailableError: On 5xx or circuit breaker open.
            ValueError: If operation_type has no known URL path.
        """
        start_time = time.perf_counter()
        execution_id = uuid4()

        # 1. Build URL
        url = self._build_url(request)

        # 2. Build payload
        payload = self._build_payload(request)

        # 3. Execute HTTP call via transport (with auth if needed)
        response_data = await self._execute_with_auth(
            url=url,
            payload=payload,
            api_key=request.api_key,
            correlation_id=correlation_id,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 4. Parse response
        return self._parse_response(
            data=response_data,
            request=request,
            correlation_id=correlation_id,
            execution_id=execution_id,
            latency_ms=latency_ms,
        )

    # ── URL building ─────────────────────────────────────────────────────

    @staticmethod
    def _build_url(request: ModelLlmInferenceRequest) -> str:
        """Build the full URL from base_url and operation type.

        Args:
            request: The inference request.

        Returns:
            Full URL string with the appropriate path suffix.

        Raises:
            ValueError: If operation_type is not CHAT_COMPLETION or COMPLETION.
        """
        path = _OPERATION_PATHS.get(request.operation_type)
        if path is None:
            msg = (
                f"Unsupported operation type for OpenAI handler: "
                f"{request.operation_type.value}"
            )
            raise ValueError(msg)

        base = request.base_url.rstrip("/")
        return f"{base}{path}"

    # ── Payload building ─────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        request: ModelLlmInferenceRequest,
    ) -> dict[str, JsonType]:
        """Build the OpenAI-compatible JSON payload.

        For CHAT_COMPLETION: builds a messages array with optional system
        prompt prepended. For COMPLETION: uses the prompt field.

        Args:
            request: The inference request.

        Returns:
            JSON-serializable payload dictionary.
        """
        payload: dict[str, JsonType] = {"model": request.model}

        if request.operation_type == EnumLlmOperationType.CHAT_COMPLETION:
            messages: list[dict[str, Any]] = []

            # Prepend system prompt as first system message
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})

            # Add user-provided messages
            messages.extend(dict(m) for m in request.messages)

            payload["messages"] = messages
        # COMPLETION mode
        elif request.prompt is not None:
            payload["prompt"] = request.prompt

        # Optional parameters -- only include if set
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = list(request.stop)

        # Tools
        if request.tools:
            payload["tools"] = [
                _serialize_tool_definition(tool) for tool in request.tools
            ]

        # Tool choice
        if request.tool_choice is not None:
            payload["tool_choice"] = _serialize_tool_choice(request.tool_choice)

        return payload

    # ── HTTP execution with auth ─────────────────────────────────────────

    async def _execute_with_auth(
        self,
        url: str,
        payload: dict[str, JsonType],
        api_key: str | None,
        correlation_id: UUID,
    ) -> dict[str, JsonType]:
        """Execute HTTP call via transport, injecting auth if needed.

        When ``api_key`` is provided, creates a temporary httpx client with
        the Authorization header and injects it into the transport for the
        duration of the call. The transport's original client reference is
        restored afterward.

        When ``api_key`` is None, delegates directly to the transport's
        ``_execute_llm_http_call``.

        Args:
            url: Full URL for the request.
            payload: JSON payload.
            api_key: Optional Bearer token. None means no auth.
            correlation_id: Correlation ID for tracing.

        Returns:
            Parsed JSON response dictionary.
        """
        if not api_key:
            return await self._transport._execute_llm_http_call(
                url=url,
                payload=payload,
                correlation_id=correlation_id,
            )

        # Create a temporary client with the auth header
        auth_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
        # Save the transport's original client and inject the auth client
        original_client = self._transport._http_client
        original_owns = self._transport._owns_http_client
        self._transport._http_client = auth_client
        self._transport._owns_http_client = False
        try:
            return await self._transport._execute_llm_http_call(
                url=url,
                payload=payload,
                correlation_id=correlation_id,
            )
        finally:
            # Restore the original client
            self._transport._http_client = original_client
            self._transport._owns_http_client = original_owns
            await auth_client.aclose()

    # ── Response parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_response(
        data: dict[str, JsonType],
        request: ModelLlmInferenceRequest,
        correlation_id: UUID,
        execution_id: UUID,
        latency_ms: float,
    ) -> ModelLlmInferenceResponse:
        """Parse an OpenAI-compatible JSON response into a ModelLlmInferenceResponse.

        Extracts the first choice's content, tool calls, usage, and finish
        reason from the response. Unknown finish_reason values are mapped
        to UNKNOWN to prevent crashes.

        Args:
            data: Parsed JSON response from the provider.
            request: The original inference request (for metadata).
            correlation_id: Correlation ID for tracing.
            execution_id: Unique execution identifier.
            latency_ms: End-to-end latency in milliseconds.

        Returns:
            ModelLlmInferenceResponse with parsed content.
        """
        # Extract provider ID
        provider_id = data.get("id")
        provider_id_str = str(provider_id) if provider_id is not None else None

        # Extract the first choice
        choices = data.get("choices", [])
        if not isinstance(choices, list) or len(choices) == 0:
            # No choices -- empty response
            return ModelLlmInferenceResponse(
                generated_text=None,
                model_used=request.model,
                operation_type=request.operation_type,
                finish_reason=EnumLlmFinishReason.UNKNOWN,
                usage=ModelLlmUsage(),
                latency_ms=latency_ms,
                backend_result=ModelBackendResult(success=True, duration_ms=latency_ms),
                correlation_id=correlation_id,
                execution_id=execution_id,
                timestamp=datetime.now(UTC),
                provider_id=provider_id_str,
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            # Malformed choice -- treat as empty
            return ModelLlmInferenceResponse(
                generated_text=None,
                model_used=request.model,
                operation_type=request.operation_type,
                finish_reason=EnumLlmFinishReason.UNKNOWN,
                usage=ModelLlmUsage(),
                latency_ms=latency_ms,
                backend_result=ModelBackendResult(success=True, duration_ms=latency_ms),
                correlation_id=correlation_id,
                execution_id=execution_id,
                timestamp=datetime.now(UTC),
                provider_id=provider_id_str,
            )

        # Parse finish reason (unknown -> UNKNOWN, no crash)
        raw_finish_reason = choice.get("finish_reason", "")
        finish_reason_str = str(raw_finish_reason) if raw_finish_reason else ""
        finish_reason = _FINISH_REASON_MAP.get(
            finish_reason_str, EnumLlmFinishReason.UNKNOWN
        )

        # Parse content and tool calls based on operation type
        generated_text: str | None = None
        tool_calls: tuple[ModelLlmToolCall, ...] = ()

        if request.operation_type == EnumLlmOperationType.CHAT_COMPLETION:
            message = choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                generated_text = str(content) if content is not None else None

                raw_tool_calls = message.get("tool_calls")
                if isinstance(raw_tool_calls, list) and raw_tool_calls:
                    tool_calls = _parse_tool_calls(raw_tool_calls)
                    # When we have tool calls, generated_text must be None
                    # (text XOR tool_calls invariant)
                    generated_text = None
        else:
            # COMPLETION mode -- text is in choice.text
            text = choice.get("text")
            generated_text = str(text) if text is not None else None

        # If we have tool calls, finish_reason should be TOOL_CALLS
        if tool_calls and finish_reason != EnumLlmFinishReason.TOOL_CALLS:
            finish_reason = EnumLlmFinishReason.TOOL_CALLS

        # Determine truncated flag
        truncated = finish_reason == EnumLlmFinishReason.LENGTH

        # Parse usage
        usage = _parse_usage(data.get("usage"))

        return ModelLlmInferenceResponse(
            generated_text=generated_text,
            tool_calls=tool_calls,
            model_used=request.model,
            provider_id=provider_id_str,
            operation_type=request.operation_type,
            finish_reason=finish_reason,
            truncated=truncated,
            usage=usage,
            latency_ms=latency_ms,
            backend_result=ModelBackendResult(
                success=True,
                duration_ms=latency_ms,
            ),
            correlation_id=correlation_id,
            execution_id=execution_id,
            timestamp=datetime.now(UTC),
        )


# ── Module-level helper functions ────────────────────────────────────────


def _serialize_tool_definition(
    tool: ModelLlmToolDefinition,
) -> dict[str, JsonType]:
    """Serialize a ModelLlmToolDefinition to OpenAI wire format.

    Args:
        tool: The tool definition to serialize.

    Returns:
        OpenAI-compatible tool definition dictionary.
    """
    func_dict: dict[str, JsonType] = {
        "name": tool.function.name,
    }

    if tool.function.description:
        func_dict["description"] = tool.function.description
    if tool.function.parameters:
        func_dict["parameters"] = tool.function.parameters

    return {
        "type": tool.type,
        "function": func_dict,
    }


def _serialize_tool_choice(
    choice: ModelLlmToolChoice,
) -> JsonType:
    """Translate ModelLlmToolChoice to OpenAI wire format.

    Translation:
        - mode="auto"     -> "auto"
        - mode="none"     -> "none"
        - mode="required" -> "required"
        - mode="function" -> {"type": "function", "function": {"name": "..."}}

    Args:
        choice: The tool choice constraint.

    Returns:
        Wire-format value (string or dict).
    """
    if choice.mode in ("auto", "none", "required"):
        return choice.mode

    # mode="function" -- must have function_name (enforced by model validator)
    return {
        "type": "function",
        "function": {"name": choice.function_name},
    }


def _parse_tool_calls(
    raw_calls: list[Any],
) -> tuple[ModelLlmToolCall, ...]:
    """Parse raw tool call dictionaries into ModelLlmToolCall instances.

    Skips malformed entries (missing id, function, or function.name)
    with a debug log rather than crashing.

    Args:
        raw_calls: List of tool call dictionaries from the response.

    Returns:
        Tuple of parsed ModelLlmToolCall instances.
    """
    parsed: list[ModelLlmToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            logger.debug("Skipping non-dict tool call entry: %s", type(raw).__name__)
            continue

        call_id = raw.get("id")
        function_data = raw.get("function")
        if not call_id or not isinstance(function_data, dict):
            logger.debug(
                "Skipping malformed tool call (missing id or function): %s",
                raw.get("id", "<no id>"),
            )
            continue

        func_name = function_data.get("name")
        if not func_name:
            logger.debug("Skipping tool call with missing function name")
            continue

        func_arguments = function_data.get("arguments", "")

        parsed.append(
            ModelLlmToolCall(
                id=str(call_id),
                function=ModelLlmFunctionCall(
                    name=str(func_name),
                    arguments=str(func_arguments),
                ),
            )
        )

    return tuple(parsed)


def _parse_usage(raw_usage: JsonType) -> ModelLlmUsage:
    """Parse the usage block from an OpenAI-compatible response.

    Handles missing or malformed usage data by defaulting to zeros.

    Args:
        raw_usage: The ``usage`` field from the response JSON, or None.

    Returns:
        ModelLlmUsage with parsed or default token counts.
    """
    if not isinstance(raw_usage, dict):
        return ModelLlmUsage()

    tokens_input = raw_usage.get("prompt_tokens", 0)
    tokens_output = raw_usage.get("completion_tokens", 0)
    tokens_total = raw_usage.get("total_tokens")

    return ModelLlmUsage(
        tokens_input=int(tokens_input) if tokens_input is not None else 0,
        tokens_output=int(tokens_output) if tokens_output is not None else 0,
        tokens_total=int(tokens_total) if tokens_total is not None else None,
    )


__all__: list[str] = ["HandlerLlmOpenaiCompatible"]
