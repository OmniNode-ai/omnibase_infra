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
    - Extracts and normalizes token usage (OMN-2238)
    - Builds ContractLlmCallMetrics for caller to publish

Handler Responsibilities:
    - Build URL from base_url + operation_type path
    - Serialize request fields to OpenAI JSON payload
    - Translate ModelLlmToolChoice to wire format
    - Serialize ModelLlmToolDefinition to wire format
    - Parse response JSON into ModelLlmInferenceResponse
    - Map unknown finish_reason values to UNKNOWN (no crash)
    - Inject Authorization header when api_key is provided
    - Extract token usage from API response (5 fallback cases)
    - Redact sensitive data from raw response before storage
    - Build per-call metrics (ContractLlmCallMetrics) for ``onex.evt.omniintelligence.llm-call-completed.v1``

Auth Strategy:
    When ``api_key`` is provided, the handler temporarily injects a
    dedicated ``httpx.AsyncClient`` with the ``Authorization: Bearer``
    header into the transport before calling ``_execute_llm_http_call``.
    The original client is restored after the call completes, even on
    error. When no ``api_key`` is provided, the transport's default
    client is used as-is.

Coroutine Safety:
    This handler is coroutine-safe for concurrent calls. When ``api_key``
    is provided, an ``asyncio.Lock`` attached to the transport instance
    serializes client injection so that all handler instances sharing the
    same transport are mutually excluded from swapping each other's clients.

Related Tickets:
    - OMN-2107: Phase 7 OpenAI-compatible inference handler
    - OMN-2104: MixinLlmHttpTransport (Phase 4)
    - OMN-2106: ModelLlmInferenceResponse (Phase 6)
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - OMN-2235: LLM cost tracking contracts (SPI layer)

See Also:
    - MixinLlmHttpTransport for HTTP call execution
    - ModelLlmInferenceResponse for output model
    - EnumLlmFinishReason for finish reason mapping
    - service_llm_usage_normalizer for normalization logic
    - ContractLlmCallMetrics for per-call metrics contract
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import httpx

from omnibase_core.types import JsonType
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLlmFinishReason,
    EnumLlmOperationType,
)
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
from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_usage_normalizer import (
    normalize_llm_usage,
)
from omnibase_spi.contracts.measurement import ContractEnumUsageSource
from omnibase_spi.contracts.measurement.contract_llm_call_metrics import (
    ContractLlmCallMetrics,
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

# Default connection limits for auth-injected httpx clients.
_AUTH_CLIENT_LIMITS: httpx.Limits = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
)

# Threading lock that guards lazy creation of the per-transport asyncio.Lock.
# This prevents the TOCTOU race where two coroutines both see hasattr() as False
# and each create a separate asyncio.Lock, with only the last assignment surviving.
_TRANSPORT_LOCK_GUARD: threading.Lock = threading.Lock()

# Default timeout for auth-injected httpx clients (seconds).
_DEFAULT_TIMEOUT_SECONDS: float = 30.0


class HandlerLlmOpenaiCompatible:
    """OpenAI wire-format handler for LLM inference calls.

    Translates between ONEX models (ModelLlmInferenceRequest) and the
    OpenAI-compatible JSON wire format used by OpenAI, vLLM, and other
    compatible inference servers.

    This handler does NOT extend MixinLlmHttpTransport directly. Instead,
    it receives a transport instance (any object that provides
    ``_execute_llm_http_call``) via constructor injection, following the
    ONEX handler pattern where handlers are stateless and transport-agnostic.

    Protocol Conformance Note:
        This handler intentionally does NOT implement ``ProtocolHandler``
        or ``ProtocolMessageHandler``. Those protocols operate on
        ``ModelOnexEnvelope`` and ``ModelEventEnvelope`` respectively,
        which are envelope-based dispatch interfaces for the node runtime.
        This handler operates at the infrastructure layer with a typed
        request model (``ModelLlmInferenceRequest``) and returns a typed
        response model (``ModelLlmInferenceResponse``), bypassing
        envelope-based dispatch entirely. LLM inference calls are direct
        infrastructure effects, not routed events. The ``handler_type``
        and ``handler_category`` properties are still provided for
        introspection and classification consistency.

    Auth Strategy:
        When a request includes ``api_key``, the handler creates a
        temporary ``httpx.AsyncClient`` with the ``Authorization: Bearer``
        header and injects it into the transport for that single call.
        The original client reference is restored after the call, and
        the temporary client is closed. This avoids mutating shared
        mutable state on the transport's default client.

    Attributes:
        _transport: The LLM HTTP transport mixin instance for making calls.
            An ``asyncio.Lock`` (``_auth_lock``) is lazily attached to the
            transport on first auth-injected call so that all handler
            instances sharing the same transport serialize their client swaps.
        last_call_metrics: The ``ContractLlmCallMetrics`` from the most recent
            ``handle()`` call, or ``None`` if metrics computation failed or
            ``handle()`` has not been called.

            .. warning:: **Not safe for concurrent access.**
                This attribute is mutable shared state on the handler instance.
                If two concurrent ``handle()`` calls interleave, the caller of
                the first call may read metrics from the second call. When
                sharing a handler instance across concurrent asyncio tasks,
                callers MUST use the metrics returned in the event output
                (``response.usage``) rather than relying on this attribute.
                This attribute exists as a convenience for single-caller
                sequential usage patterns only.

    Example:
        >>> from unittest.mock import AsyncMock, MagicMock
        >>> transport = MagicMock(spec=MixinLlmHttpTransport)
        >>> handler = HandlerLlmOpenaiCompatible(transport)
    """

    def __init__(
        self,
        transport: MixinLlmHttpTransport,
    ) -> None:
        """Initialize handler with HTTP transport.

        Args:
            transport: An object providing ``_execute_llm_http_call`` for
                making HTTP POST requests to LLM endpoints. Typically a
                node or adapter that mixes in MixinLlmHttpTransport.
        """
        self._transport = transport
        self.last_call_metrics: ContractLlmCallMetrics | None = None

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
            performs external I/O (HTTP calls to OpenAI-compatible APIs).
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelLlmInferenceRequest,
        correlation_id: UUID | None = None,
    ) -> ModelLlmInferenceResponse:
        """Execute an LLM inference call using the OpenAI wire format.

        Translates the ONEX request model into an OpenAI-compatible JSON
        payload, executes the HTTP call via the transport mixin, and parses
        the response into a ModelLlmInferenceResponse.

        Args:
            request: The inference request with all parameters.
            correlation_id: Correlation ID for distributed tracing. If None,
                a new UUID is auto-generated.

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
        if correlation_id is None:
            correlation_id = uuid4()

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
            timeout_seconds=request.timeout_seconds,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 4. Parse response
        response = self._parse_response(
            data=response_data,
            request=request,
            correlation_id=correlation_id,
            execution_id=execution_id,
            latency_ms=latency_ms,
        )

        # 5. Extract and normalize usage metrics (OMN-2238)
        # _build_usage_metrics returns the metrics directly so that
        # handle() uses the local return value. self.last_call_metrics
        # is also set as a convenience for sequential callers, but is
        # NOT safe for concurrent access (see class docstring).
        # Handlers MUST NOT publish events directly per ONEX handler
        # constraints.
        self.last_call_metrics = self._build_usage_metrics(
            raw_response=response_data,
            request=request,
            response=response,
            latency_ms=latency_ms,
        )

        return response

    # ── Usage metrics building ─────────────────────────────────────────

    def _build_usage_metrics(
        self,
        raw_response: dict[str, JsonType],
        request: ModelLlmInferenceRequest,
        response: ModelLlmInferenceResponse,
        latency_ms: float,
    ) -> ContractLlmCallMetrics | None:
        """Extract, normalize, and return LLM call metrics.

        Performs the following steps:
        1. Build prompt text for estimation fallback
        2. Normalize usage via the 5-case normalizer
        3. Build and return ContractLlmCallMetrics

        The caller (``handle()``) assigns the return value to
        ``self.last_call_metrics`` and may also use the returned value
        directly, avoiding the concurrency hazard of reading mutable
        instance state after an ``await`` boundary.

        The caller (node/dispatcher layer) is responsible for publishing the
        metrics event to the event bus. Handlers MUST NOT publish events
        directly per ONEX handler constraints.

        This method is fire-and-forget: errors in metrics building are
        logged but never propagated to the caller. LLM inference must
        not fail because metrics computation failed.

        Args:
            raw_response: The raw JSON response from the provider.
            request: The original inference request.
            response: The parsed inference response.
            latency_ms: End-to-end latency in milliseconds.

        Returns:
            The built metrics contract, or ``None`` if metrics computation
            failed.
        """
        try:
            # Build prompt text for estimation fallback.
            prompt_text = self._build_prompt_text(request)

            # Normalize usage (handles all 5 fallback cases).
            raw_usage, normalized = normalize_llm_usage(
                raw_response,
                provider="openai_compatible",
                generated_text=response.generated_text,
                prompt_text=prompt_text,
            )

            # Compute input hash for reproducibility tracking.
            input_hash = _compute_input_hash(request)

            # Build the metrics contract.
            metrics = ContractLlmCallMetrics(
                model_id=request.model,
                prompt_tokens=normalized.prompt_tokens,
                completion_tokens=normalized.completion_tokens,
                total_tokens=normalized.total_tokens,
                latency_ms=latency_ms,
                usage_raw=raw_usage,
                usage_normalized=normalized,
                usage_is_estimated=normalized.usage_is_estimated,
                input_hash=input_hash,
                timestamp_iso=datetime.now(UTC).isoformat(),
                reporting_source="handler-llm-openai-compatible",
            )

            logger.debug(
                "Built LLM call metrics. correlation_id=%s model=%s "
                "prompt_tokens=%d completion_tokens=%d source=%s",
                response.correlation_id,
                request.model,
                normalized.prompt_tokens,
                normalized.completion_tokens,
                normalized.source.value,
            )
            return metrics
        except Exception:
            # Metrics building must never break inference flow.
            logger.warning(
                "Failed to build LLM call metrics; ignoring. model=%s",
                request.model,
                exc_info=True,
            )
            return None

    @staticmethod
    def _build_prompt_text(request: ModelLlmInferenceRequest) -> str | None:
        """Build a prompt text string for token estimation fallback.

        For CHAT_COMPLETION: concatenates system_prompt and message contents.
        For COMPLETION: returns the prompt field directly.

        The text is used only for rough token estimation when the provider
        does not report token counts. It is never stored.

        Args:
            request: The inference request.

        Returns:
            Concatenated prompt text, or None if no text available.
        """
        if request.operation_type == EnumLlmOperationType.COMPLETION:
            return request.prompt

        parts: list[str] = []
        if request.system_prompt:
            parts.append(request.system_prompt)
        for msg in request.messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
        return " ".join(parts) if parts else None

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
            messages: list[JsonType] = []

            # Prepend system prompt as first system message
            if request.system_prompt:
                messages.append(
                    cast(
                        "dict[str, JsonType]",
                        {"role": "system", "content": request.system_prompt},
                    )
                )

            # Add user-provided messages
            messages.extend(
                cast("dict[str, JsonType]", dict(m)) for m in request.messages
            )

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
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
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
            timeout_seconds: HTTP request timeout in seconds for the
                auth-injected client. Sourced from the request model so
                callers can override the default (30.0) per-request.

        Returns:
            Parsed JSON response dictionary.

        Raises:
            ValueError: If api_key is an empty string (misconfiguration).
            InfraAuthenticationError: On 401/403 from the provider.
            InfraRateLimitedError: On 429 when retries are exhausted.
            InfraConnectionError: On connection failures after retries.
            InfraTimeoutError: On timeout after retries.
            InfraUnavailableError: On 5xx or circuit breaker open.
        """
        if api_key is None:
            return await self._transport._execute_llm_http_call(
                url=url,
                payload=payload,
                correlation_id=correlation_id,
            )

        if not api_key:
            msg = (
                "api_key is an empty string, which indicates misconfiguration. "
                "Provide a valid API key or omit api_key (set to None) to skip "
                "authentication."
            )
            raise ValueError(msg)

        # Lock lives on the transport so that ALL handler instances sharing
        # the same transport serialize their client-swap sections against each
        # other.  Created lazily on first use.  The threading lock guards the
        # hasattr + assignment to prevent a TOCTOU race where two coroutines
        # both evaluate hasattr() as False before either assigns the lock.
        with _TRANSPORT_LOCK_GUARD:
            if not hasattr(self._transport, "_auth_lock"):
                self._transport._auth_lock = asyncio.Lock()  # type: ignore[attr-defined]
        auth_lock: asyncio.Lock = self._transport._auth_lock  # type: ignore[attr-defined]

        async with auth_lock:
            auth_client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(timeout_seconds),
                limits=_AUTH_CLIENT_LIMITS,
            )
            # TECH DEBT: Direct access to _http_client and _owns_http_client
            # couples this handler to MixinLlmHttpTransport's private state.
            # The transport mixin (defined in omnibase_infra.mixins) should
            # expose a public context manager or method for auth-scoped calls,
            # e.g. ``async with transport.auth_scope(api_key) as scoped:``.
            # Until that API exists, we reach into private attributes here.
            # See: OMN-2104 (MixinLlmHttpTransport) for the transport layer.
            # Lock scope: auth_lock serializes auth-client swaps within this
            # handler.  No other codepath accesses _http_client concurrently.
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
                self._transport._http_client = original_client
                self._transport._owns_http_client = original_owns
                await auth_client.aclose()

    # ── Empty response builder ──────────────────────────────────────────

    @staticmethod
    def _build_empty_response(
        request: ModelLlmInferenceRequest,
        correlation_id: UUID,
        execution_id: UUID,
        latency_ms: float,
        provider_id_str: str | None,
    ) -> ModelLlmInferenceResponse:
        """Build a response for empty or malformed provider output.

        Used when the provider returns no choices array or a non-dict choice
        entry, so no content or tool calls can be extracted. The response
        is marked with ``finish_reason=UNKNOWN`` and empty usage.

        Args:
            request: The original inference request (for model and
                operation_type metadata).
            correlation_id: Correlation ID for distributed tracing.
            execution_id: Unique execution identifier.
            latency_ms: End-to-end latency in milliseconds.
            provider_id_str: Provider-assigned response ID, or None.

        Returns:
            A ModelLlmInferenceResponse with no generated text, unknown
            finish reason, and empty usage.
        """
        logger.warning(
            "Provider returned no usable choices; building empty response. "
            "correlation_id=%s model=%s provider_id=%s",
            correlation_id,
            request.model,
            provider_id_str,
        )
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

        When the ``choices`` array is empty or the first choice is malformed,
        delegates to ``_build_empty_response`` to return a well-formed
        response with ``finish_reason=UNKNOWN`` and empty usage.

        Args:
            data: Parsed JSON response from the provider.
            request: The original inference request (for metadata).
            correlation_id: Correlation ID for tracing.
            execution_id: Unique execution identifier.
            latency_ms: End-to-end latency in milliseconds.

        Returns:
            ModelLlmInferenceResponse with parsed content, or an empty
            response when the provider output is empty or malformed.
        """
        # Extract provider ID
        provider_id = data.get("id")
        provider_id_str = str(provider_id) if provider_id is not None else None

        # Extract the first choice
        choices = data.get("choices", [])
        if not isinstance(choices, list) or len(choices) == 0:
            # No choices -- empty response
            return HandlerLlmOpenaiCompatible._build_empty_response(
                request=request,
                correlation_id=correlation_id,
                execution_id=execution_id,
                latency_ms=latency_ms,
                provider_id_str=provider_id_str,
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            # Malformed choice -- treat as empty
            return HandlerLlmOpenaiCompatible._build_empty_response(
                request=request,
                correlation_id=correlation_id,
                execution_id=execution_id,
                latency_ms=latency_ms,
                provider_id_str=provider_id_str,
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
                    # Text XOR tool_calls invariant: clear any content text.
                    if generated_text is not None:
                        logger.debug(
                            "Discarding content text in favor of tool_calls "
                            "per text-XOR-tool_calls invariant. "
                            "correlation_id=%s discarded_length=%d",
                            correlation_id,
                            len(generated_text),
                        )
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
    raw_calls: list[JsonType],
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


def _safe_int(value: JsonType, default: int = 0) -> int:
    """Safely convert a JSON value to int, returning *default* on failure.

    Guards against ``ValueError`` when the value is a non-numeric string
    (e.g. ``"abc"``), which ``int()`` would reject.

    Args:
        value: A JSON-compatible value (int, float, str, or other).
        default: Fallback value when conversion is impossible.

    Returns:
        The integer conversion of *value*, or *default* if the conversion
        fails or the type is unsupported.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _safe_int_or_none(value: JsonType) -> int | None:
    """Safely convert a JSON value to int, returning ``None`` on failure.

    Identical to :func:`_safe_int` but returns ``None`` instead of a
    numeric default when the value is missing, unsupported, or
    non-numeric.

    Args:
        value: A JSON-compatible value (int, float, str, None, or other).

    Returns:
        The integer conversion of *value*, or ``None`` if the value is
        ``None``, an unsupported type, or a non-numeric string.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_usage(raw_usage: JsonType) -> ModelLlmUsage:
    """Parse the usage block from an OpenAI-compatible response.

    Handles missing or malformed usage data by defaulting to zeros.
    Non-numeric string values (e.g. ``"abc"``) are treated as zero
    (or ``None`` for ``tokens_total``) rather than raising ``ValueError``.

    When the provider returns a valid usage dict, ``usage_source`` is set
    to ``API`` and the raw dict is preserved in ``raw_provider_usage``.
    When usage data is absent or malformed, ``usage_source`` is ``MISSING``.

    Args:
        raw_usage: The ``usage`` field from the response JSON, or None.

    Returns:
        ModelLlmUsage with parsed or default token counts and provenance.
    """
    if not isinstance(raw_usage, dict):
        return ModelLlmUsage(
            usage_source=ContractEnumUsageSource.MISSING,
        )

    tokens_input = _safe_int(raw_usage.get("prompt_tokens", 0), 0)
    tokens_output = _safe_int(raw_usage.get("completion_tokens", 0), 0)
    tokens_total = _safe_int_or_none(raw_usage.get("total_tokens"))

    # Only mark as API-reported when at least one token counter is positive.
    # A usage dict with all-zero values (e.g. some providers return
    # {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    # is semantically equivalent to missing usage data.
    has_usage = (
        tokens_input > 0
        or tokens_output > 0
        or (tokens_total is not None and tokens_total > 0)
    )

    return ModelLlmUsage(
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_total=tokens_total,
        usage_source=(
            ContractEnumUsageSource.API
            if has_usage
            else ContractEnumUsageSource.MISSING
        ),
        raw_provider_usage=dict(raw_usage),
    )


def _compute_input_hash(request: ModelLlmInferenceRequest) -> str:
    """Compute a SHA-256 hash of the request input for reproducibility.

    The hash covers model, operation_type, messages/prompt, system_prompt,
    and generation parameters. It does NOT include api_key or base_url
    (infrastructure config, not semantic input).

    Args:
        request: The inference request.

    Returns:
        SHA-256 hex digest prefixed with ``sha256-``.
    """
    parts: list[str] = [
        request.model,
        request.operation_type.value,
    ]
    if request.prompt is not None:
        parts.append(request.prompt)
    if request.system_prompt is not None:
        parts.append(request.system_prompt)
    for msg in request.messages:
        parts.append(json.dumps(msg, sort_keys=True, default=str))
    if request.max_tokens is not None:
        parts.append(str(request.max_tokens))
    if request.temperature is not None:
        parts.append(str(request.temperature))
    if request.top_p is not None:
        parts.append(str(request.top_p))
    if request.stop:
        parts.append(json.dumps(list(request.stop), sort_keys=True, default=str))
    if request.tools:
        parts.append(
            json.dumps(
                [_serialize_tool_definition(t) for t in request.tools],
                sort_keys=True,
                default=str,
            )
        )
    if request.tool_choice is not None:
        parts.append(
            json.dumps(
                _serialize_tool_choice(request.tool_choice),
                sort_keys=True,
                default=str,
            )
        )

    combined = "|".join(parts)
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"sha256-{digest}"


__all__: list[str] = ["HandlerLlmOpenaiCompatible"]
