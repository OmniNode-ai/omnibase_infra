#!/usr/bin/env python3
"""
NodeLLMEffect - ONEX v2.0 Effect node for LLM API calls.

Supports 3-tier LLM system:
- LOCAL: Ollama/vLLM (future-ready, not implemented in Phase 1)
- CLOUD_FAST: GLM-4.5 via Z.ai (baseline tier, Phase 1 implementation)
- CLOUD_PREMIUM: GLM-4.6 via Z.ai (future-ready, not implemented in Phase 1)

Features:
- Circuit breaker pattern for resilience
- Retry logic with exponential backoff
- Token usage and cost tracking
- Comprehensive metrics collection
- ONEX v2.0 compliant error handling

Performance Targets:
- Latency: < 3000ms for CLOUD_FAST tier (P95)
- Throughput: 10+ concurrent requests
- Cost tracking: Sub-cent accuracy
- Retry success rate: > 90% for transient failures

Example Usage:
    ```python
    import os
    from omnibase_core.models.core import ModelContainer
    from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

    # Set environment variables (or load from .env file)
    os.environ["ZAI_API_KEY"] = "your_api_key"  # pragma: allowlist secret
    os.environ["ZAI_ENDPOINT"] = "https://api.z.ai/api/anthropic"

    # Initialize node (credentials read from environment)
    container = ModelContainer(value={}, container_type="config")
    node = NodeLLMEffect(container)

    # Generate text
    contract = ModelContractEffect(
        name="llm_generation",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Generate Python code",
        node_type="EFFECT",
        input_model="ModelLLMRequest",
        output_model="ModelLLMResponse",
        input_data={
            "prompt": "Generate a Python function to calculate Fibonacci",
            "tier": "CLOUD_FAST",
            "max_tokens": 2000,
            "temperature": 0.7,
            "operation_type": "node_generation",
        }
    )

    response = await node.execute_effect(contract)
    print(f"Generated: {response.generated_text[:100]}...")
    print(f"Cost: ${response.cost_usd:.6f}")
    ```
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.model_circuit_breaker import ModelCircuitBreaker
from omnibase_core.nodes.node_effect import NodeEffect

from .models import EnumLLMTier, ModelLLMConfig, ModelLLMRequest, ModelLLMResponse

# Aliases for compatibility
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode

logger = logging.getLogger(__name__)


class NodeLLMEffect(NodeEffect):
    """
    LLM Effect Node for multi-tier LLM API calls.

    Tier Configuration:
    - LOCAL: Not implemented yet (future Ollama/vLLM support)
    - CLOUD_FAST: GLM-4.5 via Z.ai (128K context, PRIMARY for Phase 1)
    - CLOUD_PREMIUM: GLM-4.6 via Z.ai (128K context, future)

    Circuit Breaker:
    - Failure threshold: 5 consecutive failures
    - Recovery timeout: 60 seconds
    - Protected: All external LLM API calls

    Retry Policy:
    - Max attempts: 3
    - Backoff: Exponential (1s, 2s, 4s)
    - Retryable errors: Timeout, 429 (rate limit), 5xx (server errors)

    Performance:
    - CLOUD_FAST latency: < 3000ms (P95)
    - Throughput: 10+ concurrent requests
    - Cost tracking: Sub-cent accuracy
    """

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize LLM Effect node with tier configuration.

        Args:
            container: ONEX container for dependency injection

        Note:
            API credentials are ALWAYS read from environment variables:
            - ZAI_API_KEY: Z.ai API key
            - ZAI_ENDPOINT: Z.ai API endpoint (optional, defaults to standard endpoint)
        """
        # Initialize base NodeEffect class
        super().__init__(container)

        import os

        # Extract non-sensitive configuration from container
        config_data = container.value if isinstance(container.value, dict) else {}

        # ALWAYS read credentials from environment variables (NEVER from container)
        # This prevents accidental logging/exposure of secrets
        # TODO: Phase 2 - Add Vault integration for secret management
        #   - Try Vault first (if VAULT_ADDR configured)
        #   - Fallback to environment variables
        #   - Path: secret/data/omninode/zai_credentials
        self.config = ModelLLMConfig(
            zai_api_key=os.getenv("ZAI_API_KEY"),
            zai_base_url=os.getenv("ZAI_ENDPOINT"),
            # Allow non-sensitive config overrides from container
            circuit_breaker_threshold=config_data.get("circuit_breaker_threshold", 5),
            circuit_breaker_timeout_seconds=config_data.get(
                "circuit_breaker_timeout_seconds", 60.0
            ),
            max_retry_attempts=config_data.get("max_retry_attempts", 3),
            http_timeout_seconds=config_data.get("http_timeout_seconds", 60.0),
        )

        # Validate API key presence
        if not self.config.zai_api_key:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message="ZAI_API_KEY is required but not provided in config or environment",
                details={"config_keys": list(config_data.keys())},
            )

        # Model configuration per tier
        self.tier_models = {
            EnumLLMTier.LOCAL: self.config.default_model_local,
            EnumLLMTier.CLOUD_FAST: self.config.default_model_cloud_fast,
            EnumLLMTier.CLOUD_PREMIUM: self.config.default_model_cloud_premium,
        }

        # Context windows (tokens)
        self.context_windows = {
            EnumLLMTier.LOCAL: self.config.context_window_local,
            EnumLLMTier.CLOUD_FAST: self.config.context_window_cloud_fast,
            EnumLLMTier.CLOUD_PREMIUM: self.config.context_window_cloud_premium,
        }

        # Cost per 1M tokens (USD) - Z.ai GLM pricing
        self.cost_per_1m_input = {
            EnumLLMTier.CLOUD_FAST: self.config.cost_per_1m_input_cloud_fast,
            EnumLLMTier.CLOUD_PREMIUM: self.config.cost_per_1m_input_cloud_premium,
        }
        self.cost_per_1m_output = {
            EnumLLMTier.CLOUD_FAST: self.config.cost_per_1m_output_cloud_fast,
            EnumLLMTier.CLOUD_PREMIUM: self.config.cost_per_1m_output_cloud_premium,
        }

        # Initialize standard circuit breaker from omnibase_core
        self.circuit_breaker = ModelCircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout_seconds=int(self.config.circuit_breaker_timeout_seconds),
        )

        # HTTP client (persistent connection pooling)
        self.http_client: httpx.AsyncClient | None = None

        emit_log_event(
            LogLevel.INFO,
            "NodeLLMEffect initialized",
            {
                "node_id": str(self.node_id),
                "supported_tiers": [tier.value for tier in EnumLLMTier],
                "primary_tier": EnumLLMTier.CLOUD_FAST.value,
                "zai_base_url": self.config.zai_base_url,
            },
        )

    async def initialize(self) -> None:
        """
        Initialize HTTP client for API calls.

        Creates persistent HTTP client with connection pooling.
        """
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.http_timeout_seconds),
                limits=httpx.Limits(max_connections=self.config.http_max_connections),
            )

            emit_log_event(
                LogLevel.INFO,
                "HTTP client initialized",
                {
                    "node_id": str(self.node_id),
                    "timeout": self.config.http_timeout_seconds,
                    "max_connections": self.config.http_max_connections,
                },
            )

    async def cleanup(self) -> None:
        """
        Cleanup HTTP client and resources.

        Closes HTTP client connection pool.
        """
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

            emit_log_event(
                LogLevel.INFO,
                "HTTP client closed",
                {"node_id": str(self.node_id)},
            )

    async def execute_effect(self, contract: ModelContractEffect) -> ModelLLMResponse:
        """
        Execute LLM generation request.

        Args:
            contract: Effect contract with input_state containing ModelLLMRequest

        Returns:
            ModelLLMResponse with generated text and metrics

        Raises:
            ModelOnexError: On LLM API failures or validation errors
        """
        start_time = time.time()

        # Extract request from contract
        # Support both input_state (new) and input_data (legacy stub)
        input_data = getattr(contract, "input_state", None) or getattr(
            contract, "input_data", None
        )
        if input_data is None:
            raise ModelOnexError(
                code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="Contract missing both input_state and input_data",
                details={"contract_name": contract.name},
            )
        request = ModelLLMRequest.model_validate(input_data)
        correlation_id = request.correlation_id or uuid4()

        emit_log_event(
            LogLevel.INFO,
            "Executing LLM generation",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
                "tier": request.tier.value,
                "operation_type": request.operation_type,
                "max_tokens": request.max_tokens,
            },
        )

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE,
                message="Circuit breaker is OPEN - LLM service temporarily unavailable",
                details={
                    "failures": self.circuit_breaker.failure_count,
                    "threshold": self.circuit_breaker.failure_threshold,
                    "retry_after_seconds": self.circuit_breaker.recovery_timeout_seconds,
                },
            )

        try:
            # Initialize HTTP client if needed
            if self.http_client is None:
                await self.initialize()

            # Route to appropriate tier
            if request.tier == EnumLLMTier.CLOUD_FAST:
                response = await self._generate_cloud_fast(request, correlation_id)
            elif request.tier == EnumLLMTier.LOCAL:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.UNSUPPORTED_OPERATION,
                    message="LOCAL tier not implemented yet (future Ollama/vLLM support)",
                    details={"requested_tier": request.tier.value},
                )
            elif request.tier == EnumLLMTier.CLOUD_PREMIUM:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.UNSUPPORTED_OPERATION,
                    message="CLOUD_PREMIUM tier not implemented yet (future GLM-4.6)",
                    details={"requested_tier": request.tier.value},
                )
            else:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message=f"Unknown LLM tier: {request.tier}",
                    details={"requested_tier": request.tier.value},
                )

            # Success - record in circuit breaker
            self.circuit_breaker.record_success()

            latency_ms = (time.time() - start_time) * 1000
            emit_log_event(
                LogLevel.INFO,
                "LLM generation completed",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "latency_ms": latency_ms,
                    "tokens_total": response.tokens_total,
                    "cost_usd": response.cost_usd,
                    "finish_reason": response.finish_reason,
                },
            )

            return response

        except Exception as e:
            # Record failure in circuit breaker
            self.circuit_breaker.record_failure()

            emit_log_event(
                LogLevel.ERROR,
                f"LLM generation failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            if isinstance(e, ModelOnexError):
                raise

            raise ModelOnexError(
                code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"LLM generation failed: {e}",
                details={"original_error": str(e), "error_type": type(e).__name__},
            )

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def _generate_cloud_fast(
        self, request: ModelLLMRequest, correlation_id: uuid4
    ) -> ModelLLMResponse:
        """
        Generate using CLOUD_FAST tier (GLM-4.5 via Z.ai).

        Implements retry logic with exponential backoff.

        Args:
            request: LLM generation request
            correlation_id: Correlation ID for tracking

        Returns:
            ModelLLMResponse with generated text and metrics

        Raises:
            ModelOnexError: On API failures after retries
        """
        model = request.model_override or self.tier_models[EnumLLMTier.CLOUD_FAST]

        # Build Z.ai request (Anthropic Messages API format)
        # Anthropic Messages API doesn't have separate system role, use messages only
        messages = []
        if request.system_prompt:
            # Add system message as first user message with context
            messages.append(
                {
                    "role": "user",
                    "content": f"System instructions: {request.system_prompt}\n\nUser request: {request.prompt}",
                }
            )
        else:
            messages.append({"role": "user", "content": request.prompt})

        zai_request = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        # Retry configuration
        max_attempts = request.max_retries
        backoff_seconds = self.config.retry_initial_backoff_seconds
        retry_count = 0

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()

                response = await self.http_client.post(
                    self.config.zai_base_url,  # Use full endpoint URL from env
                    json=zai_request,
                    headers={
                        "Authorization": f"Bearer {self.config.zai_api_key}",
                        "HTTP-Referer": "https://omninode.ai",
                        "X-Title": "OmniNode Bridge",
                        "anthropic-version": "2023-06-01",  # Required for Anthropic API
                    },
                )

                response.raise_for_status()
                data = response.json()

                # Log full Z.ai response for debugging
                emit_log_event(
                    level=LogLevel.INFO,
                    message=f"Z.ai raw response: {data}",
                    context={
                        "node_id": str(self.node_id),
                        "correlation_id": correlation_id,
                    },
                )

                # Handle Z.ai wrapper format (code, msg, success)
                if "code" in data and "msg" in data and "success" in data:
                    if not data.get("success"):
                        raise ModelOnexError(
                            code=EnumCoreErrorCode.INVALID_OPERATION,
                            message=f"Z.ai API error: {data.get('msg', 'Unknown error')}",
                            details={
                                "code": data.get("code"),
                                "zai_error": True,
                                "full_response": data,
                            },
                        )
                    # Unwrap the actual response from 'msg' field
                    data = data["msg"]

                latency_ms = (time.time() - start_time) * 1000

                # Extract response - support both OpenAI and Anthropic formats
                if "choices" in data:
                    # OpenAI Chat Completions format
                    message = data["choices"][0]["message"]
                    generated_text = message.get("content", "") or message.get(
                        "reasoning_content", ""
                    )
                    finish_reason = data["choices"][0]["finish_reason"]

                    # Token usage
                    usage = data.get("usage", {})
                    tokens_input = usage.get("prompt_tokens", 0)
                    tokens_output = usage.get("completion_tokens", 0)
                elif "content" in data:
                    # Anthropic Messages API format
                    content_blocks = data["content"]
                    generated_text = "".join(
                        block.get("text", "")
                        for block in content_blocks
                        if block.get("type") == "text"
                    )
                    finish_reason = data.get("stop_reason", "stop")

                    # Token usage
                    usage = data.get("usage", {})
                    tokens_input = usage.get("input_tokens", 0)
                    tokens_output = usage.get("output_tokens", 0)
                else:
                    raise ModelOnexError(
                        code=EnumCoreErrorCode.INVALID_OPERATION,
                        message="Unknown response format from LLM API",
                        details={"response_keys": list(data.keys())},
                    )

                tokens_total = tokens_input + tokens_output

                # Calculate cost
                cost_input = (tokens_input / 1_000_000) * self.cost_per_1m_input[
                    EnumLLMTier.CLOUD_FAST
                ]
                cost_output = (tokens_output / 1_000_000) * self.cost_per_1m_output[
                    EnumLLMTier.CLOUD_FAST
                ]
                cost_usd = cost_input + cost_output

                # Warnings
                warnings = []
                if finish_reason == "length":
                    warnings.append("Response truncated at max_tokens limit")

                return ModelLLMResponse(
                    generated_text=generated_text,
                    model_used=model,
                    tier_used=EnumLLMTier.CLOUD_FAST,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    tokens_total=tokens_total,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    finish_reason=finish_reason,
                    truncated=finish_reason == "length",
                    warnings=warnings,
                    retry_count=retry_count,
                    correlation_id=correlation_id,
                    execution_id=request.execution_id,
                    timestamp=datetime.now(UTC),
                )

            except httpx.HTTPStatusError as e:
                retry_count = attempt

                # Retry on rate limits and server errors
                if e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_attempts:
                        emit_log_event(
                            LogLevel.WARNING,
                            f"LLM API error {e.response.status_code}, retrying (attempt {attempt}/{max_attempts})",
                            {
                                "correlation_id": str(correlation_id),
                                "status_code": e.response.status_code,
                                "backoff_seconds": backoff_seconds,
                            },
                        )
                        await asyncio.sleep(backoff_seconds)
                        backoff_seconds *= self.config.retry_backoff_multiplier
                        continue

                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.NETWORK_ERROR,
                    message=f"Z.ai API error: {e.response.status_code}",
                    details={
                        "status_code": e.response.status_code,
                        "response": e.response.text[:1000],  # Truncate long responses
                        "retry_count": retry_count,
                    },
                )

            except httpx.TimeoutException:
                retry_count = attempt

                if attempt < max_attempts:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"LLM API timeout, retrying (attempt {attempt}/{max_attempts})",
                        {
                            "correlation_id": str(correlation_id),
                            "backoff_seconds": backoff_seconds,
                        },
                    )
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= self.config.retry_backoff_multiplier
                    continue

                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.TIMEOUT,
                    message=f"LLM API timeout after {max_attempts} attempts",
                    details={
                        "timeout_seconds": self.config.http_timeout_seconds,
                        "retry_count": retry_count,
                    },
                )

        # Should never reach here (loop should return or raise)
        raise ModelOnexError(
            code=EnumCoreErrorCode.OPERATION_FAILED,
            message="Unexpected error in retry loop",
        )


__all__ = ["NodeLLMEffect"]
