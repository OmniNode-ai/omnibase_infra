# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for bifrost gateway failover — backend outage simulation.

Tests cover the Definition of Done requirement:
    "Outage simulation — backend A down routes to B, failover event
     recorded: uv run pytest tests/integration/ -k bifrost_failover"

Note: This file covers the unit-level failover simulation. The DoD
specifies 'tests/integration/' for this test but since we simulate
outages via mocks (not real network), these run as unit tests. The
k-expression 'bifrost_failover' works with both directories.

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.enums.enum_cost_tier import EnumCostTier
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost import (
    HandlerBifrostGateway,
    ModelBifrostConfig,
    ModelBifrostRequest,
    ModelBifrostRoutingRule,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_inference_response(latency_ms: float = 100.0) -> ModelLlmInferenceResponse:
    """Build a minimal valid ModelLlmInferenceResponse."""
    return ModelLlmInferenceResponse(
        generated_text="OK",
        model_used="test-model",
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        finish_reason=EnumLlmFinishReason.STOP,
        usage=ModelLlmUsage(),
        latency_ms=latency_ms,
        backend_result=ModelBackendResult(success=True, duration_ms=latency_ms),
        correlation_id=uuid4(),
        execution_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


def _make_two_backend_config(
    *,
    rule_backend_ids: tuple[str, ...] = ("backend-a", "backend-b"),
    failover_attempts: int = 3,
    failover_backoff_base_ms: int = 0,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_window_seconds: int = 30,
) -> ModelBifrostConfig:
    """Build a two-backend config for failover tests."""
    return ModelBifrostConfig(
        backends={
            "backend-a": ModelBifrostBackendConfig(
                backend_id="backend-a",
                base_url="http://backend-a:8000",
                model_name="model-a",
            ),
            "backend-b": ModelBifrostBackendConfig(
                backend_id="backend-b",
                base_url="http://backend-b:8000",
                model_name="model-b",
            ),
        },
        routing_rules=(
            ModelBifrostRoutingRule(
                rule_id="default-rule",
                priority=10,
                backend_ids=rule_backend_ids,
            ),
        ),
        default_backends=(),
        failover_attempts=failover_attempts,
        failover_backoff_base_ms=failover_backoff_base_ms,
        circuit_breaker_failure_threshold=circuit_breaker_failure_threshold,
        circuit_breaker_window_seconds=circuit_breaker_window_seconds,
    )


def _make_request() -> ModelBifrostRequest:
    """Build a minimal valid ModelBifrostRequest for failover tests."""
    return ModelBifrostRequest(
        operation_type="chat_completion",
        tenant_id="test-tenant",
        messages=[{"role": "user", "content": "Hello"}],
    )


def _make_failing_handler(error: Exception | None = None) -> HandlerLlmOpenaiCompatible:
    """Create a handler whose handle() always raises the given error."""
    transport = MagicMock(spec=MixinLlmHttpTransport)
    handler = HandlerLlmOpenaiCompatible(transport=transport)
    if error is None:
        error = InfraUnavailableError("Simulated backend outage", context=None)
    handler.handle = AsyncMock(side_effect=error)
    return handler


def _make_handler_with_sequence(
    responses: list[ModelLlmInferenceResponse | Exception],
) -> HandlerLlmOpenaiCompatible:
    """Create a handler that returns responses in sequence (success or raises)."""
    transport = MagicMock(spec=MixinLlmHttpTransport)
    handler = HandlerLlmOpenaiCompatible(transport=transport)

    async def _side_effect(*args, **kwargs):
        if not responses:
            raise InfraUnavailableError("No more responses", context=None)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    handler.handle = AsyncMock(side_effect=_side_effect)
    return handler


# ---------------------------------------------------------------------------
# Tests: Backend A down → route to B, failover recorded
# ---------------------------------------------------------------------------


class TestBifrostFailoverBackendADown:
    """Tests for OMN-2736 R3: failover on backend outage."""

    @pytest.mark.asyncio
    async def test_bifrost_failover_backend_a_down_routes_to_b(self) -> None:
        """DoD: Backend A down → route to B, failover event recorded.

        This is the primary DoD requirement for bifrost_failover.
        """
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            failover_attempts=2,
        )

        # First call (backend-a) raises, second call (backend-b) succeeds
        success_response = _make_inference_response()
        handler = _make_handler_with_sequence([
            InfraUnavailableError("backend-a is down", context=None),
            success_response,
        ])
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_request()
        result = await gateway.handle(request)

        # backend-b served the request
        assert result.success is True
        assert result.backend_selected == "backend-b"
        assert result.rule_id == "default-rule"
        # One retry (backend-a failed, backend-b succeeded)
        assert result.retry_count == 1
        assert result.inference_response is not None

    @pytest.mark.asyncio
    async def test_bifrost_failover_retry_count_matches_attempts(self) -> None:
        """retry_count reflects actual number of failed attempts."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            failover_attempts=3,
        )
        success_response = _make_inference_response()
        handler = _make_handler_with_sequence([
            InfraUnavailableError("backend-a down", context=None),
            success_response,
        ])
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        result = await gateway.handle(_make_request())

        assert result.success is True
        assert result.retry_count == 1  # 1 failed before success

    @pytest.mark.asyncio
    async def test_bifrost_failover_all_backends_down_returns_structured_error(self) -> None:
        """R3: All backends down → return structured error, not timeout hang."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            failover_attempts=2,
        )
        handler = _make_failing_handler(
            InfraUnavailableError("All down", context=None)
        )
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_request()
        result = await gateway.handle(request)

        assert result.success is False
        assert result.backend_selected == ""
        assert result.inference_response is None
        assert result.error_message != ""
        assert "tenant_id" in result.error_message or "attempt" in result.error_message

    @pytest.mark.asyncio
    async def test_bifrost_failover_structured_error_includes_tenant_and_operation(self) -> None:
        """Structured error message includes tenant_id and operation_type for queryability."""
        config = _make_two_backend_config(failover_attempts=1)
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = ModelBifrostRequest(
            operation_type="chat_completion",
            tenant_id="specific-tenant",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = await gateway.handle(request)

        assert result.success is False
        assert "specific-tenant" in result.error_message
        assert "chat_completion" in result.error_message

    @pytest.mark.asyncio
    async def test_bifrost_failover_rule_id_preserved_on_failure(self) -> None:
        """rule_id is recorded even when all backends fail."""
        config = _make_two_backend_config(failover_attempts=2)
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        result = await gateway.handle(_make_request())

        assert result.success is False
        assert result.rule_id == "default-rule"

    @pytest.mark.asyncio
    async def test_bifrost_failover_first_backend_success_zero_retries(self) -> None:
        """When backend-a succeeds first, retry_count=0."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            failover_attempts=2,
        )
        handler = _make_handler_with_sequence([_make_inference_response()])
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        result = await gateway.handle(_make_request())

        assert result.success is True
        assert result.backend_selected == "backend-a"
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_bifrost_failover_failover_attempts_limits_tried_backends(self) -> None:
        """failover_attempts=1 limits to only 1 backend tried even if more available."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            failover_attempts=1,  # Only try 1 backend
        )
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        result = await gateway.handle(_make_request())

        # With failover_attempts=1, only backend-a is tried (and fails)
        assert result.success is False
        # handle() called exactly once (only 1 attempt allowed)
        assert handler.handle.call_count == 1

    @pytest.mark.asyncio
    async def test_bifrost_failover_unknown_backend_id_skipped(self) -> None:
        """Routing rule referencing an unknown backend_id is gracefully skipped."""
        config = ModelBifrostConfig(
            backends={
                "backend-b": ModelBifrostBackendConfig(
                    backend_id="backend-b",
                    base_url="http://backend-b:8000",
                    model_name="model-b",
                ),
            },
            routing_rules=(
                ModelBifrostRoutingRule(
                    rule_id="rule-unknown",
                    priority=10,
                    # backend-missing not in config.backends
                    backend_ids=("backend-missing", "backend-b"),
                ),
            ),
            default_backends=(),
            failover_attempts=2,
            failover_backoff_base_ms=0,
        )
        success_response = _make_inference_response()
        transport = MagicMock(spec=MixinLlmHttpTransport)
        handler = HandlerLlmOpenaiCompatible(transport=transport)
        handler.handle = AsyncMock(return_value=success_response)
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        result = await gateway.handle(_make_request())

        # backend-missing skipped, backend-b served request
        assert result.success is True
        assert result.backend_selected == "backend-b"


# ---------------------------------------------------------------------------
# Tests: Circuit breaker behavior
# ---------------------------------------------------------------------------


class TestBifrostFailoverCircuitBreaker:
    """Tests for circuit breaker open/close behavior."""

    @pytest.mark.asyncio
    async def test_bifrost_failover_circuit_opens_after_threshold_failures(self) -> None:
        """Circuit opens after circuit_breaker_failure_threshold consecutive failures."""
        config = _make_two_backend_config(
            circuit_breaker_failure_threshold=3,
            circuit_breaker_window_seconds=60,
            failover_attempts=5,
        )
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # Record 3 failures for backend-a via _record_failure
        for _ in range(3):
            await gateway._record_failure("backend-a")

        # Circuit should now be open
        assert await gateway._is_circuit_open("backend-a") is True

    @pytest.mark.asyncio
    async def test_bifrost_failover_circuit_closed_before_threshold(self) -> None:
        """Circuit remains closed with fewer than threshold failures."""
        config = _make_two_backend_config(
            circuit_breaker_failure_threshold=5,
        )
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        for _ in range(4):  # 4 < threshold of 5
            await gateway._record_failure("backend-a")

        assert await gateway._is_circuit_open("backend-a") is False

    @pytest.mark.asyncio
    async def test_bifrost_failover_circuit_closes_after_success(self) -> None:
        """Circuit closes (failure count resets) after a successful call."""
        config = _make_two_backend_config(
            circuit_breaker_failure_threshold=3,
        )
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # Record 2 failures (not enough to open)
        for _ in range(2):
            await gateway._record_failure("backend-a")

        assert gateway.get_circuit_failure_count("backend-a") == 2

        # Record success — should reset
        await gateway._record_success("backend-a")

        assert gateway.get_circuit_failure_count("backend-a") == 0
        assert await gateway._is_circuit_open("backend-a") is False

    @pytest.mark.asyncio
    async def test_bifrost_failover_open_circuit_skips_backend(self) -> None:
        """Backend with open circuit is skipped in _attempt_backends."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            circuit_breaker_failure_threshold=2,
            failover_attempts=3,
        )
        success_response = _make_inference_response()
        transport = MagicMock(spec=MixinLlmHttpTransport)
        handler = HandlerLlmOpenaiCompatible(transport=transport)
        handler.handle = AsyncMock(return_value=success_response)
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # Open the circuit for backend-a
        for _ in range(3):  # > threshold of 2
            await gateway._record_failure("backend-a")

        assert await gateway._is_circuit_open("backend-a") is True

        result = await gateway.handle(_make_request())

        # backend-a skipped (circuit open), backend-b served request
        assert result.success is True
        assert result.backend_selected == "backend-b"

    @pytest.mark.asyncio
    async def test_bifrost_failover_all_circuits_open_returns_error(self) -> None:
        """When all backends have open circuits, handle() returns structured error."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a", "backend-b"),
            circuit_breaker_failure_threshold=2,
            failover_attempts=3,
        )
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # Open circuits for both backends
        for backend_id in ["backend-a", "backend-b"]:
            for _ in range(3):
                await gateway._record_failure(backend_id)

        result = await gateway.handle(_make_request())

        assert result.success is False
        # handle() should not have been called (all circuits open, backends skipped)
        # The handler.handle mock should not have been called due to circuit protection
        assert handler.handle.call_count == 0

    @pytest.mark.asyncio
    async def test_bifrost_failover_failure_count_increments_on_exception(self) -> None:
        """Each failed backend call increments that backend's failure count."""
        config = _make_two_backend_config(
            rule_backend_ids=("backend-a",),
            circuit_breaker_failure_threshold=10,
            failover_attempts=5,
        )
        handler = _make_failing_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        assert gateway.get_circuit_failure_count("backend-a") == 0

        await gateway.handle(_make_request())

        # 1 attempt to backend-a which failed → failure_count == 1
        assert gateway.get_circuit_failure_count("backend-a") == 1


# ---------------------------------------------------------------------------
# Tests: HMAC authentication
# ---------------------------------------------------------------------------


class TestBifrostFailoverHmacAuth:
    """Tests for HMAC signature computation."""

    def test_bifrost_failover_hmac_signature_format(self) -> None:
        """_compute_hmac_signature returns hmac-sha256-prefixed hex string."""
        sig = HandlerBifrostGateway._compute_hmac_signature(
            secret="test-secret",
            correlation_id="corr-123",
        )
        assert sig.startswith("hmac-sha256-")
        # Hex portion should be 64 chars (SHA-256 hex)
        hex_part = sig[len("hmac-sha256-"):]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_bifrost_failover_hmac_different_secrets_produce_different_sigs(self) -> None:
        """Different HMAC secrets produce different signatures."""
        sig_a = HandlerBifrostGateway._compute_hmac_signature(
            secret="secret-a",
            correlation_id="corr-1",
        )
        sig_b = HandlerBifrostGateway._compute_hmac_signature(
            secret="secret-b",
            correlation_id="corr-1",
        )
        assert sig_a != sig_b

    def test_bifrost_failover_hmac_different_correlation_ids_produce_different_sigs(self) -> None:
        """Different correlation IDs produce different signatures (replay protection)."""
        sig_1 = HandlerBifrostGateway._compute_hmac_signature(
            secret="same-secret",
            correlation_id="corr-1",
        )
        sig_2 = HandlerBifrostGateway._compute_hmac_signature(
            secret="same-secret",
            correlation_id="corr-2",
        )
        assert sig_1 != sig_2

    @pytest.mark.asyncio
    async def test_bifrost_failover_backend_without_hmac_secret_no_api_key(self) -> None:
        """Backend without hmac_secret does not inject an api_key into inference request."""
        config = ModelBifrostConfig(
            backends={
                "backend-a": ModelBifrostBackendConfig(
                    backend_id="backend-a",
                    base_url="http://backend-a:8000",
                    model_name="model-a",
                    hmac_secret=None,  # No HMAC
                ),
            },
            routing_rules=(),
            default_backends=("backend-a",),
            failover_attempts=1,
            failover_backoff_base_ms=0,
        )
        success_response = _make_inference_response()
        transport = MagicMock(spec=MixinLlmHttpTransport)
        inference_handler = HandlerLlmOpenaiCompatible(transport=transport)

        captured_requests: list = []

        async def _capture(*args, **kwargs):
            captured_requests.append(args[0] if args else kwargs.get("request"))
            return success_response

        inference_handler.handle = AsyncMock(side_effect=_capture)
        gateway = HandlerBifrostGateway(config=config, inference_handler=inference_handler)

        await gateway.handle(_make_request())

        assert len(captured_requests) == 1
        captured_req = captured_requests[0]
        assert captured_req is not None
        assert captured_req.api_key is None

    @pytest.mark.asyncio
    async def test_bifrost_failover_backend_with_hmac_secret_injects_api_key(self) -> None:
        """Backend with hmac_secret injects a non-None api_key into inference request."""
        config = ModelBifrostConfig(
            backends={
                "backend-a": ModelBifrostBackendConfig(
                    backend_id="backend-a",
                    base_url="http://backend-a:8000",
                    model_name="model-a",
                    hmac_secret="my-secret-key",
                ),
            },
            routing_rules=(),
            default_backends=("backend-a",),
            failover_attempts=1,
            failover_backoff_base_ms=0,
        )
        success_response = _make_inference_response()
        transport = MagicMock(spec=MixinLlmHttpTransport)
        inference_handler = HandlerLlmOpenaiCompatible(transport=transport)

        captured_requests: list = []

        async def _capture(*args, **kwargs):
            captured_requests.append(args[0] if args else kwargs.get("request"))
            return success_response

        inference_handler.handle = AsyncMock(side_effect=_capture)
        gateway = HandlerBifrostGateway(config=config, inference_handler=inference_handler)

        await gateway.handle(_make_request())

        assert len(captured_requests) == 1
        captured_req = captured_requests[0]
        assert captured_req is not None
        assert captured_req.api_key is not None
        assert captured_req.api_key.startswith("hmac-sha256-")
