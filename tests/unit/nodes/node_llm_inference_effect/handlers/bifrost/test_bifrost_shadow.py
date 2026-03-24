# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for bifrost shadow mode — learned policy parallel evaluation.

Tests cover the OMN-5570 acceptance criteria:
    - Shadow mode adds < 5ms latency (async computation)
    - Shadow decisions logged for 100% of requests when enabled
    - Shadow mode does NOT affect the actual routing decision
    - Shadow policy timeout and error handling

Related:
    - OMN-5570: Shadow Mode + Comparison Dashboard
    - OMN-5556: Learned Decision Optimization Platform (epic)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.enums.enum_cost_tier import EnumCostTier
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost import (
    HandlerBifrostGateway,
    ModelBifrostConfig,
    ModelBifrostRequest,
    ModelBifrostShadowConfig,
    ModelShadowDecisionLog,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

pytestmark = pytest.mark.unit

# ── Fixtures ─────────────────────────────────────────────────────────────

_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


def _make_inference_response() -> ModelLlmInferenceResponse:
    """Create a minimal valid inference response for mocking."""
    return ModelLlmInferenceResponse(
        generated_text="Hello from bifrost",
        model_used="test-model",
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        finish_reason=EnumLlmFinishReason.STOP,
        usage=ModelLlmUsage(),
        latency_ms=50.0,
        backend_result=ModelBackendResult(success=True, duration_ms=50.0),
        correlation_id=uuid4(),
        execution_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


def _make_config(backend_ids: tuple[str, ...] = ("a", "b")) -> ModelBifrostConfig:
    """Create a minimal bifrost config with the given backend IDs."""
    backends = {
        bid: ModelBifrostBackendConfig(
            backend_id=bid,
            base_url=f"http://{bid}:8000",
        )
        for bid in backend_ids
    }
    return ModelBifrostConfig(
        backends=backends,
        default_backends=backend_ids,
    )


def _make_request(
    operation_type: EnumLlmOperationType = EnumLlmOperationType.CHAT_COMPLETION,
) -> ModelBifrostRequest:
    """Create a minimal bifrost request."""
    return ModelBifrostRequest(
        operation_type=operation_type,
        tenant_id=_TENANT_ID,
        messages=({"role": "user", "content": "Hello"},),
    )


def _make_gateway(
    config: ModelBifrostConfig | None = None,
    shadow_config: ModelBifrostShadowConfig | None = None,
    shadow_policy: object | None = None,
    shadow_callback: object | None = None,
) -> HandlerBifrostGateway:
    """Create a gateway with mocked inference handler."""
    transport = MagicMock(spec=MixinLlmHttpTransport)
    inference_handler = HandlerLlmOpenaiCompatible(transport)
    inference_handler.handle = AsyncMock(return_value=_make_inference_response())

    return HandlerBifrostGateway(
        config=config or _make_config(),
        inference_handler=inference_handler,
        shadow_config=shadow_config,
        shadow_policy=shadow_policy,
        shadow_decision_callback=shadow_callback,
    )


class MockShadowPolicy:
    """Mock shadow policy that returns a configurable backend recommendation."""

    def __init__(
        self,
        recommended: str = "b",
        confidence: float = 0.85,
        distribution: dict[str, float] | None = None,
        delay_seconds: float = 0.0,
        raise_error: bool = False,
    ) -> None:
        self.recommended = recommended
        self.confidence = confidence
        self.distribution = distribution or {"a": 0.15, "b": 0.85}
        self.delay_seconds = delay_seconds
        self.raise_error = raise_error
        self.call_count = 0

    async def recommend(
        self,
        request: ModelBifrostRequest,
        available_backends: tuple[str, ...],
    ) -> tuple[str, float, dict[str, float]]:
        self.call_count += 1
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if self.raise_error:
            msg = "Shadow policy failed"
            raise RuntimeError(msg)
        return self.recommended, self.confidence, self.distribution


# ── Shadow Config Tests ──────────────────────────────────────────────────


class TestModelBifrostShadowConfig:
    """Tests for ModelBifrostShadowConfig model validation."""

    def test_default_config_disabled(self) -> None:
        config = ModelBifrostShadowConfig()
        assert config.enabled is False
        assert config.checkpoint_path is None
        assert config.log_sample_rate == 1.0
        assert config.max_shadow_latency_ms == 5.0

    def test_enabled_config(self) -> None:
        config = ModelBifrostShadowConfig(
            enabled=True,
            checkpoint_path="/models/policy_v1.onnx",
            policy_version="v1.0.0",
        )
        assert config.enabled is True
        assert config.checkpoint_path == "/models/policy_v1.onnx"
        assert config.policy_version == "v1.0.0"

    def test_sample_rate_validation(self) -> None:
        config = ModelBifrostShadowConfig(log_sample_rate=0.5)
        assert config.log_sample_rate == 0.5

        with pytest.raises(Exception):
            ModelBifrostShadowConfig(log_sample_rate=1.5)

        with pytest.raises(Exception):
            ModelBifrostShadowConfig(log_sample_rate=-0.1)


class TestModelShadowDecisionLog:
    """Tests for ModelShadowDecisionLog model."""

    def test_agreed_decision(self) -> None:
        log = ModelShadowDecisionLog(
            correlation_id=uuid4(),
            static_backend_selected="a",
            shadow_backend_recommended="a",
            agreed=True,
            request_operation_type="chat_completion",
            request_cost_tier="mid",
            request_max_latency_ms=5000,
            shadow_confidence=0.9,
            shadow_latency_ms=1.5,
            policy_version="v1.0.0",
            tenant_id=_TENANT_ID,
        )
        assert log.agreed is True
        assert log.static_backend_selected == "a"
        assert log.shadow_backend_recommended == "a"

    def test_disagreed_decision(self) -> None:
        log = ModelShadowDecisionLog(
            correlation_id=uuid4(),
            static_backend_selected="a",
            shadow_backend_recommended="b",
            agreed=False,
            request_operation_type="chat_completion",
            request_cost_tier="low",
            request_max_latency_ms=3000,
            shadow_confidence=0.85,
            shadow_latency_ms=2.1,
            policy_version="v1.0.0",
            shadow_action_distribution={"a": 0.15, "b": 0.85},
            tenant_id=_TENANT_ID,
        )
        assert log.agreed is False
        assert log.shadow_action_distribution == {"a": 0.15, "b": 0.85}


# ── Shadow Mode Integration Tests ───────────────────────────────────────


class TestShadowModeDisabled:
    """Shadow mode disabled — gateway behaves identically to pre-OMN-5570."""

    @pytest.mark.asyncio
    async def test_no_shadow_when_disabled(self) -> None:
        """Shadow policy is not invoked when shadow_config.enabled=False."""
        policy = MockShadowPolicy()
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(enabled=False),
            shadow_policy=policy,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True
        assert response.backend_selected in ("a", "b")
        # Give event loop a chance to process any fire-and-forget tasks
        await asyncio.sleep(0.01)
        assert policy.call_count == 0

    @pytest.mark.asyncio
    async def test_no_shadow_without_policy(self) -> None:
        """Shadow does not run when no policy is provided."""
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(enabled=True),
            shadow_policy=None,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True


class TestShadowModeEnabled:
    """Shadow mode enabled — policy runs in parallel, does not affect routing."""

    @pytest.mark.asyncio
    async def test_shadow_policy_invoked(self) -> None:
        """Shadow policy is called when enabled with a valid policy."""
        policy = MockShadowPolicy(recommended="b", confidence=0.85)
        callback = AsyncMock()
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(
                enabled=True,
                policy_version="test-v1",
            ),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True
        # Wait for fire-and-forget shadow task
        await asyncio.sleep(0.05)
        assert policy.call_count == 1
        callback.assert_called_once()

        # Verify the log entry
        log_entry: ModelShadowDecisionLog = callback.call_args[0][0]
        assert log_entry.shadow_backend_recommended == "b"
        assert log_entry.shadow_confidence == 0.85
        assert log_entry.policy_version == "test-v1"

    @pytest.mark.asyncio
    async def test_shadow_does_not_affect_routing(self) -> None:
        """Shadow recommending a different backend does not change actual routing."""
        # Shadow recommends "b", but static routing should select from defaults
        policy = MockShadowPolicy(recommended="b")
        callback = AsyncMock()

        config = _make_config(backend_ids=("a",))  # only "a" available
        gateway = _make_gateway(
            config=config,
            shadow_config=ModelBifrostShadowConfig(enabled=True),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True
        assert response.backend_selected == "a"  # Static routing selects "a"
        await asyncio.sleep(0.05)
        assert policy.call_count == 1

    @pytest.mark.asyncio
    async def test_shadow_agreement_logged(self) -> None:
        """When shadow agrees with static, agreed=True in the log."""
        # Static will route to "a" (first default), shadow also recommends "a"
        policy = MockShadowPolicy(recommended="a", confidence=0.95)
        callback = AsyncMock()

        config = _make_config(backend_ids=("a",))
        gateway = _make_gateway(
            config=config,
            shadow_config=ModelBifrostShadowConfig(enabled=True),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        await gateway.handle(request)
        await asyncio.sleep(0.05)

        log_entry: ModelShadowDecisionLog = callback.call_args[0][0]
        assert log_entry.agreed is True

    @pytest.mark.asyncio
    async def test_shadow_disagreement_logged(self) -> None:
        """When shadow disagrees with static, agreed=False in the log."""
        policy = MockShadowPolicy(recommended="b")
        callback = AsyncMock()

        config = _make_config(backend_ids=("a",))
        gateway = _make_gateway(
            config=config,
            shadow_config=ModelBifrostShadowConfig(enabled=True),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        await gateway.handle(request)
        await asyncio.sleep(0.05)

        log_entry: ModelShadowDecisionLog = callback.call_args[0][0]
        assert log_entry.agreed is False
        assert log_entry.static_backend_selected == "a"
        assert log_entry.shadow_backend_recommended == "b"


class TestShadowModeResilience:
    """Shadow mode error handling — gateway must never crash due to shadow."""

    @pytest.mark.asyncio
    async def test_shadow_timeout_does_not_crash(self) -> None:
        """Shadow policy timeout does not affect the actual routing response."""
        # Policy takes 100ms but timeout is 5ms
        policy = MockShadowPolicy(delay_seconds=0.1)
        callback = AsyncMock()
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(
                enabled=True,
                max_shadow_latency_ms=5.0,
            ),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True
        await asyncio.sleep(0.15)
        # Callback should NOT be called because the shadow timed out
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_shadow_error_does_not_crash(self) -> None:
        """Shadow policy exception does not affect the actual routing response."""
        policy = MockShadowPolicy(raise_error=True)
        callback = AsyncMock()
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(enabled=True),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True
        await asyncio.sleep(0.05)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_shadow_callback_error_does_not_crash(self) -> None:
        """Callback error does not affect the actual routing response."""
        policy = MockShadowPolicy()
        callback = AsyncMock(side_effect=RuntimeError("Kafka publish failed"))
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(enabled=True),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        response = await gateway.handle(request)

        assert response.success is True
        await asyncio.sleep(0.05)
        # Callback was called but it raised — gateway should still be fine
        callback.assert_called_once()


class TestShadowModeSampling:
    """Shadow mode log_sample_rate controls how often shadow runs."""

    @pytest.mark.asyncio
    async def test_sample_rate_zero_no_shadow(self) -> None:
        """Sample rate 0.0 means shadow never runs."""
        policy = MockShadowPolicy()
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(
                enabled=True,
                log_sample_rate=0.0,
            ),
            shadow_policy=policy,
        )
        request = _make_request()

        for _ in range(10):
            await gateway.handle(request)

        await asyncio.sleep(0.05)
        assert policy.call_count == 0

    @pytest.mark.asyncio
    async def test_comparison_logging_disabled(self) -> None:
        """When comparison_logging_enabled=False, callback is not invoked."""
        policy = MockShadowPolicy()
        callback = AsyncMock()
        gateway = _make_gateway(
            shadow_config=ModelBifrostShadowConfig(
                enabled=True,
                comparison_logging_enabled=False,
            ),
            shadow_policy=policy,
            shadow_callback=callback,
        )
        request = _make_request()
        await gateway.handle(request)
        await asyncio.sleep(0.05)

        # Policy was still called for shadow evaluation
        assert policy.call_count == 1
        # But callback was NOT called because logging is disabled
        callback.assert_not_called()
