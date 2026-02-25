# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for bifrost gateway routing — deterministic rule evaluation.

Tests cover the Definition of Done requirement:
    "Deterministic routing config test — N synthetic requests select
     expected backends: uv run pytest tests/unit/ -k bifrost_routing"

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.enums.enum_cost_tier import EnumCostTier
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

# Stable test UUIDs for routing rule IDs
_RULE_CHAT_ONLY = UUID("e9768a9e-1d87-5e8b-bd52-df64e3a1696a")
_RULE_EMBEDDING_ONLY = UUID("3a8cac71-534c-559a-9112-7653badecad8")
_RULE_LOW_COST = UUID("cc777383-da26-5528-9e91-6372239ab5f4")
_RULE_HIGH_COST = UUID("e2ff7898-b68b-5feb-a4d5-407500ae20b6")
_RULE_TOOL_JSON = UUID("d91e087b-8ea9-5f89-b272-68dee5075f67")
_RULE_FAST = UUID("2807a315-5a3a-5568-a632-27ecb7e15b44")
_RULE_HIGH_PRIORITY = UUID("a751a784-aac8-55a1-8ac5-cb87fffa8591")
_RULE_LOW_PRIORITY = UUID("ae1ad2fc-ef0f-5e83-91d0-db7dafb0b739")
_RULE_CATCH_ALL = UUID("420f1565-313e-5fa9-b128-fb6414837c7c")
_RULE_CHAT_LOW = UUID("1a64acc1-7d94-5b74-9b56-98c959cdc494")
_RULE_RULE_CHAT = UUID("3cda38ef-f8f7-5e0c-acf1-ee1da4426262")
_RULE_RULE_EMBED = UUID("3471fed8-51bf-5ebf-9f2a-ffbd036213ef")
_RULE_COMBINED = UUID("7ccc24a0-8c5d-5841-942e-85868e1afb9a")
_RULE_CAP_MATCH = UUID("c5467def-9d89-59c2-ac00-c4e1c67f2a83")
_RULE_LATENCY_1000 = UUID("12102e49-4a09-5c79-99e0-26ab779d0b62")
_RULE_CHAT_OR_COMPLETE = UUID("f8aff824-7b8e-5321-9af3-1f28709eae34")
_RULE_LOW_ONLY = UUID("daab60d7-ef92-5813-8e25-7ecd0f051de3")
_RULE_WILDCARD = UUID("692797c9-0f81-5016-8733-ce30c1b6a36f")
_RULE_LOW_COST_5B = UUID("81ac0d1b-d645-5087-bef7-fcf20f2a9497")
_RULE_MID_COST_14B = UUID("66f96b84-68ac-5183-a3bd-092004e0fa76")
_RULE_HIGH_COST_70B = UUID("f061f4c1-fbad-54e7-8a6a-03aae3375a9c")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_inference_response(
    backend_id: str = "test-backend",
    latency_ms: float = 100.0,
) -> ModelLlmInferenceResponse:
    """Build a minimal valid ModelLlmInferenceResponse for mocking."""
    return ModelLlmInferenceResponse(
        generated_text="Hello from bifrost",
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


def _make_transport() -> MagicMock:
    """Create a mock MixinLlmHttpTransport."""
    return MagicMock(spec=MixinLlmHttpTransport)


def _make_handler(
    inference_response: ModelLlmInferenceResponse | None = None,
) -> HandlerLlmOpenaiCompatible:
    """Create a HandlerLlmOpenaiCompatible with mocked handle()."""
    transport = _make_transport()
    handler = HandlerLlmOpenaiCompatible(transport=transport)
    if inference_response is None:
        inference_response = _make_inference_response()
    handler.handle = AsyncMock(return_value=inference_response)
    return handler


def _make_two_backend_config(
    backend_a_id: str = "backend-a",
    backend_b_id: str = "backend-b",
    backend_a_url: str = "http://backend-a:8000",
    backend_b_url: str = "http://backend-b:8000",
    routing_rules: tuple[ModelBifrostRoutingRule, ...] = (),
    default_backends: tuple[str, ...] | None = None,
) -> ModelBifrostConfig:
    """Build a two-backend config for routing tests."""
    if default_backends is None:
        default_backends = (backend_a_id,)
    return ModelBifrostConfig(
        backends={
            backend_a_id: ModelBifrostBackendConfig(
                backend_id=backend_a_id,
                base_url=backend_a_url,
                model_name="model-a",
            ),
            backend_b_id: ModelBifrostBackendConfig(
                backend_id=backend_b_id,
                base_url=backend_b_url,
                model_name="model-b",
            ),
        },
        routing_rules=routing_rules,
        default_backends=default_backends,
        failover_attempts=2,
        failover_backoff_base_ms=0,  # No delay in unit tests
    )


def _make_chat_request(
    *,
    operation_type: EnumLlmOperationType = EnumLlmOperationType.CHAT_COMPLETION,
    cost_tier: EnumCostTier = EnumCostTier.MID,
    capabilities: tuple[str, ...] = (),
    max_latency_ms: int = 10_000,
    tenant_id: UUID | None = None,
) -> ModelBifrostRequest:
    """Build a minimal valid ModelBifrostRequest."""
    return ModelBifrostRequest(
        operation_type=operation_type,
        cost_tier=cost_tier,
        capabilities=capabilities,
        max_latency_ms=max_latency_ms,
        tenant_id=tenant_id or UUID("00000000-0000-0000-0000-000000000001"),
        messages=[{"role": "user", "content": "Hello"}],
    )


# ---------------------------------------------------------------------------
# Tests: Rule evaluation — deterministic backend selection
# ---------------------------------------------------------------------------


class TestBifrostRoutingRuleEvaluation:
    """Tests that routing rules deterministically select expected backends."""

    def test_bifrost_routing_no_rules_uses_default_backend(self) -> None:
        """With no routing rules, default_backends is returned."""
        config = _make_two_backend_config(
            routing_rules=(),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request()
        candidate_ids, matched_rule = gateway._evaluate_rules(request)

        assert candidate_ids == ("backend-b",)
        assert matched_rule is None

    def test_bifrost_routing_single_rule_matches_by_operation_type(self) -> None:
        """Rule with match_operation_types=CHAT_COMPLETION matches correctly."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_CHAT_ONLY,
            priority=10,
            match_operation_types=(EnumLlmOperationType.CHAT_COMPLETION.value,),
            backend_ids=("backend-a",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule,),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION
        )
        candidate_ids, matched_rule = gateway._evaluate_rules(request)

        assert candidate_ids == ("backend-a",)
        assert matched_rule is not None
        assert matched_rule.rule_id == _RULE_CHAT_ONLY

    def test_bifrost_routing_operation_type_mismatch_falls_through(self) -> None:
        """Rule for EMBEDDING does not match a CHAT_COMPLETION request."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_EMBEDDING_ONLY,
            priority=10,
            match_operation_types=(EnumLlmOperationType.EMBEDDING.value,),
            backend_ids=("backend-a",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule,),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION
        )
        candidate_ids, matched_rule = gateway._evaluate_rules(request)

        assert candidate_ids == ("backend-b",)  # default
        assert matched_rule is None

    def test_bifrost_routing_cost_tier_low_selects_low_cost_backend(self) -> None:
        """Rule matching cost_tier='low' selects the cheap backend."""
        rule_low = ModelBifrostRoutingRule(
            rule_id=_RULE_LOW_COST,
            priority=10,
            match_cost_tiers=("low",),
            backend_ids=("backend-a",),  # cheap backend
        )
        rule_high = ModelBifrostRoutingRule(
            rule_id=_RULE_HIGH_COST,
            priority=20,
            match_cost_tiers=("high",),
            backend_ids=("backend-b",),  # expensive backend
        )
        config = _make_two_backend_config(
            routing_rules=(rule_low, rule_high),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request(cost_tier=EnumCostTier.LOW)
        candidate_ids, matched_rule = gateway._evaluate_rules(request)

        assert candidate_ids == ("backend-a",)
        assert matched_rule is not None
        assert matched_rule.rule_id == _RULE_LOW_COST

    def test_bifrost_routing_cost_tier_high_selects_high_cost_backend(self) -> None:
        """Rule matching cost_tier='high' selects the premium backend."""
        rule_low = ModelBifrostRoutingRule(
            rule_id=_RULE_LOW_COST,
            priority=10,
            match_cost_tiers=("low",),
            backend_ids=("backend-a",),
        )
        rule_high = ModelBifrostRoutingRule(
            rule_id=_RULE_HIGH_COST,
            priority=20,
            match_cost_tiers=("high",),
            backend_ids=("backend-b",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule_low, rule_high),
            default_backends=("backend-a",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request(cost_tier=EnumCostTier.HIGH)
        candidate_ids, matched_rule = gateway._evaluate_rules(request)

        assert candidate_ids == ("backend-b",)
        assert matched_rule is not None
        assert matched_rule.rule_id == _RULE_HIGH_COST

    def test_bifrost_routing_capability_match_all_required(self) -> None:
        """Rule requiring ['tool_calling', 'json_mode'] only matches if BOTH present."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_TOOL_JSON,
            priority=10,
            match_capabilities=("tool_calling", "json_mode"),
            backend_ids=("backend-a",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule,),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # Only tool_calling — should NOT match
        request_partial = _make_chat_request(capabilities=("tool_calling",))
        candidate_ids, matched_rule = gateway._evaluate_rules(request_partial)
        assert candidate_ids == ("backend-b",)  # default
        assert matched_rule is None

        # Both capabilities — SHOULD match
        request_full = _make_chat_request(capabilities=("tool_calling", "json_mode"))
        candidate_ids, matched_rule = gateway._evaluate_rules(request_full)
        assert candidate_ids == ("backend-a",)
        assert matched_rule is not None
        assert matched_rule.rule_id == _RULE_TOOL_JSON

    def test_bifrost_routing_latency_constraint_filters_rules(self) -> None:
        """Rule with match_max_latency_ms_lte=1000 does not match request with 5000ms budget."""
        rule_fast = ModelBifrostRoutingRule(
            rule_id=_RULE_FAST,
            priority=10,
            match_max_latency_ms_lte=1000,
            backend_ids=("backend-a",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule_fast,),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # 5000ms > 1000ms limit — rule should not match
        request_slow = _make_chat_request(max_latency_ms=5000)
        candidate_ids, matched_rule = gateway._evaluate_rules(request_slow)
        assert candidate_ids == ("backend-b",)  # default
        assert matched_rule is None

        # 500ms <= 1000ms limit — rule should match
        request_fast = _make_chat_request(max_latency_ms=500)
        candidate_ids, matched_rule = gateway._evaluate_rules(request_fast)
        assert candidate_ids == ("backend-a",)
        assert matched_rule is not None
        assert matched_rule.rule_id == _RULE_FAST

    def test_bifrost_routing_priority_order_first_match_wins(self) -> None:
        """Lower priority value wins when multiple rules could match."""
        rule_high_priority = ModelBifrostRoutingRule(
            rule_id=_RULE_HIGH_PRIORITY,
            priority=5,  # Lower number = higher priority
            backend_ids=("backend-a",),
        )
        rule_low_priority = ModelBifrostRoutingRule(
            rule_id=_RULE_LOW_PRIORITY,
            priority=50,
            backend_ids=("backend-b",),
        )
        # Insert in reverse priority order — gateway must sort them
        config = _make_two_backend_config(
            routing_rules=(rule_low_priority, rule_high_priority),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request()
        candidate_ids, matched_rule = gateway._evaluate_rules(request)

        assert candidate_ids == ("backend-a",)
        assert matched_rule is not None
        assert matched_rule.rule_id == _RULE_HIGH_PRIORITY

    def test_bifrost_routing_wildcard_rule_matches_any_request(self) -> None:
        """Rule with no match predicates matches every request."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_CATCH_ALL,
            priority=100,
            # No match predicates = wildcard
            backend_ids=("backend-a", "backend-b"),
        )
        config = _make_two_backend_config(
            routing_rules=(rule,),
            default_backends=(),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        for op_type in [
            EnumLlmOperationType.CHAT_COMPLETION,
            EnumLlmOperationType.COMPLETION,
            EnumLlmOperationType.EMBEDDING,
        ]:
            for cost_tier in [EnumCostTier.LOW, EnumCostTier.MID, EnumCostTier.HIGH]:
                request = _make_chat_request(
                    operation_type=op_type, cost_tier=cost_tier
                )
                candidate_ids, matched_rule = gateway._evaluate_rules(request)
                assert candidate_ids == ("backend-a", "backend-b"), (
                    f"Expected catch-all match for op={op_type} tier={cost_tier}"
                )
                assert matched_rule is not None
                assert matched_rule.rule_id == _RULE_CATCH_ALL

    def test_bifrost_routing_five_backends_per_ticket_contract(self) -> None:
        """Verify routing works across 5 backends as specified in OMN-2736."""
        backends = {
            f"backend-{i}": ModelBifrostBackendConfig(
                backend_id=f"backend-{i}",
                base_url=f"http://192.168.86.20{i}:8000",
                model_name=f"model-{i}",
            )
            for i in range(1, 6)  # 5 backends
        }
        # Tiered routing rules
        rules = (
            ModelBifrostRoutingRule(
                rule_id=_RULE_LOW_COST_5B,
                priority=10,
                match_cost_tiers=("low",),
                backend_ids=("backend-1", "backend-2"),
            ),
            ModelBifrostRoutingRule(
                rule_id=_RULE_MID_COST_14B,
                priority=20,
                match_cost_tiers=("mid",),
                backend_ids=("backend-3", "backend-4"),
            ),
            ModelBifrostRoutingRule(
                rule_id=_RULE_HIGH_COST_70B,
                priority=30,
                match_cost_tiers=("high",),
                backend_ids=("backend-5",),
            ),
        )
        config = ModelBifrostConfig(
            backends=backends,
            routing_rules=rules,
            default_backends=("backend-3",),
            failover_attempts=3,
            failover_backoff_base_ms=0,
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        # Low cost → backends 1,2
        req_low = _make_chat_request(cost_tier=EnumCostTier.LOW)
        ids, rule = gateway._evaluate_rules(req_low)
        assert "backend-1" in ids
        assert rule is not None and rule.rule_id == _RULE_LOW_COST_5B

        # Mid cost → backends 3,4
        req_mid = _make_chat_request(cost_tier=EnumCostTier.MID)
        ids, rule = gateway._evaluate_rules(req_mid)
        assert "backend-3" in ids
        assert rule is not None and rule.rule_id == _RULE_MID_COST_14B

        # High cost → backend 5
        req_high = _make_chat_request(cost_tier=EnumCostTier.HIGH)
        ids, rule = gateway._evaluate_rules(req_high)
        assert "backend-5" in ids
        assert rule is not None and rule.rule_id == _RULE_HIGH_COST_70B


# ---------------------------------------------------------------------------
# Tests: handle() end-to-end — routing decisions + audit fields
# ---------------------------------------------------------------------------


class TestBifrostRoutingHandleEndToEnd:
    """Tests that handle() returns correct audit fields for routing decisions."""

    @pytest.mark.asyncio
    async def test_bifrost_routing_handle_returns_correct_backend_and_rule(
        self,
    ) -> None:
        """handle() returns backend_selected matching the matched rule's first backend."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_CHAT_LOW,
            priority=10,
            match_cost_tiers=("low",),
            backend_ids=("backend-a",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule,),
            default_backends=("backend-b",),
        )
        response = _make_inference_response()
        handler = _make_handler(inference_response=response)
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request(cost_tier=EnumCostTier.LOW)
        result = await gateway.handle(request)

        assert result.success is True
        assert result.backend_selected == "backend-a"
        assert result.matched_rule_id == _RULE_CHAT_LOW
        assert result.retry_count == 0
        assert result.tenant_id == request.tenant_id

    @pytest.mark.asyncio
    async def test_bifrost_routing_handle_uses_default_when_no_rule_matches(
        self,
    ) -> None:
        """handle() uses default_backends when no rule matches, rule_id='default'."""
        config = _make_two_backend_config(
            routing_rules=(),
            default_backends=("backend-b",),
        )
        response = _make_inference_response()
        handler = _make_handler(inference_response=response)
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request()
        result = await gateway.handle(request)

        assert result.success is True
        assert result.backend_selected == "backend-b"
        assert result.matched_rule_id is None  # No rule matched — default backend used
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_bifrost_routing_handle_latency_ms_is_positive(self) -> None:
        """handle() records a positive latency_ms in the response."""
        config = _make_two_backend_config(
            routing_rules=(),
            default_backends=("backend-a",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request()
        result = await gateway.handle(request)

        assert result.latency_ms > 0.0

    @pytest.mark.asyncio
    async def test_bifrost_routing_handle_correlation_id_propagated(self) -> None:
        """handle() propagates the caller's correlation_id into the response."""
        config = _make_two_backend_config(
            routing_rules=(),
            default_backends=("backend-a",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        my_corr_id = UUID("00000000-0000-0000-0000-000000000099")
        request = ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
            messages=[{"role": "user", "content": "Hello"}],
            correlation_id=my_corr_id,
        )
        result = await gateway.handle(request)

        assert result.correlation_id == my_corr_id

    @pytest.mark.asyncio
    async def test_bifrost_routing_handle_auto_generates_correlation_id(self) -> None:
        """handle() auto-generates correlation_id when request has none."""
        config = _make_two_backend_config(
            routing_rules=(),
            default_backends=("backend-a",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        request = _make_chat_request()  # No correlation_id
        result = await gateway.handle(request)

        assert result.correlation_id != ""
        assert result.correlation_id is not None

    @pytest.mark.asyncio
    async def test_bifrost_routing_n_synthetic_requests_select_expected_backends(
        self,
    ) -> None:
        """Deterministic routing test: N synthetic requests select expected backends.

        DoD requirement: "Deterministic routing config test — N synthetic requests
        select expected backends: uv run pytest tests/unit/ -k bifrost_routing"
        """
        rule_chat = ModelBifrostRoutingRule(
            rule_id=_RULE_RULE_CHAT,
            priority=10,
            match_operation_types=(EnumLlmOperationType.CHAT_COMPLETION.value,),
            match_cost_tiers=("low", "mid"),
            backend_ids=("backend-a",),
        )
        rule_embed = ModelBifrostRoutingRule(
            rule_id=_RULE_RULE_EMBED,
            priority=20,
            match_operation_types=(EnumLlmOperationType.EMBEDDING.value,),
            backend_ids=("backend-b",),
        )
        config = _make_two_backend_config(
            routing_rules=(rule_chat, rule_embed),
            default_backends=("backend-b",),
        )
        handler = _make_handler()
        gateway = HandlerBifrostGateway(config=config, inference_handler=handler)

        test_cases: list[tuple[ModelBifrostRequest, str, UUID | None]] = [
            # (request, expected_backend, expected_matched_rule_id)
            (
                _make_chat_request(
                    operation_type=EnumLlmOperationType.CHAT_COMPLETION,
                    cost_tier=EnumCostTier.LOW,
                ),
                "backend-a",
                _RULE_RULE_CHAT,
            ),
            (
                _make_chat_request(
                    operation_type=EnumLlmOperationType.CHAT_COMPLETION,
                    cost_tier=EnumCostTier.MID,
                ),
                "backend-a",
                _RULE_RULE_CHAT,
            ),
            (
                _make_chat_request(operation_type=EnumLlmOperationType.EMBEDDING),
                "backend-b",
                _RULE_RULE_EMBED,
            ),
            (
                # HIGH cost chat — no rule matches (rule-chat only matches low/mid)
                _make_chat_request(
                    operation_type=EnumLlmOperationType.CHAT_COMPLETION,
                    cost_tier=EnumCostTier.HIGH,
                ),
                "backend-b",
                None,  # No rule matched — default backend used
            ),
        ]

        for req, expected_backend, expected_rule in test_cases:
            result = await gateway.handle(req)
            assert result.success is True, (
                f"Expected success for op={req.operation_type} tier={req.cost_tier}"
            )
            assert result.backend_selected == expected_backend, (
                f"op={req.operation_type} tier={req.cost_tier}: "
                f"expected backend={expected_backend}, got={result.backend_selected}"
            )
            assert result.matched_rule_id == expected_rule, (
                f"op={req.operation_type} tier={req.cost_tier}: "
                f"expected rule={expected_rule}, got={result.matched_rule_id}"
            )


# ---------------------------------------------------------------------------
# Tests: ModelBifrostRoutingRule — match predicate correctness
# ---------------------------------------------------------------------------


class TestBifrostRoutingRuleMatchPredicates:
    """Unit tests for _rule_matches static method."""

    def _gateway(self) -> HandlerBifrostGateway:
        config = _make_two_backend_config()
        handler = _make_handler()
        return HandlerBifrostGateway(config=config, inference_handler=handler)

    def test_bifrost_routing_empty_rule_matches_everything(self) -> None:
        """Rule with no predicates matches any request (wildcard)."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_WILDCARD,
            priority=0,
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        for op in [
            EnumLlmOperationType.CHAT_COMPLETION,
            EnumLlmOperationType.EMBEDDING,
            EnumLlmOperationType.COMPLETION,
        ]:
            for tier in [EnumCostTier.LOW, EnumCostTier.MID, EnumCostTier.HIGH]:
                request = _make_chat_request(operation_type=op, cost_tier=tier)
                assert gateway._rule_matches(rule, request) is True

    def test_bifrost_routing_operation_type_match(self) -> None:
        """match_operation_types filters by exact operation type."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_CHAT_ONLY,
            priority=0,
            match_operation_types=(EnumLlmOperationType.CHAT_COMPLETION.value,),
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        assert (
            gateway._rule_matches(
                rule,
                _make_chat_request(operation_type=EnumLlmOperationType.CHAT_COMPLETION),
            )
            is True
        )
        assert (
            gateway._rule_matches(
                rule, _make_chat_request(operation_type=EnumLlmOperationType.EMBEDDING)
            )
            is False
        )
        assert (
            gateway._rule_matches(
                rule, _make_chat_request(operation_type=EnumLlmOperationType.COMPLETION)
            )
            is False
        )

    def test_bifrost_routing_multi_operation_type_match(self) -> None:
        """match_operation_types with multiple values accepts any listed type."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_CHAT_OR_COMPLETE,
            priority=0,
            match_operation_types=(
                EnumLlmOperationType.CHAT_COMPLETION.value,
                EnumLlmOperationType.COMPLETION.value,
            ),
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        assert (
            gateway._rule_matches(
                rule,
                _make_chat_request(operation_type=EnumLlmOperationType.CHAT_COMPLETION),
            )
            is True
        )
        assert (
            gateway._rule_matches(
                rule, _make_chat_request(operation_type=EnumLlmOperationType.COMPLETION)
            )
            is True
        )
        assert (
            gateway._rule_matches(
                rule, _make_chat_request(operation_type=EnumLlmOperationType.EMBEDDING)
            )
            is False
        )

    def test_bifrost_routing_cost_tier_match(self) -> None:
        """match_cost_tiers filters by EnumCostTier value."""
        rule_low = ModelBifrostRoutingRule(
            rule_id=_RULE_LOW_ONLY,
            priority=0,
            match_cost_tiers=("low",),
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        assert (
            gateway._rule_matches(
                rule_low, _make_chat_request(cost_tier=EnumCostTier.LOW)
            )
            is True
        )
        assert (
            gateway._rule_matches(
                rule_low, _make_chat_request(cost_tier=EnumCostTier.MID)
            )
            is False
        )
        assert (
            gateway._rule_matches(
                rule_low, _make_chat_request(cost_tier=EnumCostTier.HIGH)
            )
            is False
        )

    def test_bifrost_routing_capability_subset_match(self) -> None:
        """match_capabilities requires all listed capabilities to be present."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_CAP_MATCH,
            priority=0,
            match_capabilities=("tool_calling",),
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        assert (
            gateway._rule_matches(
                rule, _make_chat_request(capabilities=("tool_calling",))
            )
            is True
        )
        # Extra capabilities are fine
        assert (
            gateway._rule_matches(
                rule, _make_chat_request(capabilities=("tool_calling", "json_mode"))
            )
            is True
        )
        assert (
            gateway._rule_matches(rule, _make_chat_request(capabilities=("json_mode",)))
            is False
        )
        assert gateway._rule_matches(rule, _make_chat_request(capabilities=())) is False

    def test_bifrost_routing_latency_constraint_boundary(self) -> None:
        """match_max_latency_ms_lte boundary condition: <= passes, > fails."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_LATENCY_1000,
            priority=0,
            match_max_latency_ms_lte=1000,
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        assert (
            gateway._rule_matches(rule, _make_chat_request(max_latency_ms=999)) is True
        )
        assert (
            gateway._rule_matches(rule, _make_chat_request(max_latency_ms=1000)) is True
        )  # boundary
        assert (
            gateway._rule_matches(rule, _make_chat_request(max_latency_ms=1001))
            is False
        )

    def test_bifrost_routing_combined_predicates_all_must_match(self) -> None:
        """Rule with multiple predicates requires ALL to match."""
        rule = ModelBifrostRoutingRule(
            rule_id=_RULE_COMBINED,
            priority=0,
            match_operation_types=(EnumLlmOperationType.CHAT_COMPLETION.value,),
            match_cost_tiers=("low",),
            match_capabilities=("tool_calling",),
            match_max_latency_ms_lte=2000,
            backend_ids=("backend-a",),
        )
        gateway = self._gateway()

        # All match
        full_match = _make_chat_request(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            cost_tier=EnumCostTier.LOW,
            capabilities=("tool_calling",),
            max_latency_ms=1500,
        )
        assert gateway._rule_matches(rule, full_match) is True

        # Wrong cost tier
        wrong_tier = _make_chat_request(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            cost_tier=EnumCostTier.HIGH,  # wrong
            capabilities=("tool_calling",),
            max_latency_ms=1500,
        )
        assert gateway._rule_matches(rule, wrong_tier) is False

        # Latency too high
        too_slow = _make_chat_request(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            cost_tier=EnumCostTier.LOW,
            capabilities=("tool_calling",),
            max_latency_ms=9999,  # exceeds 2000ms limit
        )
        assert gateway._rule_matches(rule, too_slow) is False
