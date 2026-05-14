# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Regression tests: max_tokens must propagate unchanged through the delegation chain.

Covers the full path:
  ModelDelegationRequest.max_tokens
    → HandlerDelegationWorkflow (handle_routing_decision)
    → ModelInferenceIntent.max_tokens
    → LlmCallerDelegation.call
    → ModelLlmAdapterRequest.max_tokens
    → AdapterLlmProviderOpenai._translate_request
    → ModelLlmInferenceRequest.max_tokens

The regression: live delegation requests were observed capped at 512 tokens.
This suite proves no layer in the chain silently drops or caps the value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.adapters.llm.adapter_llm_caller_delegation import (
    LlmCallerDelegation,
)
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
    HandlerDelegationWorkflow,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_intent import (
    ModelInferenceIntent,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)


def _make_delegation_request(max_tokens: int) -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt="Write a pytest test for the add() function.",
        task_type="test",
        correlation_id=uuid4(),
        max_tokens=max_tokens,
        emitted_at=datetime.now(UTC),
    )


def _make_routing_decision(correlation_id: object) -> ModelRoutingDecision:
    return ModelRoutingDecision(
        correlation_id=correlation_id,  # type: ignore[arg-type]
        task_type="test",
        selected_model="qwen3-coder-30b",
        selected_backend_id=uuid4(),
        endpoint_url="http://192.168.86.201:8000",
        cost_tier="low",
        max_context_tokens=112_000,
        system_prompt="You are a code generation assistant.",
        rationale="Test task routed to local coder model.",
    )


def _make_adapter_response() -> ModelLlmAdapterResponse:
    return ModelLlmAdapterResponse(
        generated_text="def test_add():\n    assert add(1, 2) == 3",
        model_used="qwen3-coder-30b",
        usage_statistics={
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
        },
        finish_reason="stop",
        response_metadata={"latency_ms": 42, "provider_id": "", "correlation_id": ""},
    )


# ---------------------------------------------------------------------------
# 1. ModelDelegationRequest field: no cap at model definition level
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("max_tokens", [512, 1024, 2048, 4096, 8192, 16384])
def test_delegation_request_accepts_arbitrary_max_tokens(max_tokens: int) -> None:
    """ModelDelegationRequest must accept any positive max_tokens without capping."""
    req = _make_delegation_request(max_tokens)
    assert req.max_tokens == max_tokens


# ---------------------------------------------------------------------------
# 2. HandlerDelegationWorkflow: max_tokens flows into ModelInferenceIntent
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("max_tokens", [1024, 2048, 4096, 8192])
def test_handler_propagates_max_tokens_to_inference_intent(max_tokens: int) -> None:
    """handle_routing_decision must carry max_tokens from request → ModelInferenceIntent."""
    handler = HandlerDelegationWorkflow()
    request = _make_delegation_request(max_tokens)
    cid = request.correlation_id

    # Register the request
    handler.handle_delegation_request(request)

    # Simulate the routing reducer emitting a decision
    decision = _make_routing_decision(cid)
    intents = handler.handle_routing_decision(decision)

    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, ModelInferenceIntent)
    assert intent.max_tokens == max_tokens, (
        f"max_tokens={max_tokens} was not propagated to ModelInferenceIntent; "
        f"got {intent.max_tokens}"
    )


# ---------------------------------------------------------------------------
# 3. Regression: max_tokens must NOT be silently capped at 512
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("max_tokens", [1024, 2048, 4096, 8192])
def test_max_tokens_not_capped_at_512(max_tokens: int) -> None:
    """Regression: no layer may silently cap max_tokens at 512 (or any value < requested)."""
    handler = HandlerDelegationWorkflow()
    request = _make_delegation_request(max_tokens)
    handler.handle_delegation_request(request)
    decision = _make_routing_decision(request.correlation_id)
    intents = handler.handle_routing_decision(decision)

    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, ModelInferenceIntent)
    assert intent.max_tokens >= max_tokens, (
        f"max_tokens was capped: requested {max_tokens}, got {intent.max_tokens}"
    )
    assert intent.max_tokens == max_tokens, (
        f"max_tokens was modified: requested {max_tokens}, got {intent.max_tokens}"
    )


# ---------------------------------------------------------------------------
# 4. LlmCallerDelegation: max_tokens from intent reaches ModelLlmAdapterRequest
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [1024, 2048, 4096, 8192])
async def test_llm_caller_propagates_max_tokens_to_adapter_request(
    max_tokens: int,
) -> None:
    """LlmCallerDelegation.call() must pass intent.max_tokens into ModelLlmAdapterRequest."""
    intent = ModelInferenceIntent(
        base_url="http://192.168.86.201:8000",
        model="qwen3-coder-30b",
        system_prompt="Be concise.",
        prompt="Write a test.",
        max_tokens=max_tokens,
        temperature=0.3,
        correlation_id=uuid4(),
    )
    adapter_response = _make_adapter_response()
    caller = LlmCallerDelegation()

    captured_requests: list[ModelLlmAdapterRequest] = []

    async def _capture(req: ModelLlmAdapterRequest) -> ModelLlmAdapterResponse:
        captured_requests.append(req)
        return adapter_response

    with patch(
        "omnibase_infra.adapters.llm.adapter_llm_caller_delegation.AdapterLlmProviderOpenai"
    ) as MockProvider:
        instance = MagicMock()
        instance.generate_async = _capture
        instance.close = AsyncMock()
        MockProvider.return_value = instance

        await caller.call(intent)

    assert len(captured_requests) == 1
    adapter_req = captured_requests[0]
    assert adapter_req.max_tokens == max_tokens, (
        f"LlmCallerDelegation did not propagate max_tokens={max_tokens} to "
        f"ModelLlmAdapterRequest; got {adapter_req.max_tokens}"
    )


# ---------------------------------------------------------------------------
# 5. End-to-end chain: ModelDelegationRequest → ModelLlmAdapterRequest
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [1024, 2048, 4096, 8192])
async def test_end_to_end_max_tokens_chain(max_tokens: int) -> None:
    """Full chain: ModelDelegationRequest.max_tokens reaches the LLM adapter call unchanged."""
    # 1. Build and register a delegation request
    handler = HandlerDelegationWorkflow()
    request = _make_delegation_request(max_tokens)
    handler.handle_delegation_request(request)

    # 2. Routing reducer emits a decision
    decision = _make_routing_decision(request.correlation_id)
    intents = handler.handle_routing_decision(decision)

    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, ModelInferenceIntent)
    assert intent.max_tokens == max_tokens

    # 3. LLM caller translates intent → adapter request
    caller = LlmCallerDelegation()
    adapter_response = _make_adapter_response()
    captured_requests: list[ModelLlmAdapterRequest] = []

    async def _capture(req: ModelLlmAdapterRequest) -> ModelLlmAdapterResponse:
        captured_requests.append(req)
        return adapter_response

    with patch(
        "omnibase_infra.adapters.llm.adapter_llm_caller_delegation.AdapterLlmProviderOpenai"
    ) as MockProvider:
        instance = MagicMock()
        instance.generate_async = _capture
        instance.close = AsyncMock()
        MockProvider.return_value = instance

        await caller.call(intent)

    assert len(captured_requests) == 1
    adapter_req = captured_requests[0]
    assert adapter_req.max_tokens == max_tokens, (
        f"End-to-end: max_tokens={max_tokens} was lost or mutated; "
        f"ModelLlmAdapterRequest.max_tokens={adapter_req.max_tokens}"
    )
