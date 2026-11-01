# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed retrieval and deactivation requests cross the shared runtime boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    ProtocolLocalRuntimeBus,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_callable_target import (
    ProtocolLocalRuntimeCallableTarget,
)
from omnibase_core.runtime.runtime_local_adapter import LocalRuntimeBusAdapter
from omnibase_infra.models import ModelBackendResult
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_retrieval_complete import (
    HandlerChainRetrievalComplete,
)
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainEntry,
    ModelChainMatch,
    ModelChainRetrievalResult,
)
from omnibase_infra.nodes.node_registry_effect.handlers.handler_postgres_deactivate import (
    HandlerPostgresDeactivate,
)
from omnibase_infra.nodes.node_registry_effect.models import ModelRegistryRequest
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("is_hit", [False, True])
async def test_retrieval_request_dispatches_and_publishes(is_hit: bool) -> None:
    chain_id = uuid4()
    match = ModelChainMatch(
        chain_entry=ModelChainEntry(
            chain_id=chain_id,
            prompt_text="learn this workflow",
            prompt_hash="prompt-hash",
            chain_steps=(),
            contract_hash="contract-hash",
            success_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            workflow_ref="workflow-omn21761",
        ),
        similarity_score=0.95,
        distance=0.05,
    )
    request = ModelChainRetrievalResult(
        correlation_id=uuid4(),
        is_hit=is_hit,
        matches=(match,) if is_hit else (),
        best_match_similarity=0.95 if is_hit else 0.0,
    )
    bus = AsyncMock(spec=ProtocolLocalRuntimeBus)
    on_error = Mock()
    adapter = LocalRuntimeBusAdapter(
        handler=cast(
            "ProtocolLocalRuntimeCallableTarget", HandlerChainRetrievalComplete()
        ),
        handler_name="HandlerChainRetrievalComplete",
        input_model_cls=ModelChainRetrievalResult,
        output_topic="onex.evt.omnibase-infra.chain-decision.v1",
        bus=bus,
        on_error=on_error,
    )

    await adapter.on_message(
        SimpleNamespace(value=request.model_dump_json().encode("utf-8"))
    )

    on_error.assert_not_called()
    bus.publish.assert_awaited_once()
    topic, key, payload = bus.publish.call_args.args
    assert topic == "onex.evt.omnibase-infra.chain-decision.v1"
    assert key is None
    decision = json.loads(payload)
    assert decision["path"] == ("replay" if is_hit else "explore")
    assert decision["is_hit"] is is_hit
    if is_hit:
        assert decision["chain_id"] == str(chain_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "refused", "timeout"])
async def test_deactivation_request_preserves_correlation(outcome: str) -> None:
    request = ModelRegistryRequest(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        correlation_id=uuid4(),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    postgres = AsyncMock(spec=ProtocolPostgresAdapter)
    postgres.deactivate.return_value = ModelBackendResult(
        success=outcome == "success",
        backend_id="postgres",
        error="deactivation refused" if outcome == "refused" else None,
    )
    if outcome == "timeout":
        postgres.deactivate.side_effect = TimeoutError("deactivation timed out")
    bus = AsyncMock(spec=ProtocolLocalRuntimeBus)
    on_error = Mock()
    adapter = LocalRuntimeBusAdapter(
        handler=cast(
            "ProtocolLocalRuntimeCallableTarget", HandlerPostgresDeactivate(postgres)
        ),
        handler_name="HandlerPostgresDeactivate",
        input_model_cls=ModelRegistryRequest,
        output_topic="onex.evt.omnibase-infra.node-deactivated.v1",
        bus=bus,
        on_error=on_error,
    )

    await adapter.on_message(
        SimpleNamespace(value=request.model_dump_json().encode("utf-8"))
    )

    on_error.assert_not_called()
    postgres.deactivate.assert_awaited_once_with(node_id=request.node_id)
    bus.publish.assert_awaited_once()
    result = json.loads(bus.publish.call_args.args[2])
    assert result["correlation_id"] == str(request.correlation_id)
    assert result["success"] is (outcome == "success")
    assert result["backend_id"] == "postgres"
    if outcome == "timeout":
        assert result["error_code"] == "POSTGRES_TIMEOUT_ERROR"
    elif outcome == "refused":
        assert result["error_code"] == "POSTGRES_DEACTIVATION_ERROR"
