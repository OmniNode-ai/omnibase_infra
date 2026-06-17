# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for KafkaContractSource.materialize_cached_contract (OMN-11244).

Tests the end-to-end in-memory flow: cache population → contract hash computation
→ wiring dispatch → idempotency guard. No live Kafka required — these tests exercise
the materialization path from a populated cache through to the dispatch engine.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.models.handlers import ModelHandlerDescriptor
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.enums.enum_materialization_rejection import (
    EnumMaterializationRejection,
)
from omnibase_infra.runtime.enums.enum_materialization_status import (
    EnumMaterializationStatus,
)
from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

pytestmark = pytest.mark.integration

_WIRE_PATCH = "omnibase_infra.runtime.auto_wiring.handler_wiring._wire_single_contract"


def _make_descriptor(
    handler_id: str = "proto.int_test",
    name: str = "int_test",
    contract_config: dict | None = None,
) -> ModelHandlerDescriptor:
    return ModelHandlerDescriptor(
        handler_id=handler_id,
        name=name,
        version="1.0.0",
        handler_kind="effect",
        input_model="omnibase_infra.models.TestInput",
        output_model="omnibase_infra.models.TestOutput",
        handler_class="omnibase_infra.handlers.handler_http.HandlerHttp",
        contract_path=f"kafka://integration/{name}",
        contract_config=contract_config,
    )


def _make_wiring_result(
    outcome_value: str = "wired",
    contract_name: str = "int_test",
    dispatchers: tuple[str, ...] = ("dispatcher.int_test",),
    topics: tuple[str, ...] = ("onex.evt.integration.test.v1",),
) -> object:
    from omnibase_infra.runtime.auto_wiring.report import (
        EnumWiringOutcome,
        ModelContractWiringResult,
    )

    return ModelContractWiringResult(
        contract_name=contract_name,
        package_name="integration",
        outcome=EnumWiringOutcome(outcome_value),
        dispatchers_registered=dispatchers,
        topics_subscribed=topics,
    )


@pytest.mark.integration
def test_materialize_cached_contract_full_flow() -> None:
    """Happy path: descriptor in cache → wired → MATERIALIZED result with all fields."""
    source = KafkaContractSource(environment="integration")
    descriptor = _make_descriptor()
    source._cache.add("int_test", descriptor)

    dispatchers = ("dispatcher.int_test",)
    topics = ("onex.evt.integration.test.v1",)
    wiring_result = _make_wiring_result(dispatchers=dispatchers, topics=topics)

    with patch(_WIRE_PATCH, new_callable=AsyncMock, return_value=wiring_result):
        result = asyncio.run(
            source.materialize_cached_contract(
                node_name="int_test",
                dispatch_engine=MagicMock(),
                event_bus=AsyncMock(spec=ProtocolEventBusLike),
            )
        )

    assert result.status == EnumMaterializationStatus.MATERIALIZED
    assert result.contract_name == "int_test"
    assert result.contract_hash.startswith("sha256:")
    assert len(result.contract_hash) > 7
    assert result.subscribed_topics == topics
    assert result.registered_handlers == dispatchers
    assert result.materialization_correlation_id is not None
    assert result.reason is None


@pytest.mark.integration
def test_materialize_cached_contract_idempotency_gate() -> None:
    """Second call with identical (node_name, hash) returns ALREADY_MATERIALIZED without re-wiring."""
    source = KafkaContractSource(environment="integration")
    descriptor = _make_descriptor()
    source._cache.add("int_test", descriptor)

    wiring_result = _make_wiring_result()

    with patch(
        _WIRE_PATCH, new_callable=AsyncMock, return_value=wiring_result
    ) as mock_wire:
        result1 = asyncio.run(
            source.materialize_cached_contract(
                node_name="int_test",
                dispatch_engine=MagicMock(),
                event_bus=AsyncMock(spec=ProtocolEventBusLike),
            )
        )
        result2 = asyncio.run(
            source.materialize_cached_contract(
                node_name="int_test",
                dispatch_engine=MagicMock(),
                event_bus=AsyncMock(spec=ProtocolEventBusLike),
            )
        )

    assert result1.status == EnumMaterializationStatus.MATERIALIZED
    assert result2.status == EnumMaterializationStatus.ALREADY_MATERIALIZED
    assert mock_wire.call_count == 1


@pytest.mark.integration
def test_materialize_cached_contract_version_conflict() -> None:
    """Different hash for same node_name is rejected as VERSION_CONFLICT."""
    source = KafkaContractSource(environment="integration")
    descriptor = _make_descriptor()
    source._cache.add("int_test", descriptor)

    # Inject a stale hash to simulate a prior version already materialized.
    source._materialized_contracts.add(("int_test", "sha256:000000deadbeef"))

    result = asyncio.run(
        source.materialize_cached_contract(
            node_name="int_test",
            dispatch_engine=MagicMock(),
            event_bus=AsyncMock(spec=ProtocolEventBusLike),
        )
    )

    assert result.status == EnumMaterializationStatus.REJECTED
    assert result.reason == EnumMaterializationRejection.VERSION_CONFLICT


@pytest.mark.integration
def test_materialize_cached_contract_tracking_set_populated() -> None:
    """After successful materialization the (node_name, hash) key is in tracking set."""
    source = KafkaContractSource(environment="integration")
    descriptor = _make_descriptor()
    source._cache.add("int_test", descriptor)

    with patch(_WIRE_PATCH, new_callable=AsyncMock, return_value=_make_wiring_result()):
        asyncio.run(
            source.materialize_cached_contract(
                node_name="int_test",
                dispatch_engine=MagicMock(),
                event_bus=AsyncMock(spec=ProtocolEventBusLike),
            )
        )

    assert len(source._materialized_contracts) == 1
    node_names = {n for (n, _) in source._materialized_contracts}
    assert "int_test" in node_names


@pytest.mark.integration
def test_materialize_cached_contract_wiring_failure_does_not_persist() -> None:
    """An exception during wiring leaves no entry in the tracking set (retry possible)."""
    source = KafkaContractSource(environment="integration")
    descriptor = _make_descriptor()
    source._cache.add("int_test", descriptor)

    with patch(
        _WIRE_PATCH,
        new_callable=AsyncMock,
        side_effect=RuntimeError("wiring subsystem unavailable"),
    ):
        result = asyncio.run(
            source.materialize_cached_contract(
                node_name="int_test",
                dispatch_engine=MagicMock(),
                event_bus=AsyncMock(spec=ProtocolEventBusLike),
            )
        )

    assert result.status == EnumMaterializationStatus.REJECTED
    assert len(source._materialized_contracts) == 0
