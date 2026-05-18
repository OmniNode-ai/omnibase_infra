# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for KafkaContractSource.materialize_cached_contract (OMN-11244)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.models.handlers import ModelHandlerDescriptor
from omnibase_infra.runtime.enums.enum_materialization_rejection import (
    EnumMaterializationRejection,
)
from omnibase_infra.runtime.enums.enum_materialization_status import (
    EnumMaterializationStatus,
)
from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

pytestmark = pytest.mark.unit


_WIRE_PATCH = "omnibase_infra.runtime.auto_wiring.handler_wiring._wire_single_contract"


def _make_descriptor(
    handler_id: str = "proto.test_dynamic",
    name: str = "test_dynamic",
    contract_path: str = "kafka://dev/contracts/test_dynamic",
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
        contract_path=contract_path,
        contract_config=contract_config,
    )


def _make_wiring_result(
    outcome_value: str = "wired",
    contract_name: str = "test_dynamic",
    dispatchers: tuple[str, ...] = ("dispatcher.test",),
    topics: tuple[str, ...] = ("onex.evt.test.topic.v1",),
) -> MagicMock:
    from omnibase_infra.runtime.auto_wiring.report import (
        EnumWiringOutcome,
        ModelContractWiringResult,
    )

    outcome = EnumWiringOutcome(outcome_value)
    return ModelContractWiringResult(
        contract_name=contract_name,
        package_name="dynamic",
        outcome=outcome,
        dispatchers_registered=dispatchers,
        topics_subscribed=topics,
    )


class TestMaterializeCachedContractNotCached:
    """materialize_cached_contract returns REJECTED when node not in cache."""

    def test_returns_rejected_for_unknown_node(self) -> None:
        source = KafkaContractSource(environment="dev")
        dispatch_engine = MagicMock()
        event_bus = MagicMock()

        result = asyncio.run(
            source.materialize_cached_contract(
                node_name="nonexistent.node",
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
            )
        )
        assert result.status == EnumMaterializationStatus.REJECTED
        assert result.reason == EnumMaterializationRejection.PARSE_FAILURE
        assert result.contract_name == "nonexistent.node"


class TestMaterializeCachedContractIdempotency:
    """Same (node_name, contract_hash) is wired exactly once."""

    def test_second_call_returns_already_materialized(self) -> None:
        source = KafkaContractSource(environment="dev")
        descriptor = _make_descriptor()
        source._cache.add("test_dynamic", descriptor)

        dispatch_engine = MagicMock()
        event_bus = AsyncMock()
        wiring_result = _make_wiring_result()

        with patch(
            _WIRE_PATCH,
            new_callable=AsyncMock,
            return_value=wiring_result,
        ) as mock_wire:
            result1 = asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=dispatch_engine,
                    event_bus=event_bus,
                )
            )
            assert result1.status == EnumMaterializationStatus.MATERIALIZED
            assert mock_wire.call_count == 1

            result2 = asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=dispatch_engine,
                    event_bus=event_bus,
                )
            )
            assert result2.status == EnumMaterializationStatus.ALREADY_MATERIALIZED
            assert mock_wire.call_count == 1  # not called again


class TestMaterializeCachedContractVersionConflict:
    """Same node_name with different hash is rejected as VERSION_CONFLICT."""

    def test_version_conflict_rejected(self) -> None:
        source = KafkaContractSource(environment="dev")

        descriptor1 = _make_descriptor()
        source._cache.add("test_dynamic", descriptor1)

        # Inject a different hash for the same node name to simulate a
        # version already materialized under a different contract revision.
        fake_hash = "sha256:aabbccddeeff00112233445566778899"
        source._materialized_contracts.add(("test_dynamic", fake_hash))

        # Now try to materialize — the canonical hash won't match fake_hash.
        result = asyncio.run(
            source.materialize_cached_contract(
                node_name="test_dynamic",
                dispatch_engine=MagicMock(),
                event_bus=MagicMock(),
            )
        )
        assert result.status == EnumMaterializationStatus.REJECTED
        assert result.reason == EnumMaterializationRejection.VERSION_CONFLICT


class TestMaterializeCachedContractSuccess:
    """Happy-path: descriptor wired into dispatch engine, result populated."""

    def test_materialized_result_fields(self) -> None:
        source = KafkaContractSource(environment="dev")
        descriptor = _make_descriptor()
        source._cache.add("test_dynamic", descriptor)

        dispatchers = ("dispatcher.effect_test",)
        topics = ("onex.evt.test.dynamic.v1",)
        wiring_result = _make_wiring_result(dispatchers=dispatchers, topics=topics)

        with patch(
            _WIRE_PATCH,
            new_callable=AsyncMock,
            return_value=wiring_result,
        ):
            result = asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert result.status == EnumMaterializationStatus.MATERIALIZED
        assert result.contract_name == "test_dynamic"
        assert result.contract_hash.startswith("sha256:")
        assert result.subscribed_topics == topics
        assert result.registered_handlers == dispatchers
        assert result.materialization_correlation_id is not None

    def test_materialized_contract_added_to_tracking_set(self) -> None:
        source = KafkaContractSource(environment="dev")
        descriptor = _make_descriptor()
        source._cache.add("test_dynamic", descriptor)

        with patch(
            _WIRE_PATCH,
            new_callable=AsyncMock,
            return_value=_make_wiring_result(),
        ):
            asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert len(source._materialized_contracts) == 1
        node_names = {n for (n, _) in source._materialized_contracts}
        assert "test_dynamic" in node_names


class TestMaterializeCachedContractWiringSkipped:
    """When _wire_single_contract returns SKIPPED, result is REJECTED."""

    def test_skipped_wiring_returns_rejected(self) -> None:
        source = KafkaContractSource(environment="dev")
        descriptor = _make_descriptor()
        source._cache.add("test_dynamic", descriptor)

        wiring_result = _make_wiring_result(outcome_value="skipped")

        with patch(
            _WIRE_PATCH,
            new_callable=AsyncMock,
            return_value=wiring_result,
        ):
            result = asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert result.status == EnumMaterializationStatus.REJECTED
        assert result.reason is None
        # Should NOT have been added to tracking set
        assert len(source._materialized_contracts) == 0


class TestMaterializeCachedContractWiringException:
    """Exceptions from _wire_single_contract return REJECTED gracefully."""

    def test_exception_returns_rejected(self) -> None:
        source = KafkaContractSource(environment="dev")
        descriptor = _make_descriptor()
        source._cache.add("test_dynamic", descriptor)

        with patch(
            _WIRE_PATCH,
            new_callable=AsyncMock,
            side_effect=RuntimeError("dispatch engine is frozen"),
        ):
            result = asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert result.status == EnumMaterializationStatus.REJECTED
        assert len(source._materialized_contracts) == 0


class TestMaterializeCachedContractEventBusConfig:
    """Contracts with event_bus config pass correct wiring to _wire_single_contract."""

    def test_event_bus_topics_passed_to_wiring(self) -> None:
        source = KafkaContractSource(environment="dev")
        config = {
            "event_bus": {
                "subscribe_topics": ["onex.evt.test.foo.v1"],
                "publish_topics": ["onex.evt.test.bar.v1"],
            }
        }
        descriptor = _make_descriptor(contract_config=config)
        source._cache.add("test_dynamic", descriptor)

        captured: list[object] = []

        async def _capture_wire(**kwargs: object) -> object:
            captured.append(kwargs["contract"])
            return _make_wiring_result()

        with patch(
            _WIRE_PATCH,
            side_effect=_capture_wire,
        ):
            asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert len(captured) == 1
        contract = captured[0]
        assert hasattr(contract, "event_bus")
        assert contract.event_bus is not None
        assert "onex.evt.test.foo.v1" in contract.event_bus.subscribe_topics


class TestMaterializeCachedContractHandlerRouting:
    """Contracts with handler_routing config pass correct routing entries."""

    def test_handler_routing_passed_to_wiring(self) -> None:
        source = KafkaContractSource(environment="dev")
        config = {
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "handlers": [
                    {
                        "handler": {
                            "name": "HandlerHttp",
                            "module": "omnibase_infra.handlers.handler_http",
                        },
                        "event_model": {
                            "name": "ModelTestEvent",
                            "module": "omnibase_infra.models.TestEvent",
                        },
                    }
                ],
            }
        }
        descriptor = _make_descriptor(contract_config=config)
        source._cache.add("test_dynamic", descriptor)

        captured: list[object] = []

        async def _capture_wire(**kwargs: object) -> object:
            captured.append(kwargs["contract"])
            return _make_wiring_result()

        with patch(
            _WIRE_PATCH,
            side_effect=_capture_wire,
        ):
            asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert len(captured) == 1
        contract = captured[0]
        assert contract.handler_routing is not None
        assert contract.handler_routing.routing_strategy == "payload_type_match"
        assert len(contract.handler_routing.handlers) == 1
        assert contract.handler_routing.handlers[0].handler.name == "HandlerHttp"


class TestMaterializeCachedContractSyntheticPath:
    """Synthetic contract_path uses /kafka/ prefix for Kafka-sourced contracts."""

    def test_contract_path_is_synthetic(self) -> None:
        source = KafkaContractSource(environment="staging")
        descriptor = _make_descriptor()
        source._cache.add("test_dynamic", descriptor)

        captured: list[object] = []

        async def _capture_wire(**kwargs: object) -> object:
            captured.append(kwargs["contract"])
            return _make_wiring_result()

        with patch(
            _WIRE_PATCH,
            side_effect=_capture_wire,
        ):
            asyncio.run(
                source.materialize_cached_contract(
                    node_name="test_dynamic",
                    dispatch_engine=MagicMock(),
                    event_bus=AsyncMock(),
                )
            )

        assert len(captured) == 1
        contract_path = captured[0].contract_path
        assert isinstance(contract_path, Path)
        assert "kafka" in str(contract_path)
        assert "staging" in str(contract_path)
        assert "test_dynamic" in str(contract_path)
