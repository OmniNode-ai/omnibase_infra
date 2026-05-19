# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""E2E integration test for dynamic contract registration (OMN-11248).

Proves the full dynamic registration flow with real MessageDispatchEngine
and EventBusInmemory (no mocks on the wiring path):

    register contract -> materialize -> dispatcher in engine
    -> topic subscribed -> idempotency -> deregistration -> version conflict
    -> malformed YAML

Uses HandlerNoop (tests.fixtures.handler_noop) as the in-process handler
to avoid network dependencies.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.enums.enum_materialization_rejection import (
    EnumMaterializationRejection,
)
from omnibase_infra.runtime.enums.enum_materialization_status import (
    EnumMaterializationStatus,
)
from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Minimal contract YAML for a Noop in-process handler.
# handler_class in metadata so ContractYamlParser can extract it.
# NOTE: topic strings here are contract-declared per ONEX convention.
# ---------------------------------------------------------------------------

_NOOP_CONTRACT_YAML = """\
handler_id: proto.e2e_dynamic
name: node_e2e_dynamic
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  node_archetype: effect
input_model: omnibase_infra.models.types.JsonDict
output_model: omnibase_core.models.dispatch.model_handler_output.ModelHandlerOutput
description: E2E test for dynamic contract registration
metadata:
  handler_class: tests.fixtures.handler_noop.HandlerNoop
event_bus:
  subscribe_topics:
    - onex.evt.test.e2e-dynamic.v1
  publish_topics: []
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerNoop
        module: tests.fixtures.handler_noop
"""

_EVIL_MODULE_CONTRACT_YAML = (
    _NOOP_CONTRACT_YAML.replace(
        "module: tests.fixtures.handler_noop",
        "module: os",
    )
    .replace(
        "name: HandlerNoop",
        "name: system",
    )
    .replace(
        "handler_class: tests.fixtures.handler_noop.HandlerNoop",
        "handler_class: os.system",
    )
)

_MALFORMED_YAML = "this: is: [not: valid: yaml: {{"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> KafkaContractSource:
    return KafkaContractSource(environment="test", graceful_mode=True)


def _make_engine() -> MessageDispatchEngine:
    return MessageDispatchEngine()


def _make_bus() -> EventBusInmemory:
    return EventBusInmemory(environment="test")


# ---------------------------------------------------------------------------
# Test: full registration flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_then_materialize_wires_dispatcher_and_topic() -> None:
    """Full flow: register -> materialize -> dispatcher in engine -> topic subscribed."""
    source = _make_source()
    engine = _make_engine()
    bus = _make_bus()

    # Step 1: cache the contract.
    cached = source.on_contract_registered(
        node_name="node_e2e_dynamic",
        contract_yaml=_NOOP_CONTRACT_YAML,
        correlation_id=uuid4(),
    )
    assert cached is True
    assert source.cached_count == 1

    descriptor = source.get_cached_descriptor("node_e2e_dynamic")
    assert descriptor is not None
    assert descriptor.handler_id == "proto.e2e_dynamic"

    # Step 2: materialize into live engine.
    result = await source.materialize_cached_contract(
        node_name="node_e2e_dynamic",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result.status == EnumMaterializationStatus.MATERIALIZED
    assert result.contract_name == "node_e2e_dynamic"
    assert result.contract_hash.startswith("sha256:")
    assert result.materialization_correlation_id is not None

    # Step 3: verify dispatcher registered in engine.
    # _dispatchers/_routes inspection is test-only.
    dynamic_dispatcher_ids = [
        did
        for did in engine._dispatchers
        if "e2e_dynamic" in did or "HandlerNoop" in did
    ]
    assert len(dynamic_dispatcher_ids) > 0, (
        f"Expected dynamic dispatcher, found: {list(engine._dispatchers.keys())}"
    )

    # Step 4: verify at least one topic was subscribed via EventBusInmemory.
    subscribed = await bus.get_topics()
    assert "onex.evt.test.e2e-dynamic.v1" in subscribed, (
        f"Expected topic subscription, found: {subscribed}"
    )

    # Step 5: result.subscribed_topics reflects the wiring.
    assert "onex.evt.test.e2e-dynamic.v1" in result.subscribed_topics


# ---------------------------------------------------------------------------
# Test: idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_materialize_is_already_materialized() -> None:
    """Second materialize_cached_contract call for same node is ALREADY_MATERIALIZED."""
    source = _make_source()
    engine = _make_engine()
    bus = _make_bus()

    source.on_contract_registered(
        node_name="node_e2e_dynamic",
        contract_yaml=_NOOP_CONTRACT_YAML,
        correlation_id=uuid4(),
    )

    result1 = await source.materialize_cached_contract(
        node_name="node_e2e_dynamic",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result1.status == EnumMaterializationStatus.MATERIALIZED

    dispatchers_after_first = set(engine._dispatchers)

    result2 = await source.materialize_cached_contract(
        node_name="node_e2e_dynamic",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result2.status == EnumMaterializationStatus.ALREADY_MATERIALIZED

    # Dispatcher count must not have changed.
    assert set(engine._dispatchers) == dispatchers_after_first


# ---------------------------------------------------------------------------
# Test: deregistration removes cache but NOT live dispatchers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deregistration_removes_cache_not_live_dispatchers() -> None:
    """Deregistration removes descriptor from cache; live dispatchers remain (MVP)."""
    source = _make_source()
    engine = _make_engine()
    bus = _make_bus()
    correlation_id = uuid4()

    dereg_yaml = _NOOP_CONTRACT_YAML.replace(
        "node_e2e_dynamic", "node_e2e_dereg"
    ).replace("proto.e2e_dynamic", "proto.e2e_dereg")

    source.on_contract_registered(
        node_name="node_e2e_dereg",
        contract_yaml=dereg_yaml,
        correlation_id=correlation_id,
    )

    result = await source.materialize_cached_contract(
        node_name="node_e2e_dereg",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result.status == EnumMaterializationStatus.MATERIALIZED

    dispatchers_before = set(engine._dispatchers)
    assert len(dispatchers_before) > 0

    # Deregister from cache.
    removed = source.on_contract_deregistered(
        node_name="node_e2e_dereg",
        correlation_id=correlation_id,
    )
    assert removed is True
    assert source.cached_count == 0

    # Live dispatchers must still exist after cache deregistration (MVP: no hot-unload).
    assert set(engine._dispatchers) == dispatchers_before


# ---------------------------------------------------------------------------
# Test: version conflict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_version_conflict_rejected() -> None:
    """Same node_name re-materialized with different hash -> VERSION_CONFLICT."""
    source = _make_source()
    engine = _make_engine()
    bus = _make_bus()

    versioned_yaml = _NOOP_CONTRACT_YAML.replace(
        "node_e2e_dynamic", "node_versioned"
    ).replace("proto.e2e_dynamic", "proto.versioned")

    source.on_contract_registered(
        node_name="node_versioned",
        contract_yaml=versioned_yaml,
        correlation_id=uuid4(),
    )
    result1 = await source.materialize_cached_contract(
        node_name="node_versioned",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result1.status == EnumMaterializationStatus.MATERIALIZED

    # Inject a fake pre-existing hash to force a conflict on the next call.
    # (materialize_cached_contract computes a canonical hash from contract_path;
    # injecting a different hash for the same node_name triggers VERSION_CONFLICT.)
    fake_hash = "sha256:" + "a" * 64
    source._materialized_contracts.discard(
        next(e for e in source._materialized_contracts if e[0] == "node_versioned")
    )
    source._materialized_contracts.add(("node_versioned", fake_hash))

    result2 = await source.materialize_cached_contract(
        node_name="node_versioned",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result2.status == EnumMaterializationStatus.REJECTED
    assert result2.reason == EnumMaterializationRejection.VERSION_CONFLICT


# ---------------------------------------------------------------------------
# Test: malformed YAML
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_yaml_does_not_crash() -> None:
    """Malformed YAML: on_contract_registered returns False; materialize returns REJECTED."""
    source = _make_source()
    engine = _make_engine()

    cached = source.on_contract_registered(
        node_name="bad_node",
        contract_yaml=_MALFORMED_YAML,
        correlation_id=uuid4(),
    )
    assert cached is False
    assert source.cached_count == 0

    # Not in cache -> REJECTED with PARSE_FAILURE reason.
    result = await source.materialize_cached_contract(
        node_name="bad_node",
        dispatch_engine=engine,
    )
    assert result.status == EnumMaterializationStatus.REJECTED
    assert result.reason == EnumMaterializationRejection.PARSE_FAILURE


# ---------------------------------------------------------------------------
# Test: handler module outside allowed namespaces is rejected by wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_outside_allowed_namespace_wiring_fails_gracefully() -> None:
    """Contract with handler module outside any allowed namespace returns REJECTED.

    The handler_class points to os.system — this will either fail to import
    as a proper handler class or fail to wire, returning REJECTED.
    """
    source = _make_source()
    engine = _make_engine()
    bus = _make_bus()

    source.on_contract_registered(
        node_name="node_evil",
        contract_yaml=_EVIL_MODULE_CONTRACT_YAML,
        correlation_id=uuid4(),
    )

    result = await source.materialize_cached_contract(
        node_name="node_evil",
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert result.status == EnumMaterializationStatus.REJECTED
    assert result.reason == EnumMaterializationRejection.HANDLER_ALLOWLIST
