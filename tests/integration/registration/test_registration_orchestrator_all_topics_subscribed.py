# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: node_registration_orchestrator subscribes all declared topics.

Regression test for OMN-9413: contract.yaml declares 7 subscribe_topics but
only 3 consumer groups were wired at runtime.

Root cause: the _prepare_handler_wiring special-case for node_registration_orchestrator
skips all handler dispatchers (RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP) so the generic
contract auto-wiring path owns topic subscription. This test verifies that all 7
declared subscribe_topics receive Kafka consumer subscriptions via
subscribe_wired_contract_topics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
    / "contract.yaml"
)

_EXPECTED_SUBSCRIBE_TOPICS = (
    "onex.evt.platform.node-introspection.v1",
    "onex.evt.platform.registry-request-introspection.v1",
    "onex.intent.platform.runtime-tick.v1",
    "onex.cmd.platform.node-registration-acked.v1",
    "onex.evt.platform.node-heartbeat.v1",
    "onex.cmd.platform.topic-catalog-query.v1",
    "onex.cmd.platform.request-introspection.v1",
)


async def test_registration_orchestrator_contract_has_7_subscribe_topics() -> None:
    """Contract YAML declares exactly 7 subscribe_topics."""
    assert _CONTRACT_PATH.exists(), f"Contract not found: {_CONTRACT_PATH}"
    manifest = discover_contracts_from_paths([_CONTRACT_PATH])
    assert manifest.total_discovered == 1
    contract = manifest.contracts[0]
    assert contract.event_bus is not None
    assert set(contract.event_bus.subscribe_topics) == set(
        _EXPECTED_SUBSCRIBE_TOPICS
    ), (
        f"Contract subscribe_topics mismatch.\n"
        f"Expected: {sorted(_EXPECTED_SUBSCRIBE_TOPICS)}\n"
        f"Got:      {sorted(contract.event_bus.subscribe_topics)}"
    )


async def test_registration_orchestrator_all_declared_topics_get_subscribed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All 7 contract subscribe_topics must receive a Kafka subscription.

    This is the regression test for OMN-9413. wire_from_manifest skips
    handler dispatchers for the orchestrator (all handlers are
    RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP) but subscribe_wired_contract_topics
    must still subscribe every declared topic.
    """
    assert _CONTRACT_PATH.exists(), f"Contract not found: {_CONTRACT_PATH}"
    manifest = discover_contracts_from_paths([_CONTRACT_PATH])
    assert manifest.total_discovered == 1, (
        f"Expected 1 contract, got {manifest.total_discovered}. "
        f"Errors: {manifest.errors}"
    )

    event_bus = EventBusInmemory(environment="test", group="test")
    await event_bus.start()

    dispatch_engine = MagicMock()
    dispatch_engine.is_frozen = True
    dispatch_engine._routes = {}

    monkeypatch.setenv("RUNTIME_PROFILE", "main")

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=dispatch_engine,
        event_bus=event_bus,
        environment="test",
        subscribe_immediately=False,
    )

    assert report.total_failed == 0, (
        f"wire_from_manifest reported failures: "
        f"{[r for r in report.results if r.outcome.value == 'failed']}"
    )

    await subscribe_wired_contract_topics(
        manifest=manifest,
        report=report,
        dispatch_engine=dispatch_engine,
        event_bus=event_bus,
        environment="test",
    )

    subscribed_topics = sorted(event_bus._subscribers.keys())
    missing = sorted(set(_EXPECTED_SUBSCRIBE_TOPICS) - set(subscribed_topics))
    extra = sorted(set(subscribed_topics) - set(_EXPECTED_SUBSCRIBE_TOPICS))

    assert not missing, (
        f"OMN-9413 regression: {len(missing)} topic(s) were NOT subscribed.\n"
        f"Missing: {missing}\n"
        f"Subscribed: {subscribed_topics}"
    )
    assert len(subscribed_topics) == len(_EXPECTED_SUBSCRIBE_TOPICS), (
        f"Expected {len(_EXPECTED_SUBSCRIBE_TOPICS)} subscriptions, "
        f"got {len(subscribed_topics)}. Extra: {extra}"
    )

    await event_bus.shutdown()
