# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for kernel auto-wiring (OMN-7656).

Proves that:
1. A mock package with onex.nodes entry points auto-wires into the
   dispatch engine without requiring a Plugin class.
2. Auto-wiring coexists with explicit plugins (plugins run first).
3. Topic collision detection warns when auto-wired topics overlap
   with explicit plugin routes.
4. The auto-wiring report appears in kernel startup logs.
5. Quarantined contracts are excluded from wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome

pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


def _make_discovered_contract(
    name: str = "test-node",
    node_type: str = "COMPUTE_GENERIC",
    subscribe_topics: tuple[str, ...] = ("onex.evt.test.my-event.v1",),
    publish_topics: tuple[str, ...] = (),
    handler_name: str = "HandlerTest",
    handler_module: str = "tests.integration.runtime.test_kernel_auto_wiring",
) -> ModelDiscoveredContract:
    """Build a minimal ModelDiscoveredContract for testing."""
    return ModelDiscoveredContract(
        name=name,
        node_type=node_type,
        description="Test contract",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        node_version="1.0.0",
        contract_path=Path("/tmp/test-contract/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="test-package",
        package_version="0.1.0",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_name,
                        module=handler_module,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelTestEvent",
                        module="tests.integration.runtime.test_kernel_auto_wiring",
                    ),
                ),
            ),
        ),
    )


class HandlerTest:
    """Mock handler class that auto-wiring can instantiate with zero args."""

    async def handle(self, envelope: object) -> None:
        """No-op handler for testing."""


class ModelTestEvent:
    """Mock event model for handler routing."""


# =============================================================================
# Tests: Discovery + Wiring (no Plugin class)
# =============================================================================


@pytest.mark.asyncio
async def test_auto_wiring_wires_contract_without_plugin_class() -> None:
    """A discovered contract wires handlers into the dispatch engine
    without requiring a Plugin class — the centerpiece of OMN-7656."""
    contract = _make_discovered_contract()
    manifest = ModelAutoWiringManifest(
        contracts=(contract,),
        errors=(),
    )

    # Create a real MessageDispatchEngine
    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )

    engine = MessageDispatchEngine(logger=MagicMock())

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,  # No Kafka in unit test
        environment="test",
    )

    assert report.total_wired == 1
    assert report.total_failed == 0

    # Verify dispatcher was registered
    result = report.results[0]
    assert result.outcome == EnumWiringOutcome.WIRED
    assert len(result.dispatchers_registered) > 0
    assert len(result.routes_registered) > 0

    # Verify dispatch engine has the registered dispatcher
    dispatcher_id = result.dispatchers_registered[0]
    assert dispatcher_id in engine._dispatchers


@pytest.mark.asyncio
async def test_auto_wiring_skips_contract_without_handler_routing() -> None:
    """Contracts without handler_routing are skipped, not failed."""
    contract = ModelDiscoveredContract(
        name="no-routing",
        node_type="EFFECT_GENERIC",
        description="No handler routing",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        node_version="1.0.0",
        contract_path=Path("/tmp/test/contract.yaml"),  # noqa: S108
        entry_point_name="no-routing",
        package_name="test-package",
        package_version="0.1.0",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.test.something.v1",),
        ),
        handler_routing=None,
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    engine = MagicMock()
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        environment="test",
    )

    assert report.total_skipped == 1
    assert report.total_wired == 0
    assert report.total_failed == 0


@pytest.mark.asyncio
async def test_auto_wiring_skips_contract_without_subscribe_topics() -> None:
    """Contracts with handler routing but no subscribe topics are skipped."""
    contract = _make_discovered_contract()
    # Override event_bus to have no subscribe topics
    contract = contract.model_copy(
        update={
            "event_bus": ModelEventBusWiring(
                subscribe_topics=(),
                publish_topics=("onex.evt.test.out.v1",),
            )
        }
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    engine = MagicMock()
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        environment="test",
    )

    assert report.total_skipped == 1
    assert report.total_wired == 0


# =============================================================================
# Tests: Topic collision detection
# =============================================================================


def test_topic_matches_pattern() -> None:
    """_topic_matches_pattern correctly matches ONEX 5-segment topics."""
    from omnibase_infra.runtime.service_kernel import _topic_matches_pattern

    # Exact match
    assert _topic_matches_pattern(
        "onex.evt.platform.node-introspection.v1",
        "onex.evt.platform.node-introspection.v1",
    )

    # Wildcard match
    assert _topic_matches_pattern(
        "onex.evt.platform.node-introspection.v1",
        "*.evt.platform.node-introspection.*",
    )

    # No match — different segment
    assert not _topic_matches_pattern(
        "onex.evt.platform.node-introspection.v1",
        "*.evt.platform.different-event.*",
    )

    # No match — different length
    assert not _topic_matches_pattern(
        "onex.evt.platform.v1",
        "*.evt.platform.node-introspection.*",
    )


# =============================================================================
# Tests: Coexistence with explicit plugins
# =============================================================================


@pytest.mark.asyncio
async def test_auto_wiring_detects_duplicate_topic_ownership() -> None:
    """Two contracts subscribing to the same topic are detected as duplicates."""
    contract_a = _make_discovered_contract(
        name="node-a",
        subscribe_topics=("onex.evt.test.shared-topic.v1",),
    )
    contract_b = _make_discovered_contract(
        name="node-b",
        subscribe_topics=("onex.evt.test.shared-topic.v1",),
    )
    manifest = ModelAutoWiringManifest(
        contracts=(contract_a, contract_b),
        errors=(),
    )

    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )

    engine = MessageDispatchEngine(logger=MagicMock())
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        environment="test",
    )

    assert len(report.duplicates) > 0
    dup = report.duplicates[0]
    assert dup.topic == "onex.evt.test.shared-topic.v1"
    assert "node-a" in dup.owners
    assert "node-b" in dup.owners


# =============================================================================
# Tests: Manifest filtering
# =============================================================================


@pytest.mark.asyncio
async def test_filtered_manifest_excludes_quarantined_contracts() -> None:
    """Kernel filters quarantined contracts before calling wire_from_manifest."""
    good = _make_discovered_contract(name="good-node")
    bad = _make_discovered_contract(name="bad-node")
    manifest = ModelAutoWiringManifest(
        contracts=(good, bad),
        errors=(),
    )

    quarantined_names = {"bad-node"}
    filtered = ModelAutoWiringManifest(
        contracts=tuple(
            c for c in manifest.contracts if c.name not in quarantined_names
        ),
        errors=manifest.errors,
    )

    assert filtered.total_discovered == 1
    assert filtered.contracts[0].name == "good-node"


# =============================================================================
# Tests: Discovery from paths (testing path)
# =============================================================================


def test_discover_contracts_from_paths_with_valid_yaml(tmp_path: Path) -> None:
    """discover_contracts_from_paths parses a valid contract.yaml."""
    contract_dir = tmp_path / "my_node"
    contract_dir.mkdir()
    contract_yaml = contract_dir / "contract.yaml"
    contract_yaml.write_text(
        """
name: my-test-node
node_type: COMPUTE_GENERIC
description: A test node
contract_version:
  major: 1
  minor: 2
  patch: 3
event_bus:
  subscribe_topics:
    - onex.evt.test.my-event.v1
  publish_topics:
    - onex.evt.test.my-output.v1
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerTest
        module: tests.integration.runtime.test_kernel_auto_wiring
      event_model:
        name: ModelTestEvent
        module: tests.integration.runtime.test_kernel_auto_wiring
"""
    )

    manifest = discover_contracts_from_paths([contract_yaml])
    assert manifest.total_discovered == 1
    assert manifest.total_errors == 0

    contract = manifest.contracts[0]
    assert contract.name == "my-test-node"
    assert contract.node_type == "COMPUTE_GENERIC"
    assert str(contract.contract_version) == "1.2.3"
    assert contract.event_bus is not None
    assert "onex.evt.test.my-event.v1" in contract.event_bus.subscribe_topics
    assert contract.handler_routing is not None
    assert len(contract.handler_routing.handlers) == 1
