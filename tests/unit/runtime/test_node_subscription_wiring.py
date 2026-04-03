# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for package-node subscription wiring [OMN-7410].

Validates that _wire_package_node_subscriptions() correctly discovers
node contracts from the installed package tree and wires Kafka
subscriptions for nodes that have event_bus.subscribe_topics declared.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


@pytest.fixture()
def tmp_nodes_dir(tmp_path: Path) -> Path:
    """Create temporary node contract directory structure."""
    # Node A: has subscribe_topics (should be wired)
    node_a = tmp_path / "nodes" / "node_alpha" / "contract.yaml"
    node_a.parent.mkdir(parents=True)
    node_a.write_text(
        textwrap.dedent("""\
        name: "node_alpha"
        node_type: "EFFECT_GENERIC"
        event_bus:
          version:
            major: 1
            minor: 0
            patch: 0
          subscribe_topics:
            - "onex.cmd.omnibase-infra.alpha-request.v1"
          publish_topics:
            - "onex.evt.omnibase-infra.alpha-completed.v1"
        """)
    )

    # Node B: has subscribe_topics (should be wired)
    node_b = tmp_path / "nodes" / "node_beta" / "contract.yaml"
    node_b.parent.mkdir(parents=True)
    node_b.write_text(
        textwrap.dedent("""\
        name: "node_beta"
        node_type: "COMPUTE"
        event_bus:
          version:
            major: 1
            minor: 0
            patch: 0
          subscribe_topics:
            - "onex.cmd.omnibase-infra.beta-request.v1"
        """)
    )

    # Node C: no subscribe_topics (should be skipped)
    node_c = tmp_path / "nodes" / "node_gamma" / "contract.yaml"
    node_c.parent.mkdir(parents=True)
    node_c.write_text(
        textwrap.dedent("""\
        name: "node_gamma"
        node_type: "EFFECT_GENERIC"
        event_bus:
          version:
            major: 1
            minor: 0
            patch: 0
          publish_topics:
            - "onex.evt.omnibase-infra.gamma-completed.v1"
        """)
    )

    # Node D: no event_bus at all (should be skipped)
    node_d = tmp_path / "nodes" / "node_delta" / "contract.yaml"
    node_d.parent.mkdir(parents=True)
    node_d.write_text(
        textwrap.dedent("""\
        name: "node_delta"
        node_type: "COMPUTE"
        description: "Node with no event bus section"
        """)
    )

    return tmp_path


@pytest.fixture()
def mock_event_bus_wiring() -> MagicMock:
    """Mock EventBusSubcontractWiring with tracking."""
    wiring = MagicMock()
    wiring.wire_subscriptions = AsyncMock()
    wiring._wire_calls: list[dict[str, Any]] = []

    original_wire = wiring.wire_subscriptions

    async def tracking_wire(subcontract: Any, node_name: str) -> None:
        wiring._wire_calls.append(
            {
                "node_name": node_name,
                "subscribe_topics": list(subcontract.subscribe_topics),
            }
        )

    wiring.wire_subscriptions = AsyncMock(side_effect=tracking_wire)
    return wiring


class TestPackageNodeSubscriptionWiring:
    """Tests for _wire_package_node_subscriptions."""

    @pytest.mark.asyncio
    async def test_package_nodes_get_subscriptions_wired(
        self, tmp_nodes_dir: Path, mock_event_bus_wiring: MagicMock
    ) -> None:
        """Nodes with subscribe_topics get wire_subscriptions() called."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            _discover_package_node_contracts,
        )

        contracts = _discover_package_node_contracts(tmp_nodes_dir)
        eligible = [
            c
            for c in contracts
            if c.get("event_bus") and c["event_bus"].get("subscribe_topics")
        ]

        from omnibase_infra.runtime.service_runtime_host_process import (
            _wire_package_node_subscriptions,
        )

        wired, skipped_existing, skipped_no_topics = (
            await _wire_package_node_subscriptions(
                contracts=contracts,
                event_bus_wiring=mock_event_bus_wiring,
                already_wired_names=set(),
            )
        )

        assert wired == len(eligible)
        assert mock_event_bus_wiring.wire_subscriptions.call_count == len(eligible)

        wired_names = {c["node_name"] for c in mock_event_bus_wiring._wire_calls}
        assert "node_alpha" in wired_names
        assert "node_beta" in wired_names

    @pytest.mark.asyncio
    async def test_plugin_prewired_nodes_skipped(
        self, tmp_nodes_dir: Path, mock_event_bus_wiring: MagicMock
    ) -> None:
        """Nodes already wired by domain plugins are not double-wired."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            _discover_package_node_contracts,
            _wire_package_node_subscriptions,
        )

        contracts = _discover_package_node_contracts(tmp_nodes_dir)

        wired, skipped_existing, skipped_no_topics = (
            await _wire_package_node_subscriptions(
                contracts=contracts,
                event_bus_wiring=mock_event_bus_wiring,
                already_wired_names={"node_alpha"},
            )
        )

        assert wired == 1  # only node_beta
        assert skipped_existing == 1  # node_alpha skipped
        wired_names = {c["node_name"] for c in mock_event_bus_wiring._wire_calls}
        assert "node_alpha" not in wired_names
        assert "node_beta" in wired_names

    @pytest.mark.asyncio
    async def test_nodes_without_topics_not_wired(
        self, tmp_nodes_dir: Path, mock_event_bus_wiring: MagicMock
    ) -> None:
        """Nodes without subscribe_topics are silently skipped."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            _discover_package_node_contracts,
            _wire_package_node_subscriptions,
        )

        contracts = _discover_package_node_contracts(tmp_nodes_dir)

        wired, skipped_existing, skipped_no_topics = (
            await _wire_package_node_subscriptions(
                contracts=contracts,
                event_bus_wiring=mock_event_bus_wiring,
                already_wired_names=set(),
            )
        )

        wired_names = {c["node_name"] for c in mock_event_bus_wiring._wire_calls}
        assert "node_gamma" not in wired_names
        assert "node_delta" not in wired_names
        assert skipped_no_topics == 2

    @pytest.mark.asyncio
    async def test_subscription_wiring_is_idempotent(
        self, tmp_nodes_dir: Path, mock_event_bus_wiring: MagicMock
    ) -> None:
        """Calling wiring twice with same already_wired produces same count."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            _discover_package_node_contracts,
            _wire_package_node_subscriptions,
        )

        contracts = _discover_package_node_contracts(tmp_nodes_dir)

        wired1, _, _ = await _wire_package_node_subscriptions(
            contracts=contracts,
            event_bus_wiring=mock_event_bus_wiring,
            already_wired_names=set(),
        )

        # Second call: the names wired in first call are now "already wired"
        first_wired_names = {c["node_name"] for c in mock_event_bus_wiring._wire_calls}

        mock_event_bus_wiring._wire_calls.clear()
        mock_event_bus_wiring.wire_subscriptions.reset_mock()

        wired2, _, _ = await _wire_package_node_subscriptions(
            contracts=contracts,
            event_bus_wiring=mock_event_bus_wiring,
            already_wired_names=first_wired_names,
        )

        assert wired2 == 0  # all already wired


class TestDiscoverPackageNodeContracts:
    """Tests for _discover_package_node_contracts."""

    def test_discovers_contracts_from_nodes_dir(self, tmp_nodes_dir: Path) -> None:
        from omnibase_infra.runtime.service_runtime_host_process import (
            _discover_package_node_contracts,
        )

        contracts = _discover_package_node_contracts(tmp_nodes_dir)
        names = {c["name"] for c in contracts}
        assert "node_alpha" in names
        assert "node_beta" in names
        assert "node_gamma" in names
        assert "node_delta" in names

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        from omnibase_infra.runtime.service_runtime_host_process import (
            _discover_package_node_contracts,
        )

        contracts = _discover_package_node_contracts(tmp_path / "nonexistent")
        assert contracts == []
