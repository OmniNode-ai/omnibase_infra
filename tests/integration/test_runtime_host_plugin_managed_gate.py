# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests: RuntimeHostProcess._wire_event_bus_subscriptions skips plugin_managed.

OMN-10864 second fix: the RuntimeHostProcess subscription wiring path must not create
Kafka consumer groups for contracts flagged plugin_managed=true.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _write_contract(path: Path, name: str, *, plugin_managed: bool = False) -> None:
    pm_line = f"  plugin_managed: {'true' if plugin_managed else 'false'}\n"
    path.write_text(
        textwrap.dedent(f"""\
        name: "{name}"
        node_type: "ORCHESTRATOR_GENERIC"
        event_bus:
          version:
            major: 1
            minor: 0
            patch: 0
          subscribe_topics:
            - "onex.cmd.omnibase-infra.{name}-request.v1"
        """)
        + pm_line
    )


pytestmark = pytest.mark.integration


class TestRuntimeHostProcessPluginManagedGate:
    """_wire_event_bus_subscriptions must honour plugin_managed=true (OMN-10864 v2)."""

    def _make_descriptor(self, contract_path: Path, name: str) -> Any:
        d = MagicMock()
        d.contract_path = str(contract_path)
        d.name = name
        return d

    @pytest.mark.asyncio
    async def test_plugin_managed_handler_skipped(self, tmp_path: Path) -> None:
        """Handler whose contract.yaml has plugin_managed=true is not subscribed."""
        contract_path = tmp_path / "contract.yaml"
        _write_contract(
            contract_path, "node_delegation_orchestrator", plugin_managed=True
        )

        mock_event_bus = MagicMock()
        mock_event_bus.subscribe = AsyncMock(return_value=MagicMock())
        mock_dispatch = MagicMock()

        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._event_bus = mock_event_bus
        process._dispatch_engine = mock_dispatch
        process._handler_descriptors = {
            "TestHandler": self._make_descriptor(
                contract_path, "node_delegation_orchestrator"
            )
        }
        process._runtime_node_graph_config = None
        process._node_identity = MagicMock()
        process._node_identity.service = "test"
        process._node_identity.version = "0.0.1"

        mock_wiring = MagicMock()
        mock_wiring.wire_subscriptions = AsyncMock()

        with (
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.EventBusSubcontractWiring",
                return_value=mock_wiring,
            ),
            patch.object(
                type(process),
                "_get_environment_from_config",
                return_value="test",
            ),
        ):
            await process._wire_event_bus_subscriptions()

        mock_wiring.wire_subscriptions.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_plugin_managed_handler_wired(self, tmp_path: Path) -> None:
        """Handler with plugin_managed=false is subscribed normally."""
        contract_path = tmp_path / "contract.yaml"
        _write_contract(contract_path, "node_normal_worker", plugin_managed=False)

        mock_event_bus = MagicMock()
        mock_dispatch = MagicMock()

        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._event_bus = mock_event_bus
        process._dispatch_engine = mock_dispatch
        process._handler_descriptors = {
            "NormalHandler": self._make_descriptor(contract_path, "node_normal_worker")
        }
        process._runtime_node_graph_config = None
        process._node_identity = MagicMock()
        process._node_identity.service = "test"
        process._node_identity.version = "0.0.1"

        mock_wiring = MagicMock()
        mock_wiring.wire_subscriptions = AsyncMock()

        with (
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.EventBusSubcontractWiring",
                return_value=mock_wiring,
            ),
            patch.object(
                type(process),
                "_get_environment_from_config",
                return_value="test",
            ),
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.load_event_bus_subcontract",
            ) as mock_load,
        ):
            mock_subcontract = MagicMock()
            mock_subcontract.subscribe_topics = (
                "onex.cmd.omnibase-infra.node-normal-worker-request.v1",
            )
            mock_load.return_value = mock_subcontract

            await process._wire_event_bus_subscriptions()

        mock_wiring.wire_subscriptions.assert_called_once()
