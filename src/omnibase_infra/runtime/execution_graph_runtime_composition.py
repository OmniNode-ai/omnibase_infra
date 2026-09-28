# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot-time composition for the opt-in execution graph read effect."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_infra.gateway.utils.util_key_loader import load_private_key_from_pem
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.execution_graph_read_fold import (
    ExecutionGraphReadFold,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    PostgresExecutionGraphStoredChainReader,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandExecutor,
)
from omnibase_infra.runtime.execution_graph_read_composition import (
    compose_execution_graph_read_executor,
    open_execution_graph_read_pools,
)
from omnibase_infra.runtime.execution_graph_terminal_publisher import (
    ExecutionGraphTerminalPublisher,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.models.model_execution_graph_read_runtime_config import (
    ModelExecutionGraphReadRuntimeConfig,
)
from omnibase_infra.runtime.models.model_execution_graph_trusted_gateway_config import (
    ModelExecutionGraphTrustedGatewayConfig,
)

GRAPH_READ_CONTRACT_NAME = "node_execution_graph_read_effect"
GRAPH_READ_WORKFLOW_TYPE = "delegation-execution-graph-read"


def select_execution_graph_contract(
    manifest: ModelAutoWiringManifest, *, enabled: bool
) -> tuple[ModelAutoWiringManifest, ModelDiscoveredContract | None]:
    """Exclude a disabled contract before lifecycle, routes, or subscriptions."""
    graph_contracts = tuple(
        contract
        for contract in manifest.contracts
        if contract.name == GRAPH_READ_CONTRACT_NAME
    )
    if enabled:
        if len(graph_contracts) != 1:
            raise ValueError(
                "execution_graph_read is enabled but its node contract "
                "is not uniquely discovered on this runtime profile"
            )
        return manifest, graph_contracts[0]
    return (
        manifest.model_copy(
            update={
                "contracts": tuple(
                    contract
                    for contract in manifest.contracts
                    if contract.name != GRAPH_READ_CONTRACT_NAME
                )
            }
        ),
        None,
    )


@asynccontextmanager
async def open_execution_graph_runtime_executor(
    *,
    config: ModelExecutionGraphReadRuntimeConfig,
    gateway: ModelExecutionGraphTrustedGatewayConfig,
    contract: ModelDiscoveredContract,
    event_bus: ProtocolEventBusLike,
) -> AsyncIterator[ExecutionGraphReadCommandExecutor]:
    """Resolve the pinned topology, signer, and two independently owned pools."""
    if contract.name != GRAPH_READ_CONTRACT_NAME or contract.event_bus is None:
        raise ValueError("graph read runtime requires its discovered node contract")
    if contract.event_bus.subscribe_topics != (gateway.command_topic,):
        raise ValueError("graph gateway command differs from the node contract")
    terminal_config = config.terminal_publisher
    if contract.event_bus.publish_topics != (terminal_config.terminal_topic,):
        raise ValueError("graph terminal topic differs from the node contract")
    if terminal_config.workflow_type != GRAPH_READ_WORKFLOW_TYPE:
        raise ValueError("graph workflow type differs from the declared workflow")

    topology = PackagedExecutionGraphTopologyContract().resolve(config.topology_version)
    private_key = load_private_key_from_pem(config.private_key_path)

    async def publish_terminal(
        topic: str, envelope: ModelMessageEnvelope[dict[str, object]]
    ) -> None:
        await event_bus.publish_envelope(envelope=envelope, topic=topic)

    publisher = ExecutionGraphTerminalPublisher(
        config=terminal_config,
        private_key=private_key,
        publish=publish_terminal,
    )
    async with open_execution_graph_read_pools(config.databases) as (
        analytics_pool,
        ledger_pool,
    ):
        yield compose_execution_graph_read_executor(
            analytics_pool=analytics_pool,
            ledger_pool=ledger_pool,
            topology=topology,
            workflow_type=terminal_config.workflow_type,
            fold=ExecutionGraphReadFold(
                workflow_type=terminal_config.workflow_type,
                read_clock=lambda: datetime.now(UTC),
            ),
            stored_chain_reader_factory=PostgresExecutionGraphStoredChainReader,
            publish_terminal=publisher.publish,
        )


__all__ = [
    "GRAPH_READ_CONTRACT_NAME",
    "GRAPH_READ_WORKFLOW_TYPE",
    "open_execution_graph_runtime_executor",
    "select_execution_graph_contract",
]
