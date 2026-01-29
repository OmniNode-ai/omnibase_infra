# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Ledger Write Effect - Capability-oriented event ledger write node.

This effect node provides event ledger write capabilities using PostgreSQL.
Named by capability ("ledger.write"), not by vendor (e.g., PostgreSQL).

Core Principle:
    "I'm interested in what you do, not what you are"

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom storage logic - all behavior from handler
    - Lightweight shell that delegates to handler implementation
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for external I/O operations.
All storage logic is 100% driven by handler implementations, not Python code.

Capabilities:
    - ledger.write: Append events to the audit ledger
    - ledger.query: Query events from the audit ledger

Event Topics:
    Consumed:
        - {env}.{namespace}.onex.cmd.ledger-append.v1
        - {env}.{namespace}.onex.cmd.ledger-query.v1
    Published:
        - {env}.{namespace}.onex.evt.ledger-appended.v1
        - {env}.{namespace}.onex.evt.ledger-query-result.v1

Design Decisions:
    - 100% Contract-Driven: All capabilities in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Handler wired externally
    - Capability-Oriented: Named by what it does, not what it uses

Related:
    - contract.yaml: Capability definitions and IO operations
    - models/: Input, output models (ModelPayloadLedgerAppend, ModelLedgerAppendResult)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLedgerWriteEffect(NodeEffect):
    """Effect node for event ledger write operations.

    Capability: ledger.write

    Provides a capability-oriented interface for event ledger operations.
    Uses PostgreSQL for storing events in an append-only audit ledger with
    idempotent write support via (topic, partition, kafka_offset) constraint.

    This node is declarative - all behavior is defined in contract.yaml and
    implemented through the handler. No custom storage logic exists in this class.

    The audit ledger serves as the system's source of truth for all events.
    Events are never dropped - even malformed events are captured with raw
    data intact for later analysis or re-processing.

    Attributes:
        container: ONEX dependency injection container

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> node = NodeLedgerWriteEffect(container)
        >>> # Handler must be wired externally via registry
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the ledger write effect node.

        Args:
            container: ONEX dependency injection container for resolving
                dependencies defined in contract.yaml.
        """
        super().__init__(container)


__all__ = ["NodeLedgerWriteEffect"]
