# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Validation Ledger Write Effect - validation ledger write node (OMN-14524).

This effect node provides validation event ledger write capabilities using
PostgreSQL, mirroring NodeLedgerWriteEffect's pattern for the generic
event_ledger.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom storage logic - all behavior from handler
    - Lightweight shell that delegates to handler implementation

Extends NodeEffect from omnibase_core for external I/O operations.
All storage logic is 100% driven by handler implementations, not Python code.

Capabilities:
    - validation_ledger.write: Append validation events to the validation
      ledger

Related:
    - contract.yaml: Capability definitions and IO operations
    - handlers/handler_validation_ledger_append.py: All write logic
    - OMN-14524: validation_event_ledger had 0 rows -- no constructable
      write-effect existed until this node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeValidationLedgerWriteEffect(NodeEffect):
    """Effect node for validation event ledger write operations.

    Capability: validation_ledger.write

    Uses PostgreSQL for storing cross-repo validation events in an
    append-only ledger with idempotent write support via (kafka_topic,
    kafka_partition, kafka_offset) constraint.

    This node is declarative - all behavior is defined in contract.yaml and
    implemented through the handler. No custom storage logic exists in this
    class.

    Attributes:
        container: ONEX dependency injection container

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> node = NodeValidationLedgerWriteEffect(container)
        >>> # Handler must be wired externally via registry
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the validation ledger write effect node.

        Args:
            container: ONEX dependency injection container for resolving
                dependencies defined in contract.yaml.
        """
        super().__init__(container)


__all__ = ["NodeValidationLedgerWriteEffect"]
