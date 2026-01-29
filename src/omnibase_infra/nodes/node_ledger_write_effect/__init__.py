# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Ledger Write Effect - Capability-oriented event ledger write node.

This package provides the NodeLedgerWriteEffect, a capability-oriented
effect node for event ledger write operations using PostgreSQL.

Core Principle:
    "I'm interested in what you do, not what you are"

    Named by capability (ledger.write), not by specific backend implementation.
    Uses PostgreSQL for append-only event ledger storage with idempotent writes.

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

Available Exports:
    - NodeLedgerWriteEffect: The declarative effect node
    - ModelLedgerEntry: Single ledger entry representing one event
    - ModelLedgerAppendResult: Result of a ledger write operation
    - ModelLedgerQuery: Query parameters for ledger searches
    - ModelLedgerQueryResult: Result of a ledger query operation

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_ledger_write_effect import (
    ...     NodeLedgerWriteEffect,
    ...     ModelLedgerEntry,
    ...     ModelLedgerAppendResult,
    ... )
    >>>
    >>> container = ModelONEXContainer()
    >>> node = NodeLedgerWriteEffect(container)

Related Modules:
    - models: Pydantic models for ledger operations
    - registry: Dependency injection registration
"""

from omnibase_infra.nodes.node_ledger_write_effect.handlers import (
    HandlerLedgerAppend,
    HandlerLedgerQuery,
)
from omnibase_infra.nodes.node_ledger_write_effect.models import (
    ModelLedgerAppendResult,
    ModelLedgerEntry,
    ModelLedgerQuery,
    ModelLedgerQueryResult,
)
from omnibase_infra.nodes.node_ledger_write_effect.node import NodeLedgerWriteEffect
from omnibase_infra.nodes.node_ledger_write_effect.protocols import (
    ProtocolLedgerPersistence,
)
from omnibase_infra.nodes.node_ledger_write_effect.registry import (
    RegistryInfraLedgerWrite,
)

__all__ = [
    # Node
    "NodeLedgerWriteEffect",
    # Handlers
    "HandlerLedgerAppend",
    "HandlerLedgerQuery",
    # Protocol
    "ProtocolLedgerPersistence",
    # Registry
    "RegistryInfraLedgerWrite",
    # Models
    "ModelLedgerAppendResult",
    "ModelLedgerEntry",
    "ModelLedgerQuery",
    "ModelLedgerQueryResult",
]
