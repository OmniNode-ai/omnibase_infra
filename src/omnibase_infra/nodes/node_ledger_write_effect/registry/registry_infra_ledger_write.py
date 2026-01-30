# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for event ledger write effect node components.

This registry exports all public symbols for the node_ledger_write_effect node,
providing a single import point for consumers of this node.

Exported Components:
    - NodeLedgerWriteEffect: The declarative effect node
    - HandlerLedgerAppend: Handler for idempotent append operations
    - HandlerLedgerQuery: Handler for query operations
    - ProtocolLedgerPersistence: Protocol for ledger persistence
    - Models: Data models for ledger operations

Example:
    >>> from omnibase_infra.nodes.node_ledger_write_effect.registry import (
    ...     RegistryInfraLedgerWrite,
    ... )
    >>> # Access all components via registry
    >>> node_cls = RegistryInfraLedgerWrite.node
    >>> append_handler = RegistryInfraLedgerWrite.handler_append
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Handlers
from omnibase_infra.nodes.node_ledger_write_effect.handlers import (
    HandlerLedgerAppend,
    HandlerLedgerQuery,
)

# Models
from omnibase_infra.nodes.node_ledger_write_effect.models import (
    ModelLedgerAppendResult,
    ModelLedgerEntry,
    ModelLedgerQuery,
    ModelLedgerQueryResult,
)

# Node
from omnibase_infra.nodes.node_ledger_write_effect.node import NodeLedgerWriteEffect

# Protocol
from omnibase_infra.nodes.node_ledger_write_effect.protocols import (
    ProtocolLedgerPersistence,
)

# Intent payload model (from reducers)
from omnibase_infra.nodes.reducers.models import ModelPayloadLedgerAppend


class RegistryInfraLedgerWrite:
    """Registry providing access to all ledger write effect node components.

    This class provides a centralized access point for all components of the
    node_ledger_write_effect node. Use this registry for dependency injection
    and container registration.

    Class Attributes:
        node: The NodeLedgerWriteEffect class
        handler_append: The HandlerLedgerAppend class
        handler_query: The HandlerLedgerQuery class
        protocol: The ProtocolLedgerPersistence protocol
        models: Tuple of all model classes

    Example:
        >>> from omnibase_infra.nodes.node_ledger_write_effect.registry import (
        ...     RegistryInfraLedgerWrite,
        ... )
        >>> # Create node instance
        >>> node = RegistryInfraLedgerWrite.node(container)
        >>> # Create handlers
        >>> append_handler = RegistryInfraLedgerWrite.handler_append(container, db_handler)
    """

    # Node
    node = NodeLedgerWriteEffect

    # Handlers
    handler_append = HandlerLedgerAppend
    handler_query = HandlerLedgerQuery

    # Protocol
    protocol = ProtocolLedgerPersistence

    # Models (as tuple for iteration)
    models = (
        ModelLedgerAppendResult,
        ModelLedgerEntry,
        ModelLedgerQuery,
        ModelLedgerQueryResult,
        ModelPayloadLedgerAppend,
    )

    # Individual model references
    model_append_result = ModelLedgerAppendResult
    model_entry = ModelLedgerEntry
    model_query = ModelLedgerQuery
    model_query_result = ModelLedgerQueryResult
    model_payload_append = ModelPayloadLedgerAppend


# Re-export all components at module level for convenience
__all__ = [
    "RegistryInfraLedgerWrite",
    # Node
    "NodeLedgerWriteEffect",
    # Handlers
    "HandlerLedgerAppend",
    "HandlerLedgerQuery",
    # Protocol
    "ProtocolLedgerPersistence",
    # Models
    "ModelLedgerAppendResult",
    "ModelLedgerEntry",
    "ModelLedgerQuery",
    "ModelLedgerQueryResult",
    "ModelPayloadLedgerAppend",
]
