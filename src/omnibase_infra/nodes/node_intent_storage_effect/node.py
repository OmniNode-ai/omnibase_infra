# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Intent Storage Effect - Capability-oriented intent storage node.

This effect node provides intent storage capabilities using Memgraph graph database.
Named by capability ("intent.storage"), not by vendor (e.g., Memgraph).

Core Principle:
    "I'm interested in what you do, not what you are"

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom storage logic - all behavior from handler (HandlerIntent)
    - Lightweight shell that delegates to HandlerIntent -> HandlerGraph
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for external I/O operations.
All storage logic is 100% driven by handler implementations, not Python code.

Capabilities:
    - intent.storage: Store, query, and analyze intents
    - intent.storage.store: Store classified intents as graph nodes
    - intent.storage.query_session: Query intents by session identifier
    - intent.storage.query_distribution: Get intent distribution statistics

Event Topics:
    Consumed:
        - {env}.{namespace}.onex.evt.intent-classified.v1
        - {env}.{namespace}.onex.cmd.intent-query-session.v1
        - {env}.{namespace}.onex.cmd.intent-query-distribution.v1
    Published:
        - {env}.{namespace}.onex.evt.intent-stored.v1
        - {env}.{namespace}.onex.evt.intent-session-query-result.v1
        - {env}.{namespace}.onex.evt.intent-distribution-result.v1

Handler Stack:
    NodeIntentStorageEffect -> HandlerIntent -> HandlerGraph -> Memgraph

Design Decisions:
    - 100% Contract-Driven: All capabilities in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Handler wired externally
    - Capability-Oriented: Named by what it does, not what it uses

Related:
    - contract.yaml: Capability definitions and IO operations
    - HandlerIntent: Intent-specific graph operations handler
    - HandlerGraph: Underlying Memgraph graph handler
    - models/: Input, output models
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeIntentStorageEffect(NodeEffect):
    """Effect node for intent storage operations.

    Capability: intent.storage

    Provides a capability-oriented interface for intent storage operations.
    Uses Memgraph graph database for storing intents as nodes with properties.

    This node is declarative - all behavior is defined in contract.yaml and
    implemented through the handler (HandlerIntent). No custom storage logic
    exists in this class.

    Attributes:
        container: ONEX dependency injection container

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> node = NodeIntentStorageEffect(container)
        >>> # Handler must be wired externally via registry
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the intent storage effect node.

        Args:
            container: ONEX dependency injection container for resolving
                dependencies defined in contract.yaml.
        """
        super().__init__(container)


__all__ = ["NodeIntentStorageEffect"]
