# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Intent Storage Effect - Capability-oriented intent storage node.

This package provides the NodeIntentStorageEffect, a capability-oriented
effect node for intent storage operations using Memgraph graph database.

Core Principle:
    "I'm interested in what you do, not what you are"

    Named by capability (intent.storage), not by specific backend implementation.
    Uses Memgraph for graph-based intent persistence.

Capabilities:
    - intent.storage: Store, query, and analyze intents
    - intent.storage.store: Store classified intents as graph nodes
    - intent.storage.query_session: Query intents by session identifier
    - intent.storage.query_distribution: Get intent distribution statistics

Event Topics:
    Consumed:
        - onex.evt.platform.intent-classified.v1
        - onex.cmd.platform.intent-query-session.v1
        - onex.cmd.platform.intent-query-distribution.v1
    Published:
        - onex.evt.platform.intent-stored.v1
        - onex.evt.platform.intent-session-query-result.v1
        - onex.evt.platform.intent-distribution-result.v1

Available Exports:
    - NodeIntentStorageEffect: The declarative effect node

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_intent_storage_effect import (
    ...     NodeIntentStorageEffect,
    ... )
    >>>
    >>> container = ModelONEXContainer()
    >>> node = NodeIntentStorageEffect(container)

Related Modules:
    - models: Pydantic models for intent storage operations
    - registry: Dependency injection registration
    - HandlerIntent: Handler for graph operations (omnibase_infra.handlers)
"""

from .node import NodeIntentStorageEffect

__all__ = ["NodeIntentStorageEffect"]
