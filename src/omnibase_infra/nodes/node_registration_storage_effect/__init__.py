# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Storage Effect - Capability-oriented storage node.

This package provides the NodeRegistrationStorageEffect, a capability-oriented
effect node for registration storage operations.

Core Principle:
    "I'm interested in what you do, not what you are"

    Named by capability (registration.storage), not by specific backend implementation.
    Supports pluggable backends through the handler protocol.

Capabilities:
    - registration.storage: Store, query, update, delete registration records
    - registration.storage.query: Query with filtering and pagination
    - registration.storage.upsert: Idempotent insert/update operations
    - registration.storage.delete: Delete by node ID
    - registration.storage.health: Backend health checks

Available Exports:
    - NodeRegistrationStorageEffect: The declarative effect node

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_registration_storage_effect import (
    ...     NodeRegistrationStorageEffect,
    ... )
    >>>
    >>> container = ModelONEXContainer()
    >>> node = NodeRegistrationStorageEffect(container)

Related Modules:
    - models: Pydantic models for storage operations
    - protocols: Handler protocol for pluggable backends
    - registry: Dependency injection registration
"""

from .node import NodeRegistrationStorageEffect

__all__ = ["NodeRegistrationStorageEffect"]
