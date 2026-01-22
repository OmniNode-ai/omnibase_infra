# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Storage Effect - Capability-oriented storage node.

This effect node provides registration storage capabilities with pluggable backends.
Named by capability ("registration.storage"), not by vendor (e.g., PostgreSQL).

Core Principle:
    "I'm interested in what you do, not what you are"

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom storage logic - all behavior from handler protocol
    - Lightweight shell that delegates to ProtocolRegistrationPersistence
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for external I/O operations.
All storage logic is 100% driven by handler implementations, not Python code.

Capabilities:
    - registration.storage: Store, query, update, delete registration records
    - registration.storage.query: Query with filtering and pagination
    - registration.storage.upsert: Idempotent insert/update operations
    - registration.storage.delete: Delete by node ID
    - registration.storage.health: Backend health checks

Pluggable Backends:
    The node supports multiple storage backends through the handler protocol:
    - PostgreSQL (default): For relational storage requirements
    - Mock: For testing and development

    Backend configuration is done through the registry at bootstrap:
    ```python
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.node_registration_storage_effect import (
        NodeRegistrationStorageEffect,
    )
    from omnibase_infra.nodes.node_registration_storage_effect.registry import (
        RegistryInfraRegistrationStorage,
    )
    from omnibase_infra.handlers.registration_storage import (
        HandlerRegistrationStoragePostgres,
    )

    # Create container and register handler
    container = ModelONEXContainer()
    handler = HandlerRegistrationStoragePostgres(
        container=container,
        dsn="postgresql://...",
    )
    RegistryInfraRegistrationStorage.register(container)
    RegistryInfraRegistrationStorage.register_handler(container, handler)

    # Create node with container (handler resolved via DI)
    node = NodeRegistrationStorageEffect(container)
    ```

Design Decisions:
    - 100% Contract-Driven: All capabilities in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Handler wired externally
    - Capability-Oriented: Named by what it does, not what it uses

Related:
    - contract.yaml: Capability definitions and IO operations
    - ProtocolRegistrationPersistence: Handler protocol for backends
    - models/: Input, output, and record models
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeRegistrationStorageEffect(NodeEffect):
    """Effect node for registration storage operations.

    Capability: registration.storage

    Provides a capability-oriented interface for registration storage operations.
    Supports pluggable handlers for different storage backends (PostgreSQL, mock).

    This node is declarative - all behavior is defined in contract.yaml and
    implemented through the handler protocol. No custom storage logic exists
    in this class.

    Attributes:
        container: ONEX dependency injection container

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> node = NodeRegistrationStorageEffect(container)
        >>> # Handler must be wired externally via registry
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registration storage effect node.

        Args:
            container: ONEX dependency injection container for resolving
                dependencies defined in contract.yaml.
        """
        super().__init__(container)


__all__ = ["NodeRegistrationStorageEffect"]
