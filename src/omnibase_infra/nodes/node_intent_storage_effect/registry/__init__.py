# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for Intent Storage Effect Node.

This module exports the RegistryInfraIntentStorage for dependency registration
with the ONEX container. The registry wires up the NodeIntentStorageEffect
with its handler dependencies (HandlerIntent -> HandlerGraph -> Memgraph).

Architecture:
    The registry follows the ONEX container-based dependency injection pattern:
    - Registers NodeIntentStorageEffect with the container
    - Wires handler dependencies defined in contract.yaml
    - Enables capability-oriented resolution (intent.storage)

    Registration Flow:
        Container -> RegistryInfraIntentStorage.register() -> NodeIntentStorageEffect
                                                           -> HandlerIntent (resolved)

Usage:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_intent_storage_effect.registry import (
    ...     RegistryInfraIntentStorage,
    ... )
    >>> container = ModelONEXContainer()
    >>> RegistryInfraIntentStorage.register(container)

Related:
    - NodeIntentStorageEffect: The effect node registered by this registry
    - HandlerIntent: Handler wired for intent storage operations
    - ModelONEXContainer: ONEX dependency injection container
"""

from .registry_infra_intent_storage import RegistryInfraIntentStorage

__all__ = ["RegistryInfraIntentStorage"]
