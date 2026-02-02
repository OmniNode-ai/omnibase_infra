# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry package for NodeContractPersistenceEffect.

This package provides infrastructure registry components for the
NodeContractPersistenceEffect node, following ONEX naming conventions.

Exports:
    RegistryInfraContractPersistenceEffect: Factory and metadata registry for
        creating NodeContractPersistenceEffect instances with dependency injection.

Usage:
    >>> from omnibase_infra.nodes.node_contract_persistence_effect.registry import (
    ...     RegistryInfraContractPersistenceEffect,
    ... )
    >>> effect = RegistryInfraContractPersistenceEffect.create(container)

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from omnibase_infra.nodes.node_contract_persistence_effect.registry.registry_infra_contract_persistence_effect import (
    RegistryInfraContractPersistenceEffect,
)

__all__ = ["RegistryInfraContractPersistenceEffect"]
