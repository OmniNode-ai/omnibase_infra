# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry package for NodeRegistryEffect.

This package provides infrastructure registry components for the
NodeRegistryEffect node, following ONEX naming conventions.

Exports:
    RegistryInfraRegistryEffect: Factory and metadata registry for
        creating NodeRegistryEffect instances with dependency injection.

Usage:
    >>> from omnibase_infra.nodes.node_registry_effect.registry import (
    ...     RegistryInfraRegistryEffect,
    ... )
    >>> effect = RegistryInfraRegistryEffect.create(container)

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_effect.registry.registry_infra_registry_effect import (
    RegistryInfraRegistryEffect,
)

__all__ = ["RegistryInfraRegistryEffect"]
