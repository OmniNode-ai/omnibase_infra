# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Registry API Effect — contract-driven wrapper for the Registry API.

Refactors the standalone FastAPI Registry API into a proper ONEX EFFECT node,
sourcing all operational configuration from ``contract.yaml`` rather than
hardcoded module-level constants.

Ticket: OMN-1441

Architecture:
    - node.py: Declarative node shell extending NodeEffect
    - contract.yaml: All config, I/O model definitions, and handler routing
    - registry/: Infrastructure registry for dependency injection
    - models/: Node-specific Pydantic input/output envelopes

Usage:
    >>> from omnibase_infra.nodes.node_registry_api_effect import (
    ...     NodeRegistryApiEffect,
    ...     RegistryInfraRegistryApiEffect,
    ... )
    >>> cfg = NodeRegistryApiEffect.get_config()
    >>> max_fetch = cfg["max_node_type_filter_fetch"]
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_api_effect.node import (
    NodeRegistryApiEffect,
    load_registry_api_config,
)
from omnibase_infra.nodes.node_registry_api_effect.registry import (
    RegistryInfraRegistryApiEffect,
)

__all__: list[str] = [
    "NodeRegistryApiEffect",
    "RegistryInfraRegistryApiEffect",
    "load_registry_api_config",
]
