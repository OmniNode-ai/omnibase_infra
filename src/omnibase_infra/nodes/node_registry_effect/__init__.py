# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registry Effect package - Alias module.

This module provides an alias to the actual implementation in
`omnibase_infra.nodes.effects.registry_effect`. The implementation follows
a domain-organized pattern where multiple effect nodes are grouped under
the `effects/` directory.

Architecture Note:
    The orchestrator's contract.yaml references `omnibase_infra.nodes.node_registry_effect`
    as the effect node module. Rather than duplicating the implementation here, this module
    re-exports from the canonical implementation location.

    Canonical Implementation: omnibase_infra.nodes.effects.registry_effect
    This Alias Module: omnibase_infra.nodes.node_registry_effect

    Both import paths work identically:
        from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect
        from omnibase_infra.nodes.effects import NodeRegistryEffect

Node Type: EFFECT
Purpose: Execute infrastructure operations (Consul registration, PostgreSQL upsert)
         based on intents from the registration orchestrator.

Implementation Details:
    - Dual-backend registration (Consul + PostgreSQL)
    - Partial failure handling with targeted retries
    - Idempotency store for retry safety
    - Error sanitization for security

Related:
    - Contract: omnibase_infra/nodes/effects/contract.yaml
    - Implementation: omnibase_infra/nodes/effects/registry_effect.py
    - Orchestrator: omnibase_infra/nodes/node_registration_orchestrator/
"""

from __future__ import annotations

# Re-export from the canonical implementation location
from omnibase_infra.nodes.effects import (
    ModelBackendResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
    NodeRegistryEffect,
)

__all__ = [
    "ModelBackendResult",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "NodeRegistryEffect",
]
