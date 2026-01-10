# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for NodeRegistryEffect.

This package contains node-specific Pydantic models for the registry effect node.
Models follow ONEX naming conventions: Model<Name>.

Shared models are re-exported from the effects package for convenience.
Node-specific models are defined here.

Model Categories:
    - Input Models: Request payloads for effect operations
    - Output Models: Response structures from effect operations
    - Result Models: Per-backend operation outcomes
    - Config Models: Node configuration schemas

Related:
    - omnibase_infra.nodes.effects.models: Shared effect models
    - contract.yaml: Model references in input_model/output_model
"""

from __future__ import annotations

# Re-export shared effect models for convenience
from omnibase_infra.nodes.effects.models import (
    ModelBackendResult,
    ModelEffectIdempotencyConfig,
    ModelRegistryRequest,
    ModelRegistryResponse,
)

# Node-specific models
from omnibase_infra.nodes.node_registry_effect.models.model_partial_retry_request import (
    ModelPartialRetryRequest,
)

__all__ = [
    "ModelBackendResult",
    "ModelEffectIdempotencyConfig",
    "ModelPartialRetryRequest",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
]
