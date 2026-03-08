# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for NodeRegistryEffect.

This package contains node-specific Pydantic models for the registry effect node.
Models follow ONEX naming conventions: Model<Name>.

All models are now local to this package (OMN-3989: migrated from nodes.effects.models).

Model Categories:
    - Input Models: Request payloads for effect operations
    - Output Models: Response structures from effect operations
    - Result Models: Per-backend operation outcomes
    - Config Models: Node configuration schemas

Related:
    - contract.yaml: Model references in input_model/output_model
"""

from __future__ import annotations

# Re-export shared models for convenience
from omnibase_infra.models import ModelBackendResult
from omnibase_infra.nodes.node_registry_effect.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.node_registry_effect.models.model_partial_retry_request import (
    ModelPartialRetryRequest,
)
from omnibase_infra.nodes.node_registry_effect.models.model_registry_request import (
    ModelRegistryRequest,
)
from omnibase_infra.nodes.node_registry_effect.models.model_registry_response import (
    ModelRegistryResponse,
)

__all__ = [
    "ModelBackendResult",
    "ModelEffectIdempotencyConfig",
    "ModelPartialRetryRequest",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
]
