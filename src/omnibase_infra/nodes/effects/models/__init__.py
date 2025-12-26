# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Effect nodes.

This module exports models used by Effect layer nodes for external I/O operations.

Available Models:
    - ModelBackendResult: Individual backend operation result
    - ModelEffectIdempotencyConfig: Configuration for effect idempotency store
    - ModelRegistryRequest: Registry effect input request
    - ModelRegistryResponse: Dual-backend registry operation response
"""

from omnibase_infra.nodes.effects.models.model_backend_result import (
    ModelBackendResult,
)
from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.models.model_registry_request import (
    ModelRegistryRequest,
)
from omnibase_infra.nodes.effects.models.model_registry_response import (
    ModelRegistryResponse,
)

__all__ = [
    "ModelBackendResult",
    "ModelEffectIdempotencyConfig",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
]
