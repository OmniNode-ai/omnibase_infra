# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Effect nodes for ONEX infrastructure.

This module exports Effect layer nodes responsible for external I/O operations
such as Consul registration and PostgreSQL persistence.

Available:
    - NodeRegistryEffect: Dual-backend registration effect node
    - ModelRegistryResponse: Response model for registry operations
    - ModelBackendResult: Individual backend result model
"""

from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models import (
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.nodes.effects.registry_effect import NodeRegistryEffect

__all__ = [
    "ModelBackendResult",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "NodeRegistryEffect",
]
