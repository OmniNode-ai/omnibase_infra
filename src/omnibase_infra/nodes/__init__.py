# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Nodes Module.

This module provides node implementations for the ONEX 4-node architecture:
- EFFECT: External I/O operations (Kafka, Consul, Vault, PostgreSQL adapters)
- COMPUTE: Pure data transformations (compute plugins)
- REDUCER: State aggregation from multiple sources
- ORCHESTRATOR: Workflow coordination across nodes

Available Submodules:
- effects: Effect nodes for external I/O operations
- reducers: Reducer nodes for state aggregation

Available Classes:
- RegistrationReducer: Pure reducer for dual registration to Consul
  and PostgreSQL backends.
- RegistryEffect: Effect node for dual-backend registration execution.
"""

from omnibase_infra.nodes.effects import (
    ModelBackendResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
    RegistryEffect,
)
from omnibase_infra.nodes.reducers import RegistrationReducer

__all__ = [
    "ModelBackendResult",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "RegistrationReducer",
    "RegistryEffect",
]
