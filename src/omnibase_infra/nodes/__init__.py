# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Nodes Module.

This module provides node implementations for the ONEX 4-node architecture:
- EFFECT_GENERIC: External I/O operations (Kafka, Consul, Vault, PostgreSQL adapters)
- COMPUTE_GENERIC: Pure data transformations (compute plugins)
- REDUCER_GENERIC: State aggregation from multiple sources
- ORCHESTRATOR_GENERIC: Workflow coordination across nodes

Available Submodules:
- effects: Effect nodes for external I/O operations
- reducers: Reducer nodes for state aggregation
- node_registration_reducer: Declarative FSM-driven registration reducer
- node_registration_orchestrator: Registration workflow orchestrator
- node_ledger_projection_compute: Event ledger projection compute node

Available Classes:
- NodeRegistrationReducer: Declarative FSM-driven reducer (ONEX pattern)
- RegistrationReducer: Pure function reducer implementation
- NodeRegistryEffect: Effect node for dual-backend registration execution
- NodeRegistrationOrchestrator: Workflow orchestrator for registration
- NodeLedgerProjectionCompute: Event ledger projection compute node
- RegistryInfraLedgerProjection: Registry for ledger projection node
"""

from omnibase_infra.nodes.effects import (
    ModelBackendResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
    NodeRegistryEffect,
)
from omnibase_infra.nodes.node_ledger_projection_compute import (
    NodeLedgerProjectionCompute,
    RegistryInfraLedgerProjection,
)
from omnibase_infra.nodes.node_registration_orchestrator import (
    NodeRegistrationOrchestrator,
)
from omnibase_infra.nodes.node_registration_reducer import (
    NodeRegistrationReducer,
    RegistryInfraNodeRegistrationReducer,
)
from omnibase_infra.nodes.reducers import RegistrationReducer

__all__: list[str] = [
    "ModelBackendResult",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "NodeLedgerProjectionCompute",
    "NodeRegistrationOrchestrator",
    "NodeRegistrationReducer",
    "NodeRegistryEffect",
    "RegistrationReducer",
    "RegistryInfraLedgerProjection",
    "RegistryInfraNodeRegistrationReducer",
]
