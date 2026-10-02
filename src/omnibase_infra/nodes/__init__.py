# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ONEX Infrastructure Nodes Module.

Node implementations for the ONEX 4-node architecture:
- EFFECT_GENERIC: External I/O operations (Kafka, Consul, Vault, PostgreSQL adapters)
- COMPUTE_GENERIC: Pure data transformations (compute plugins)
- REDUCER_GENERIC: State aggregation from multiple sources
- ORCHESTRATOR_GENERIC: Workflow coordination across nodes

Available Submodules:
- node_registry_effect: NodeRegistryEffect + registry models + protocols
- node_registration_reducer: Declarative FSM-driven registration reducer + RegistrationReducer
- node_registration_orchestrator: Registration workflow orchestrator
- node_auth_gate_compute: Work authorization decision compute node
- node_ledger_projection_compute: Event ledger projection compute node

Available Classes:
- NodeRegistrationReducer: Declarative FSM-driven reducer (ONEX pattern)
- RegistrationReducer: Pure function reducer implementation
- NodeRegistryEffect: Effect node for dual-backend registration execution
- NodeRegistrationOrchestrator: Workflow orchestrator for registration
- NodeAuthGateCompute: Work authorization decision compute node
- RegistryInfraAuthGateCompute: Registry for auth gate compute node
- NodeLedgerProjectionCompute: Event ledger projection compute node
- RegistryInfraLedgerProjection: Registry for ledger projection node
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.models import ModelBackendResult
    from omnibase_infra.nodes.node_auth_gate_compute import (
        NodeAuthGateCompute,
        RegistryInfraAuthGateCompute,
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
        RegistrationReducer,
        RegistryInfraNodeRegistrationReducer,
    )
    from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect
    from omnibase_infra.nodes.node_registry_effect.models import (
        ModelRegistryRequest,
        ModelRegistryResponse,
    )
    from omnibase_infra.nodes.node_session_lifecycle_reducer import (
        ModelSessionLifecycleState,
        NodeSessionLifecycleReducer,
        RegistryInfraSessionLifecycle,
    )
    from omnibase_infra.nodes.node_session_state_effect import (
        ModelRunContext,
        ModelSessionIndex,
        ModelSessionStateResult,
        NodeSessionStateEffect,
        RegistryInfraSessionState,
    )

# OMN-19444: Lazy imports keep `onex <cmd> --help` fast.
_LAZY_EXPORTS: dict[str, str] = {
    "ModelBackendResult": "omnibase_infra.models",
    "ModelRegistryRequest": "omnibase_infra.nodes.node_registry_effect.models",
    "ModelRegistryResponse": "omnibase_infra.nodes.node_registry_effect.models",
    "ModelRunContext": "omnibase_infra.nodes.node_session_state_effect",
    "ModelSessionIndex": "omnibase_infra.nodes.node_session_state_effect",
    "ModelSessionLifecycleState": "omnibase_infra.nodes.node_session_lifecycle_reducer",
    "ModelSessionStateResult": "omnibase_infra.nodes.node_session_state_effect",
    "NodeAuthGateCompute": "omnibase_infra.nodes.node_auth_gate_compute",
    "NodeLedgerProjectionCompute": "omnibase_infra.nodes.node_ledger_projection_compute",
    "NodeRegistrationOrchestrator": "omnibase_infra.nodes.node_registration_orchestrator",
    "NodeRegistrationReducer": "omnibase_infra.nodes.node_registration_reducer",
    "NodeRegistryEffect": "omnibase_infra.nodes.node_registry_effect",
    "NodeSessionLifecycleReducer": "omnibase_infra.nodes.node_session_lifecycle_reducer",
    "NodeSessionStateEffect": "omnibase_infra.nodes.node_session_state_effect",
    "RegistrationReducer": "omnibase_infra.nodes.node_registration_reducer",
    "RegistryInfraAuthGateCompute": "omnibase_infra.nodes.node_auth_gate_compute",
    "RegistryInfraLedgerProjection": "omnibase_infra.nodes.node_ledger_projection_compute",
    "RegistryInfraNodeRegistrationReducer": "omnibase_infra.nodes.node_registration_reducer",
    "RegistryInfraSessionLifecycle": "omnibase_infra.nodes.node_session_lifecycle_reducer",
    "RegistryInfraSessionState": "omnibase_infra.nodes.node_session_state_effect",
}

__all__: list[str] = [
    "ModelBackendResult",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "ModelRunContext",
    "ModelSessionIndex",
    "ModelSessionLifecycleState",
    "ModelSessionStateResult",
    "NodeAuthGateCompute",
    "NodeLedgerProjectionCompute",
    "NodeRegistrationOrchestrator",
    "NodeRegistrationReducer",
    "NodeRegistryEffect",
    "NodeSessionLifecycleReducer",
    "NodeSessionStateEffect",
    "RegistrationReducer",
    "RegistryInfraAuthGateCompute",
    "RegistryInfraLedgerProjection",
    "RegistryInfraNodeRegistrationReducer",
    "RegistryInfraSessionLifecycle",
    "RegistryInfraSessionState",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value: object = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
