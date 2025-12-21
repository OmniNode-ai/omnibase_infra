# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Re-export module for registration orchestrator protocols.

This module provides backwards-compatible imports for ProtocolReducer and
ProtocolEffect. The protocols are now defined in separate files per ONEX
architecture rules (single protocol per file).

For new code, prefer direct imports:
    from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.protocol_reducer import (
        ProtocolReducer,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.protocol_effect import (
        ProtocolEffect,
    )

Protocol Responsibilities:
    ProtocolReducer: Pure function that computes intents from events
    ProtocolEffect: Side-effectful executor that performs infrastructure operations

Related Modules:
    - protocol_reducer: ProtocolReducer definition
    - protocol_effect: ProtocolEffect definition
    - omnibase_infra.models.registration: Event models
    - omnibase_infra.nodes.node_registry_effect: Effect implementation
"""

from __future__ import annotations

# Re-export models for backwards compatibility with existing imports
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_registration_intent import (
    ModelRegistrationIntent,
)

# Re-export protocols from their dedicated files
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.protocol_effect import (
    ProtocolEffect,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.protocol_reducer import (
    ProtocolReducer,
)

__all__ = [
    "ModelReducerState",
    "ModelRegistrationIntent",
    "ProtocolEffect",
    "ProtocolReducer",
]
