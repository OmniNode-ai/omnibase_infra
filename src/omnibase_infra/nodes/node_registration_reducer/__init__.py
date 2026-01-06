# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""NodeRegistrationReducer - FSM-driven declarative reducer for node registration.

This module exports the declarative NodeRegistrationReducer that processes
node introspection events and emits registration intents for Consul and
PostgreSQL backends.

Architecture:
    The NodeRegistrationReducer follows the ONEX declarative pattern:
    - Extends NodeReducer from omnibase_core
    - Uses FSM state machine defined in contract.yaml
    - All state transitions driven by contract, not Python code
    - Emits registration intents for Effect layer execution

Available Exports:
    - NodeRegistrationReducer: The declarative FSM-driven reducer node
    - RegistryInfraNodeRegistrationReducer: Registry for dependency injection

Models (from .models):
    - ModelValidationResult: Validation result with error details
    - ModelRegistrationState: Immutable state for FSM
    - ModelRegistrationConfirmation: Confirmation events from Effect layer
    - ModelPayloadConsulRegister: Consul registration intent payload
    - ModelPayloadPostgresUpsertRegistration: PostgreSQL upsert intent payload

Related:
    - contract.yaml: FSM state machine and configuration
    - OMN-889: Infrastructure MVP
    - OMN-1104: Declarative refactoring ticket
"""

from omnibase_infra.nodes.node_registration_reducer.node import (
    NodeRegistrationReducer,
)
from omnibase_infra.nodes.node_registration_reducer.registry import (
    RegistryInfraNodeRegistrationReducer,
)

__all__ = [
    "NodeRegistrationReducer",
    "RegistryInfraNodeRegistrationReducer",
]
