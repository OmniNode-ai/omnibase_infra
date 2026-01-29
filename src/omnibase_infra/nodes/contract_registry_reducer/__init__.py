# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Registry Reducer Node.

This module provides the declarative reducer node for projecting contract
registration events to PostgreSQL. The reducer is FSM-driven and follows
the ONEX declarative pattern.

Exports:
    NodeContractRegistryReducer: Declarative reducer node shell.
    ContractRegistryReducer: Pure function reducer class.
    ModelContractRegistryState: Immutable state model for the reducer.
"""

from omnibase_infra.nodes.contract_registry_reducer.models import (
    ModelContractRegistryState,
)
from omnibase_infra.nodes.contract_registry_reducer.node import (
    NodeContractRegistryReducer,
)
from omnibase_infra.nodes.contract_registry_reducer.reducer import (
    ContractRegistryReducer,
)

__all__ = [
    "ContractRegistryReducer",
    "ModelContractRegistryState",
    "NodeContractRegistryReducer",
]
