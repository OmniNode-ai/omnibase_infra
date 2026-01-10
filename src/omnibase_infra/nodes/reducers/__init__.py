# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Reducers Module.

This module provides reducer nodes for aggregating and consolidating state
from multiple sources in the ONEX 4-node architecture.

Reducers are responsible for:
- State aggregation from multiple sources
- Event sourcing and state reconstruction
- Multi-source data consolidation
- Dual registration coordination (Consul + PostgreSQL)

Available Reducers:
- NodeRegistrationReducer: Declarative FSM-driven reducer (ONEX pattern).
  Located at: nodes/node_registration_reducer/node.py
  Uses contract.yaml for FSM state transitions.

- RegistrationReducer: Pure function reducer implementation.
  Uses ModelReducerOutput from omnibase_core.

Related:
    - OMN-1104: Refactor RegistrationReducer to be fully declarative
"""

from omnibase_infra.nodes.reducers.registration_reducer import RegistrationReducer

__all__ = [
    "RegistrationReducer",
]
