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
- RegistrationReducer: Canonical pure reducer for dual registration workflow.
  Uses ModelReducerOutput from omnibase_core. (~80 lines, stateless)
"""

from omnibase_infra.nodes.reducers.registration_reducer import RegistrationReducer

__all__ = [
    "RegistrationReducer",
]
