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
- NodeDualRegistrationReducer: Coordinates parallel registration to Consul
  and PostgreSQL backends with graceful degradation for partial failures.
"""

from omnibase_infra.nodes.reducers.node_dual_registration_reducer import (
    NodeDualRegistrationReducer,
)

__all__ = ["NodeDualRegistrationReducer"]
