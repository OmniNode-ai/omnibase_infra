# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Session Lifecycle Reducer — pure FSM for session lifecycle.

This package provides the NodeSessionLifecycleReducer, a pure FSM reducer
that tracks the lifecycle of pipeline runs: idle -> run_created -> run_active
-> run_ended -> idle.

Available Exports:
    - NodeSessionLifecycleReducer: The declarative reducer node
    - ModelSessionLifecycleState: Immutable FSM state model

Tracking:
    - OMN-2117: Canonical State Nodes
"""

from omnibase_infra.nodes.node_session_lifecycle_reducer.models import (
    ModelSessionLifecycleState,
)
from omnibase_infra.nodes.node_session_lifecycle_reducer.node import (
    NodeSessionLifecycleReducer,
)

__all__: list[str] = [
    "ModelSessionLifecycleState",
    "NodeSessionLifecycleReducer",
]
