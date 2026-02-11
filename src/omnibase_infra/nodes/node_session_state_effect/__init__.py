# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Session State Effect — filesystem I/O for session state management.

This package provides the NodeSessionStateEffect, an effect node that owns
all filesystem I/O for ``~/.claude/state/``.

Capabilities:
    - session.state.index_read: Read session.json
    - session.state.index_write: Atomic write session.json (with flock)
    - session.state.run_read: Read runs/{run_id}.json
    - session.state.run_write: Atomic write runs/{run_id}.json
    - session.state.gc: Garbage-collect stale run documents (4hr TTL)

Available Exports:
    - NodeSessionStateEffect: The declarative effect node
    - ModelSessionIndex: Session index (session.json schema)
    - ModelRunContext: Run context (runs/{run_id}.json schema)
    - ModelSessionStateResult: Result of filesystem operations
    - HandlerSessionIndexRead: Read session.json
    - HandlerSessionIndexWrite: Atomic write session.json
    - HandlerRunContextRead: Read run context
    - HandlerRunContextWrite: Atomic write run context
    - HandlerStaleRunGC: GC stale run documents
    - RegistryInfraSessionState: DI registry

Tracking:
    - OMN-2117: Canonical State Nodes
"""

from omnibase_infra.nodes.node_session_state_effect.handlers import (
    HandlerRunContextRead,
    HandlerRunContextWrite,
    HandlerSessionIndexRead,
    HandlerSessionIndexWrite,
    HandlerStaleRunGC,
)
from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionIndex,
    ModelSessionStateResult,
)
from omnibase_infra.nodes.node_session_state_effect.node import NodeSessionStateEffect
from omnibase_infra.nodes.node_session_state_effect.registry import (
    RegistryInfraSessionState,
)

__all__: list[str] = [
    # Node
    "NodeSessionStateEffect",
    # Handlers
    "HandlerRunContextRead",
    "HandlerRunContextWrite",
    "HandlerSessionIndexRead",
    "HandlerSessionIndexWrite",
    "HandlerStaleRunGC",
    # Models
    "ModelRunContext",
    "ModelSessionIndex",
    "ModelSessionStateResult",
    # Registry
    "RegistryInfraSessionState",
]
